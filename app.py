import os
import json
import streamlit as st
import time
import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from urllib.parse import unquote
from galileo_api_helper import get_galileo_project_id, get_galileo_log_stream_id, list_galileo_experiments, delete_all_galileo_experiments
from chat_lib.galileo_logger import initialize_galileo_logger

# Import tools
from log_hallucination import log_hallucination
from tools.get_ticker_symbol import get_ticker_symbol
from tools.get_stock_price import get_stock_price
from tools.purchase_stocks import purchase_stocks

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger_debug = logging.getLogger(__name__)

os.environ["GALILEO_API_KEY"] = st.secrets["galileo_api_key"]
os.environ["GALILEO_PROJECT_NAME"] = st.secrets["galileo_project"]
os.environ["GALILEO_LOG_STREAM_NAME"] = st.secrets["galileo_log_stream"]
os.environ["GALILEO_CONSOLE_URL"] = st.secrets["galileo_console_url"]
# Initialize OpenAI client
logger_debug.info("Initializing OpenAI client")
openai_client = OpenAI(
    api_key=st.secrets["openai_api_key"]
)
logger_debug.debug(f"OpenAI API Key loaded: {'*' * 8}{st.secrets['openai_api_key'][-4:] if st.secrets['openai_api_key'] else 'Not found'}")

# Initialize Pinecone
logger_debug.info("Initializing Pinecone client")
pc = Pinecone(
    api_key=st.secrets["pinecone_api_key"],
    spec=ServerlessSpec(cloud="aws", region="us-west-2")
)
logger_debug.debug(f"Pinecone API Key loaded: {'*' * 8}{st.secrets['pinecone_api_key'][-4:] if st.secrets['pinecone_api_key'] else 'Not found'}")

# Define RAG response type
class RagResponse:
    def __init__(self, documents: List[Dict[str, Any]]):
        self.documents = documents

async def get_rag_response(query: str, namespace: str, top_k: int) -> Optional[RagResponse]:
    """Get RAG response using Pinecone vector store."""
    try:
        logger_debug.info(f"Making RAG request - Query: {query}, Namespace: {namespace}, Top K: {top_k}")
        
        # Get embeddings for the query
        logger_debug.info("Getting embeddings for query")
        embedding_response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = embedding_response.data[0].embedding
        logger_debug.debug(f"Generated embedding of length: {len(query_embedding)}")
        
        # Initialize Pinecone index
        index_name = st.secrets["pinecone_index_name"]
        logger_debug.debug(f"Using Pinecone index: {index_name}")
        if not index_name:
            logger_debug.error("PINECONE_INDEX_NAME environment variable is not set")
            return None
            
        index = pc.Index(index_name)
        
        # Query Pinecone
        logger_debug.info("Querying Pinecone index")
        try:
            query_response = index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=namespace if namespace and namespace != "" else None,
                include_metadata=True
            )
            # Log query response in a serializable format
            logger_debug.debug("Pinecone query response:")
            logger_debug.debug(f"Number of matches: {len(query_response.matches)}")
            if query_response.matches:
                for i, match in enumerate(query_response.matches):
                    logger_debug.debug(f"Match {i + 1}:")
                    logger_debug.debug(f"  ID: {match.id}")
                    logger_debug.debug(f"  Score: {match.score}")
                    logger_debug.debug(f"  Metadata: {json.dumps(match.metadata, indent=2)}")
        except Exception as e:
            logger_debug.error(f"Error querying Pinecone: {str(e)}", exc_info=True)
            return None
        
        # Process results
        if not query_response.matches:
            logger_debug.warning("No matches found in Pinecone")
            return None
            
        logger_debug.info(f"Found {len(query_response.matches)} matches in Pinecone")
        logger_debug.debug(f"First match score: {query_response.matches[0].score}")
        
        # Log the full metadata structure of the first match
        if query_response.matches:
            first_match = query_response.matches[0]
            logger_debug.info("Metadata structure of first match:")
            logger_debug.debug(f"Full metadata: {json.dumps(first_match.metadata, indent=2)}")
            logger_debug.debug(f"Available metadata keys: {list(first_match.metadata.keys())}")
            logger_debug.debug(f"Match ID: {first_match.id}")
            logger_debug.debug(f"Match score: {first_match.score}")
        
        # Format documents
        documents = [
            {
                "content": match.metadata.get("text", ""),
                "metadata": {
                    "score": match.score,
                    **match.metadata  # Include all metadata fields
                }
            }
            for match in query_response.matches
        ]
        
        logger_debug.info(f"Formatted {len(documents)} documents for response")
        if documents:
            logger_debug.debug(f"First document content preview: {documents[0]['content'][:200]}")
        
        return RagResponse(documents=documents)
        
    except Exception as e:
        logger_debug.error(f"Error in RAG request: {str(e)}", exc_info=True)
        return None

# Define tools
tools = {
    "getTickerSymbol": {
        "description": "Get the ticker symbol for a company",
        "parameters": {
            "type": "object",
            "properties": {
                "company": {
                    "type": "string",
                    "description": "The name of the company"
                }
            },
            "required": ["company"]
        }
    },
    "purchaseStocks": {
        "description": "Purchase a specified number of shares of a stock at a given price.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol to purchase"
                },
                "quantity": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "The number of shares to purchase"
                },
                "price": {
                    "type": "number",
                    "minimum": 0.01,
                    "description": "The price per share at which to purchase"
                }
            },
            "required": ["ticker", "quantity", "price"]
        }
    },
    "getStockPrice": {
        "description": "Get the current stock price and other market data for a given ticker symbol",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol to look up"
                }
            },
            "required": ["ticker"]
        }
    }
}

# Format tools for OpenAI API
openai_tools = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": tool["description"],
            "parameters": tool["parameters"]
        }
    }
    for name, tool in tools.items()
]

def escape_dollar_signs(text: str) -> str:
    """Escape dollar signs in text to prevent LaTeX interpretation."""
    return text.replace('$', '\\$')

def format_message(role: str, content: str = None, tool_calls=None, tool_call_id=None) -> dict:
    """Format a message for the chat.
    
    Args:
        role: The role of the message (system, user, assistant, tool)
        content: The content of the message
        tool_calls: Tool calls for assistant messages
        tool_call_id: Tool call ID for tool messages
        
    Returns:
        A properly formatted message dictionary
    """
    message = {"role": role}
    
    if content is not None:
        message["content"] = content
        
    if role == "assistant" and tool_calls is not None:
        message["tool_calls"] = [{
            "id": tool_call.get("id", f"toolcall-{i}"),
            "type": tool_call.get("type", "function"),
            "function": {
                "name": tool_call.get("function", {}).get("name", ""),
                "arguments": tool_call.get("function", {}).get("arguments", "{}")
            }
        } for i, tool_call in enumerate(tool_calls)]
        
    if role == "tool" and tool_call_id is not None:
        message["tool_call_id"] = tool_call_id
        
    return message

def handle_tool_call(tool_call, tool_result, description, messages_to_use, logger):
    """Handle a tool call and its response.
    
    Args:
        tool_call: The tool call object from OpenAI
        tool_result: The result from executing the tool
        description: Human-readable description of what the tool is doing
        messages_to_use: The message history to append to
        logger: The Galileo logger
    """
    # Create tool call data
    tool_call_data = {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments
        }
    }
    
    # Add tool response to messages
    messages_to_use.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [tool_call_data]
    })
    
    messages_to_use.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": tool_result
    })
    
    # Display the tool response in chat
    st.session_state.messages.append(format_message(
        role="assistant", 
        content=description,
        tool_calls=[tool_call_data]
    ))
    
    st.session_state.messages.append(format_message(
        role="tool", 
        content=tool_result,
        tool_call_id=tool_call.id
    ))
    
    with st.chat_message("assistant"):
        st.markdown(escape_dollar_signs(description))
    
    with st.chat_message("tool"):
        st.markdown(escape_dollar_signs(tool_result))

async def process_chat_message(
    prompt: str,
    message_history: List[Dict[str, Any]], 
    model: str = "gpt-4", 
    system_prompt: str = None,
    use_rag: bool = True,
    namespace: str = "sp500-qa-demo",
    top_k: int = 10,
    galileo_logger = None
) -> Dict[str, Any]:
    """Process a chat message independently of Streamlit UI.
    
    Args:
        prompt (str): The user's message/prompt
        message_history (List[Dict[str, Any]]): Previous message history
        model (str): The OpenAI model to use
        system_prompt (str): System prompt to use
        use_rag (bool): Whether to use RAG for context
        namespace (str): Namespace for RAG
        top_k (int): Number of top documents to retrieve for RAG
        galileo_logger: Optional Galileo logger for observability
        
    Returns:
        Dict containing:
            - response_message: The final response message from the model
            - updated_history: The updated message history
            - rag_documents: Any RAG documents retrieved (if RAG was used)
    """
    
    start_time = time.time()
    logger_debug.info(f"Processing chat message: {prompt}")
    
    # Start Galileo trace if available
    if galileo_logger:
        logger_debug.info("Starting new Galileo trace")
        trace = galileo_logger.start_trace(
            input=prompt,
            name="Chat Workflow",
            tags=["chat"],
        )
    
    try:
        # Copy message history to avoid modifying the original
        messages_to_use = message_history.copy()
        
        # Add user message to history
        messages_to_use.append(format_message("user", prompt))
        
        rag_documents = []
        
        # Handle RAG if enabled
        if use_rag:
            logger_debug.info("RAG enabled, fetching relevant documents")
            rag_response = await get_rag_response(prompt, namespace, top_k)
            
            if rag_response and rag_response.documents:
                logger_debug.info(f"RAG returned {len(rag_response.documents)} documents")
                rag_documents = rag_response.documents
                
                # Log RAG retrieval to Galileo if available
                if galileo_logger:
                    galileo_logger.add_retriever_span(
                        input=prompt,
                        output=[doc['content'] for doc in rag_response.documents],
                        name="RAG Retriever",
                        duration_ns=int((time.time() - start_time) * 1000000),
                        metadata={
                            "document_count": str(len(rag_response.documents)),
                            "namespace": namespace
                        }
                    )
                
                # Add context to system message
                context = "\n\n".join(doc['content'] for doc in rag_response.documents)
                logger_debug.debug(f"Adding RAG context to messages: {context[:200]}...")
                
                messages_to_use = [
                    {
                        "role": "system",
                        "content": f"{system_prompt}\n\nHere is the relevant context that you should use to answer the user's questions:\n\n{context}\n\nMake sure to use this context when answering questions."
                    },
                    *messages_to_use
                ]
            else:
                logger_debug.warning("No RAG documents found for query")
                if system_prompt:
                    messages_to_use = [
                        {"role": "system", "content": system_prompt},
                        *messages_to_use
                    ]
        elif system_prompt:
            logger_debug.info("Adding system prompt without RAG")
            messages_to_use = [
                {"role": "system", "content": system_prompt},
                *messages_to_use
            ]
        
        # Get response from OpenAI
        logger_debug.info(f"Calling OpenAI API with model {model}")
        logger_debug.debug(f"Messages being sent to OpenAI: {json.dumps([format_message(msg['role'], msg['content']) for msg in messages_to_use], indent=2)}")
        logger_debug.debug(f"Tools being sent to OpenAI: {json.dumps(openai_tools, indent=2)}")
        
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages_to_use,
            tools=openai_tools,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        logger_debug.info("Received response from OpenAI")
        
        # Calculate token counts safely
        input_tokens = len(prompt.split()) if prompt else 0
        output_tokens = len(response_message.content.split()) if response_message.content else 0
        total_tokens = input_tokens + output_tokens
        
        # Log the API call to Galileo if available
        if galileo_logger:
            logger_debug.info("Logging API call to Galileo")
            galileo_logger.add_llm_span(
                input=[format_message(msg["role"], msg["content"]) for msg in messages_to_use],
                output={
                    "role": response_message.role,
                    "content": response_message.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": call.type,
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments
                            }
                        } for call in (response_message.tool_calls or [])
                    ] if response_message.tool_calls else None
                },
                model=model,
                name="OpenAI API Call",
                tools=[{"name": name, "parameters": list(tool["parameters"]["properties"].keys())} 
                      for name, tool in tools.items()],
                duration_ns=int((time.time() - start_time) * 1000000),
                metadata={"temperature": "0.7", "model": model},
                tags=["api-call"],
                num_input_tokens=input_tokens,
                num_output_tokens=output_tokens,
                total_tokens=total_tokens
            )
        
        # Handle tool calls if present
        tool_results = []
        if response_message.tool_calls:
            logger_debug.info("Processing tool calls")
            continue_conversation = True
            
            while continue_conversation and response_message.tool_calls:
                current_tool_calls = []
                
                # Process each tool call and its response
                for tool_call in response_message.tool_calls:
                    tool_result = None
                    
                    if tool_call.function.name == "getTickerSymbol":
                        company = json.loads(tool_call.function.arguments)["company"]
                        ticker = get_ticker_symbol(company, galileo_logger)
                        logger_debug.info(f"Got ticker symbol for {company}: {ticker}")
                        tool_result = ticker
                        description = f"Looking up ticker symbol for {company}..."
                        
                    elif tool_call.function.name == "getStockPrice":
                        ticker = json.loads(tool_call.function.arguments)["ticker"]
                        result = get_stock_price(ticker, galileo_logger=galileo_logger)
                        logger_debug.info(f"Got stock price for {ticker}")
                        tool_result = result
                        description = f"Getting current price for {ticker}..."
                        
                    elif tool_call.function.name == "purchaseStocks":
                        args = json.loads(tool_call.function.arguments)
                        result = purchase_stocks(
                            ticker=args["ticker"],
                            quantity=args["quantity"],
                            price=args["price"],
                            galileo_logger=galileo_logger
                        )
                        logger_debug.info(f"Processed stock purchase for {args['ticker']}")
                        tool_result = result
                        description = f"Processing purchase of {args['quantity']} shares of {args['ticker']}..."
                    
                    if tool_result:
                        # Create tool call data for tracking
                        current_tool_calls.append({
                            "tool_call": tool_call,
                            "result": tool_result,
                            "description": description
                        })
                        
                        # Add tool call and response to messages
                        messages_to_use.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            }]
                        })
                        
                        messages_to_use.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result
                        })
                        
                        # Track all tool results for return
                        tool_results.append({
                            "name": tool_call.function.name,
                            "arguments": json.loads(tool_call.function.arguments),
                            "result": tool_result,
                            "description": description
                        })
                
                # Get a new response from OpenAI with the tool results
                logger_debug.info("Getting follow-up response with tool results")
                follow_up_response = openai_client.chat.completions.create(
                    model=model,
                    messages=messages_to_use,
                    tools=openai_tools,
                    tool_choice="auto"
                )
                
                response_message = follow_up_response.choices[0].message
                logger_debug.debug(f"Received follow-up response from OpenAI")
                
                # Calculate token counts for follow-up response
                follow_up_input_tokens = sum(len(msg.get("content", "").split()) for msg in messages_to_use if msg.get("content"))
                follow_up_output_tokens = len(response_message.content.split()) if response_message.content else 0
                follow_up_total_tokens = follow_up_input_tokens + follow_up_output_tokens
                
                # Log the follow-up API call to Galileo if available
                if galileo_logger:
                    logger_debug.info("Logging follow-up API call to Galileo")
                    galileo_logger.add_llm_span(
                        input=[format_message(msg["role"], msg["content"]) for msg in messages_to_use],
                        output={
                            "role": response_message.role,
                            "content": response_message.content,
                            "tool_calls": [
                                {
                                    "id": call.id,
                                    "type": call.type,
                                    "function": {
                                        "name": call.function.name,
                                        "arguments": call.function.arguments
                                    }
                                } for call in (response_message.tool_calls or [])
                            ] if response_message.tool_calls else None
                        },
                        model=model,
                        name="Follow-up OpenAI API Call",
                        tools=[{"name": name, "parameters": list(tool["parameters"]["properties"].keys())} 
                               for name, tool in tools.items()],
                        duration_ns=int((time.time() - start_time) * 1000000),
                        metadata={"temperature": "0.7", "model": model},
                        tags=["api-call", "follow-up"],
                        num_input_tokens=follow_up_input_tokens,
                        num_output_tokens=follow_up_output_tokens,
                        total_tokens=follow_up_total_tokens
                    )
                
                # If no more tool calls, end the conversation
                if not response_message.tool_calls:
                    continue_conversation = False
        
        # Add final assistant response to history
        if response_message.content:
            messages_to_use.append(format_message("assistant", response_message.content))
        
        # Conclude the Galileo trace if available
        if galileo_logger:
            logger_debug.info("Concluding Galileo trace")
            galileo_logger.conclude(
                output=response_message.content,
                duration_ns=int((time.time() - start_time) * 1000000),
                status_code=200
            )
            galileo_logger.flush()
        
        # Return the results
        return {
            "response_message": response_message,
            "updated_history": messages_to_use,
            "rag_documents": rag_documents,
            "tool_results": tool_results,
            "total_tokens": total_tokens
        }
        
    except Exception as e:
        logger_debug.error(f"Error processing chat message: {str(e)}", exc_info=True)
        
        # Log error to Galileo if available
        if galileo_logger:
            logger_debug.info("Logging error to Galileo")
            galileo_logger.conclude(
                output=f"Error: {str(e)}",
                duration_ns=int((time.time() - start_time) * 1000000),
                status_code=500
            )
        
        # Re-raise the exception
        raise

async def main():
    st.title("RAG Chat Application")
    logger_debug.info("Starting Streamlit application")
    
    # Get query parameters
    default_project = unquote(st.query_params.get("project", st.secrets["galileo_project"]))
    default_log_stream = unquote(st.query_params.get("log_stream", st.secrets["galileo_log_stream"]))
    
    # Initialize session state variables if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_active" not in st.session_state:
        st.session_state.session_active = False
    
    if "galileo_session_id" not in st.session_state:
        st.session_state.galileo_session_id = None
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        if not "galileo_logger" in st.session_state:
            # Add Galileo configuration fields
            st.subheader("Galileo Configuration")
            galileo_project = st.text_input(
                "Galileo Project",
                value=default_project,
                help="The name of your Galileo project"
            )
            galileo_log_stream = st.text_input(
                "Galileo Log Stream",
                value=default_log_stream,
                help="The name of your Galileo log stream"
            )

            galileo_api_key = st.text_input(
                "Galileo API Key",
                value=st.secrets["galileo_api_key"],
                help="The API key for your Galileo project"
            )

            galileo_console_url = st.text_input(
                "Galileo Console URL",
                value=st.secrets["galileo_console_url"],
                help="The URL of your Galileo console"
            )
        

        # Add model selection dropdown
        st.subheader("Model Configuration")
        model_option = st.selectbox(
            "Select GPT Model",
            options=["gpt-4", "gpt-3.5-turbo"],
            index=0,  # Default to GPT-4
            format_func=lambda x: "GPT-4" if x == "gpt-4" else "GPT-3.5 Turbo",
            help="Select which OpenAI model to use for chat responses"
        )
        logger_debug.debug(f"Selected model: {model_option}")
        

        # Session control buttons
        if not st.session_state.session_active:
            # Show Start Session button when no active session
            if st.button("Start New Session", type="primary"):

                os.environ["GALILEO_API_KEY"] = galileo_api_key
                os.environ["GALILEO_CONSOLE_URL"] = galileo_console_url

                st.session_state.galileo_logger = initialize_galileo_logger(galileo_project, galileo_log_stream)
                logger = st.session_state.galileo_logger

                st.session_state.session_active = True
                st.session_state.messages = []  # Clear any previous messages
                
                # Start a new Galileo session
                logger_debug.info("Starting new Galileo session")
                try:
                    # start_session doesn't return a session ID
                    st.session_state.galileo_logger.start_session(
                        name=f"Chat Session {time.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    # Generate our own session ID for reference
                    st.session_state.galileo_session_id = f"session-{time.time()}"
                    logger_debug.info(f"Started Galileo session with reference ID: {st.session_state.galileo_session_id}")
                except Exception as e:
                    logger_debug.error(f"Error starting Galileo session: {str(e)}")
                    st.session_state.galileo_session_id = None
                
                st.rerun()  # Rerun to update UI
        
        # Existing configuration
        st.subheader("RAG Configuration")
        use_rag = st.checkbox("Use RAG", value=True)
        namespace = st.text_input("Namespace", value="sp500-qa-demo")
        top_k = st.number_input("Top K", min_value=1, max_value=20, value=10)
        system_prompt = st.text_area("System Prompt", value="""You are a stock market analyst and trading assistant. You help users analyze stocks and execute trades. Follow these guidelines:

1. For analysis questions, first use the provided context to answer. Only use tools if the context doesn't contain the information needed.
2. For purchase requests:
   - First get the ticker symbol using getTickerSymbol
   - Then get the current stock price using getStockPrice
   - Finally, execute the purchase using purchaseStocks with the current price
3. Format all monetary values with dollar signs and two decimal places.""")
        logger_debug.debug(f"Configuration - RAG: {use_rag}, Namespace: {namespace}, Top K: {top_k}")
        

        if "galileo_logger" in st.session_state:
            hallucination_button = st.button(
                "Log Sample Hallucination", 
                type="primary", 
            )

            if hallucination_button:
                log_hallucination(st.session_state.galileo_logger.project_name, st.session_state.galileo_logger.log_stream_name)

            # Add a danger zone section at the bottom
            st.markdown("---")
            st.subheader("⚠️ Danger Zone")
            
            # Collapsible section to avoid accidental clicks
            with st.expander("Experiment Management"):
                # First show the current experiments
                if st.button("List Experiments"):
                    api_key = st.secrets["galileo_api_key"]
                    project_id = get_galileo_project_id(api_key, st.session_state.galileo_logger.project_name)
                    
                    if project_id:
                        experiments = list_galileo_experiments(api_key, project_id)
                        if experiments:
                            st.write(f"Found {len(experiments)} experiments:")
                            for exp in experiments:
                                st.write(f"• {exp.get('name', 'Unnamed')} ({exp.get('id')})")
                        else:
                            st.info("No experiments found.")
                    else:
                        st.error(f"Project '{galileo_project}' not found.")
                
                # Add the delete all button with a confirmation and admin key verification
                delete_confirm = st.checkbox("I understand this will delete ALL experiments")
                
                # Add admin key input field
                admin_key_input = st.text_input("Admin Key", type="password", 
                                                help="Enter the admin key to enable deletion")
                
                # Only enable the delete button if the confirmation is checked AND the admin key is correct
                is_admin_key_valid = admin_key_input == st.secrets.get("admin_key", "")
                delete_button = st.button(
                    "Delete All Experiments", 
                    type="primary", 
                    disabled=not (delete_confirm and is_admin_key_valid)
                )
                
                # Provide feedback if admin key is incorrect but entered
                if admin_key_input and not is_admin_key_valid:
                    st.error("Invalid admin key")
                
                if delete_button and delete_confirm and is_admin_key_valid:
                    api_key = st.secrets["galileo_api_key"]
                    project_id = get_galileo_project_id(api_key, st.session_state.galileo_logger.project_name)
                    
                    if project_id:
                        with st.spinner("Deleting experiments..."):
                            result = delete_all_galileo_experiments(api_key, project_id)
                        
                        if result["success"]:
                            st.success(result["message"])
                        else:
                            st.warning(result["message"])
                            if result["failed"] > 0:
                                st.error(f"Failed to delete {result['failed']} experiments.")
                    else:
                        st.error(f"Project '{st.session_state.galileo_logger.project_name}' not found.")
        
    # Display session status
    if not st.session_state.session_active:
        st.info("⏸️ No active session. Click 'Start New Session' in the sidebar to begin.")
    else:
        st.success(f"✅ Session active - ID: {st.session_state.galileo_session_id}")
    
    # Display chat messages
    for message in st.session_state.messages:
        # Skip system messages - they should not be displayed in the UI
        if message["role"] == "system":
            continue
        
        if message["role"] == "tool":
            # Display tool response
            with st.chat_message("tool"):
                st.markdown(escape_dollar_signs(message["content"]))
        elif message["role"] == "assistant" and "tool_calls" in message and message["tool_calls"]:
            # Display assistant tool call
            with st.chat_message("assistant"):
                st.markdown(escape_dollar_signs(message["content"]))
        else:
            # Display regular message
            with st.chat_message(message["role"]):
                st.markdown(escape_dollar_signs(message["content"]))
    
    # Only show chat input when session is active
    if st.session_state.session_active:
        # Chat input
        if prompt := st.chat_input("What would you like to know?"):
            logger_debug.info(f"Received user input: {prompt}")
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(escape_dollar_signs(prompt))
            
            try:
                # Process the chat message using our extracted function
                chat_result = await process_chat_message(
                    prompt=prompt,
                    message_history=st.session_state.messages,
                    model=model_option,
                    system_prompt=system_prompt,
                    use_rag=use_rag,
                    namespace=namespace,
                    top_k=top_k,
                    galileo_logger=st.session_state.galileo_logger
                )
                
                # Update the message history
                st.session_state.messages = []  # Clear and rebuild to ensure proper order
                
                # First, add all messages up to the user's message
                for msg in chat_result["updated_history"]:
                    if msg["role"] == "user" and msg["content"] == prompt:
                        st.session_state.messages.append(msg)
                        break
                    st.session_state.messages.append(msg)
                
                # Display tool calls and results if any were used
                if chat_result["tool_results"]:
                    for tool_result in chat_result["tool_results"]:
                        # Create tool call data
                        tool_call = {
                            "id": f"call-{time.time()}-{tool_result['name']}",
                            "type": "function",
                            "function": {
                                "name": tool_result["name"],
                                "arguments": json.dumps(tool_result["arguments"])
                            }
                        }
                        
                        # Display the tool call
                        with st.chat_message("assistant"):
                            if "description" in tool_result:
                                description = tool_result["description"]
                            else:
                                if tool_result["name"] == "getTickerSymbol":
                                    description = f"Looking up ticker symbol for {tool_result['arguments']['company']}..."
                                elif tool_result["name"] == "getStockPrice":
                                    description = f"Getting current price for {tool_result['arguments']['ticker']}..."
                                elif tool_result["name"] == "purchaseStocks":
                                    args = tool_result["arguments"]
                                    description = f"Processing purchase of {args['quantity']} shares of {args['ticker']}..."
                                else:
                                    description = f"Using tool: {tool_result['name']}..."
                            
                            st.markdown(escape_dollar_signs(description))
                        
                        # Add tool call message to session state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": description,
                            "tool_calls": [tool_call]
                        })
                        
                        # Display the tool result
                        with st.chat_message("tool"):
                            st.markdown(escape_dollar_signs(tool_result["result"]))
                        
                        # Add tool result message to session state
                        st.session_state.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": tool_result["result"]
                        })
                
                # Display the final assistant response
                response_message = chat_result["response_message"]
                if response_message.content:
                    # Add final response to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_message.content
                    })
                    
                    # Display the final response
                    with st.chat_message("assistant"):
                        st.markdown(escape_dollar_signs(response_message.content))
                
                # Display Galileo link if available
                if "galileo_logger" in st.session_state:
                    api_key = st.secrets["galileo_api_key"]
                    project_id = get_galileo_project_id(api_key, st.session_state.galileo_logger.project_name)
                    log_stream_id = get_galileo_log_stream_id(api_key, project_id, st.session_state.galileo_logger.log_stream_name) if project_id else None
                    
                    if project_id and log_stream_id:
                        project_url = f"{st.secrets['galileo_console_url']}/project/{project_id}/log-streams/{log_stream_id}"
                        # Add a small icon with tooltip in the sidebar
                        with st.sidebar:
                            st.markdown("---")  # Add a subtle separator
                            st.markdown(
                                f'<div style="font-size: 0.8em; color: #666;">'
                                f'<a href="{project_url}" target="_blank" title="View traces in Galileo">'
                                f'📊 View traces</a></div>',
                                unsafe_allow_html=True
                            )
                    else:
                        with st.sidebar:
                            st.markdown("---")
                            st.markdown(
                                '<div style="font-size: 0.8em; color: #666;">'
                                '📊 Traces logged</div>',
                                unsafe_allow_html=True
                            )
                
            except Exception as e:
                logger_debug.error(f"Error occurred: {str(e)}", exc_info=True)
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 