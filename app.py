import os
import json
import streamlit as st
import openai
from galileo import GalileoLogger
import time
import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# Import tools
from tools.get_ticker_symbol import get_ticker_symbol
from tools.get_stock_price import get_stock_price
from tools.purchase_stocks import purchase_stocks

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger_debug = logging.getLogger(__name__)

os.environ["GALILEO_API_KEY"] = st.secrets["galileo_api_key"]
os.environ["GALILEO_PROJECT_NAME"] = st.secrets["galileo_project"]
os.environ["GALILEO_LOG_STREAM_NAME"] = st.secrets["galileo_log_stream"]

# Initialize OpenAI client
logger_debug.info("Initializing OpenAI client")
openai_client = OpenAI(
    api_key=st.secrets["openai_api_key"]
)
logger_debug.debug(f"OpenAI API Key loaded: {'*' * 8}{st.secrets["openai_api_key"][-4:] if st.secrets["openai_api_key"] else 'Not found'}")

# Initialize Pinecone
logger_debug.info("Initializing Pinecone client")
pc = Pinecone(
    api_key=st.secrets["pinecone_api_key"],
    spec=ServerlessSpec(cloud="aws", region="us-west-2")
)
logger_debug.debug(f"Pinecone API Key loaded: {'*' * 8}{st.secrets["pinecone_api_key"][-4:] if st.secrets["pinecone_api_key"] else 'Not found'}")

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

def format_message(role: str, content: str) -> dict:
    """Format a message for the chat."""
    return {"role": role, "content": content}

async def main():
    st.title("RAG Chat Application")
    logger_debug.info("Starting Streamlit application")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        logger_debug.info("Initializing new chat history")
        st.session_state.messages = []
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Add Galileo configuration fields
        st.subheader("Galileo Configuration")
        galileo_project = st.text_input(
            "Galileo Project",
            value=st.secrets["galileo_project"],
            help="The name of your Galileo project"
        )
        galileo_log_stream = st.text_input(
            "Galileo Log Stream",
            value=st.secrets["galileo_log_stream"],
            help="The name of your Galileo log stream"
        )
        
        # Existing configuration
        st.subheader("RAG Configuration")
        use_rag = st.checkbox("Use RAG", value=True)
        namespace = st.text_input("Namespace", value="sp500-qa-demo")
        top_k = st.number_input("Top K", min_value=1, max_value=10, value=3)
        system_prompt = st.text_area("System Prompt", value="""You are a stock market analyst and trading assistant. You help users analyze stocks and execute trades. Follow these guidelines:

1. For analysis questions, first use the provided context to answer. Only use tools if the context doesn't contain the information needed.
2. For purchase requests:
   - First get the ticker symbol using getTickerSymbol
   - Then get the current stock price using getStockPrice
   - Finally, execute the purchase using purchaseStocks with the current price
3. Format all monetary values with dollar signs and two decimal places.""")
        logger_debug.debug(f"Configuration - RAG: {use_rag}, Namespace: {namespace}, Top K: {top_k}")
    
    # Initialize Galileo logger with values from input fields
    logger_debug.info("Initializing Galileo logger")
    logger = GalileoLogger(
        project=galileo_project,
        log_stream=galileo_log_stream
    )
    logger_debug.debug(f"Galileo Project: {galileo_project}")
    logger_debug.debug(f"Galileo Log Stream: {galileo_log_stream}")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(escape_dollar_signs(message["content"]))
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        logger_debug.info(f"Received user input: {prompt}")
        
        # Add user message to chat history
        st.session_state.messages.append(format_message("user", prompt))
        logger_debug.debug(f"Current message history length: {len(st.session_state.messages)}")
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(escape_dollar_signs(prompt))
        
        # Start a new trace
        start_time = time.time()
        logger_debug.info("Starting new Galileo trace")
        trace = logger.start_trace(
            input=prompt,
            name="Chat Workflow",
            tags=["chat"],
            metadata={"session_id": str(time.time())}
        )
        
        try:
            messages_to_use = st.session_state.messages.copy()
            
            # Handle RAG if enabled
            if use_rag:
                logger_debug.info("RAG enabled, fetching relevant documents")
                rag_response = await get_rag_response(prompt, namespace, top_k)
                
                if rag_response and rag_response.documents:
                    logger_debug.info(f"RAG returned {len(rag_response.documents)} documents")
                    
                    # Log RAG retrieval to Galileo
                    logger.add_retriever_span(
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
            elif system_prompt:
                logger_debug.info("Adding system prompt without RAG")
                messages_to_use = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    *messages_to_use
                ]
            
            # Get response from OpenAI
            logger_debug.info("Calling OpenAI API")
            logger_debug.debug(f"Messages being sent to OpenAI: {json.dumps([format_message(msg['role'], msg['content']) for msg in messages_to_use], indent=2)}")
            logger_debug.debug(f"Tools being sent to OpenAI: {json.dumps(openai_tools, indent=2)}")
            
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=messages_to_use,
                tools=openai_tools,
                tool_choice="auto"
            )
            
            response_message = response.choices[0].message
            logger_debug.info("Received response from OpenAI")
            logger_debug.debug(f"Response message: {json.dumps({
                'role': response_message.role,
                'content': response_message.content,
                'tool_calls': [{
                    'id': call.id,
                    'type': call.type,
                    'function': {
                        'name': call.function.name,
                        'arguments': call.function.arguments
                    }
                } for call in (response_message.tool_calls or [])]
            }, indent=2)}")

            # Calculate token counts safely
            input_tokens = len(prompt.split()) if prompt else 0
            output_tokens = len(response_message.content.split()) if response_message.content else 0
            total_tokens = input_tokens + output_tokens

            # Log the API call
            logger_debug.info("Logging API call to Galileo")
            logger.add_llm_span(
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
                model="gpt-4",
                name="OpenAI API Call",
                tools=[{"name": name, "parameters": list(tool["parameters"]["properties"].keys())} 
                      for name, tool in tools.items()],
                duration_ns=int((time.time() - start_time) * 1000000),
                metadata={"temperature": "0.7"},
                tags=["api-call"],
                num_input_tokens=input_tokens,
                num_output_tokens=output_tokens,
                total_tokens=total_tokens
            )
            
            # Handle tool calls if present
            if response_message.tool_calls:
                logger_debug.info("Processing tool calls")
                continue_conversation = True
                
                while continue_conversation and response_message.tool_calls:
                    # Process each tool call and its response
                    for tool_call in response_message.tool_calls:
                        # First add the assistant's message with this specific tool call
                        messages_to_use.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [tool_call]  # Only include this specific tool call
                        })
                        
                        if tool_call.function.name == "getTickerSymbol":
                            company = json.loads(tool_call.function.arguments)["company"]
                            ticker = get_ticker_symbol(company, logger)
                            logger_debug.info(f"Got ticker symbol for {company}: {ticker}")
                            
                            # Add tool response to messages
                            messages_to_use.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": ticker
                            })
                            
                            # Display the tool response in chat
                            st.session_state.messages.append(format_message("assistant", f"Looking up ticker symbol for {company}..."))
                            st.session_state.messages.append(format_message("tool", ticker))
                            with st.chat_message("assistant"):
                                st.markdown(escape_dollar_signs(f"Looking up ticker symbol for {company}..."))
                            with st.chat_message("tool"):
                                st.markdown(escape_dollar_signs(ticker))
                            
                        elif tool_call.function.name == "purchaseStocks":
                            args = json.loads(tool_call.function.arguments)
                            result = purchase_stocks(
                                ticker=args["ticker"],
                                quantity=args["quantity"],
                                price=args["price"],
                                galileo_logger=logger
                            )
                            logger_debug.info(f"Processed stock purchase for {args['ticker']}")
                            
                            # Add tool response to messages
                            messages_to_use.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result
                            })
                            
                            # Display the tool response in chat
                            st.session_state.messages.append(format_message("assistant", f"Processing purchase of {args['quantity']} shares of {args['ticker']}..."))
                            st.session_state.messages.append(format_message("tool", result))
                            with st.chat_message("assistant"):
                                st.markdown(escape_dollar_signs(f"Processing purchase of {args['quantity']} shares of {args['ticker']}..."))
                            with st.chat_message("tool"):
                                st.markdown(escape_dollar_signs(result))
                            
                        elif tool_call.function.name == "getStockPrice":
                            ticker = json.loads(tool_call.function.arguments)["ticker"]
                            result = get_stock_price(ticker, galileo_logger=logger)
                            logger_debug.info(f"Got stock price for {ticker}")
                            
                            # Add tool response to messages
                            messages_to_use.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result
                            })
                            
                            # Display the tool response in chat
                            st.session_state.messages.append(format_message("assistant", f"Getting current price for {ticker}..."))
                            st.session_state.messages.append(format_message("tool", result))
                            with st.chat_message("assistant"):
                                st.markdown(escape_dollar_signs(f"Getting current price for {ticker}..."))
                            with st.chat_message("tool"):
                                st.markdown(escape_dollar_signs(result))
                    
                    # Get a new response from OpenAI with the tool results
                    logger_debug.info("Getting follow-up response with tool results")
                    follow_up_response = openai_client.chat.completions.create(
                        model="gpt-4",
                        messages=messages_to_use,
                        tools=openai_tools,
                        tool_choice="auto"
                    )
                    
                    response_message = follow_up_response.choices[0].message
                    logger_debug.debug(f"Follow-up response: {json.dumps({
                        'role': response_message.role,
                        'content': response_message.content,
                        'tool_calls': [{
                            'id': call.id,
                            'type': call.type,
                            'function': {
                                'name': call.function.name,
                                'arguments': call.function.arguments
                            }
                        } for call in (response_message.tool_calls or [])]
                    }, indent=2)}")

                    # Calculate token counts for follow-up response
                    follow_up_input_tokens = sum(len(msg.get("content", "").split()) for msg in messages_to_use if msg.get("content"))
                    follow_up_output_tokens = len(response_message.content.split()) if response_message.content else 0
                    follow_up_total_tokens = follow_up_input_tokens + follow_up_output_tokens

                    # Log the follow-up API call
                    logger_debug.info("Logging follow-up API call to Galileo")
                    logger.add_llm_span(
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
                        model="gpt-4",
                        name="Follow-up OpenAI API Call",
                        tools=[{"name": name, "parameters": list(tool["parameters"]["properties"].keys())} 
                              for name, tool in tools.items()],
                        duration_ns=int((time.time() - start_time) * 1000000),
                        metadata={"temperature": "0.7"},
                        tags=["api-call", "follow-up"],
                        num_input_tokens=follow_up_input_tokens,
                        num_output_tokens=follow_up_output_tokens,
                        total_tokens=follow_up_total_tokens
                    )
                    
                    # If no more tool calls, end the conversation
                    if not response_message.tool_calls:
                        continue_conversation = False
            
            # Add assistant message to chat history
            if response_message.content:
                st.session_state.messages.append(format_message("assistant", response_message.content))
                logger_debug.debug(f"Updated message history length: {len(st.session_state.messages)}")
                
                # Display assistant message
                with st.chat_message("assistant"):
                    st.markdown(escape_dollar_signs(response_message.content))
            
            # Conclude the trace
            logger_debug.info("Concluding Galileo trace")
            logger.conclude(
                output=response_message.content,
                duration_ns=int((time.time() - start_time) * 1000000),
                status_code=200
            )
            
            # Flush the trace to Galileo
            logger_debug.info("Flushing trace to Galileo")
            logger.flush()
            logger_debug.info("Trace successfully flushed")
            
        except Exception as e:
            logger_debug.error(f"Error occurred: {str(e)}", exc_info=True)
            st.error(f"An error occurred: {str(e)}")
            # Log error and conclude trace
            logger_debug.info("Logging error to Galileo")
            logger.conclude(
                output=f"Error: {str(e)}",
                duration_ns=int((time.time() - start_time) * 1000000),
                status_code=500
            )
            logger.flush()
            logger_debug.info("Error trace flushed")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 