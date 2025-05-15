import os
import json
import streamlit as st
import time
import logging
import asyncio
from typing import Dict, Any, Optional
from urllib.parse import unquote
from chat_lib.chat_core import ChatConfig, ChatCore
from galileo_api_helper import get_galileo_project_id, get_galileo_log_stream_id, list_galileo_experiments, delete_all_galileo_experiments

# Import tools
from log_hallucination import log_hallucination
from tools.get_ticker_symbol import get_ticker_symbol
from tools.get_stock_price import get_stock_price
from tools.purchase_stocks import purchase_stocks

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger_debug = logging.getLogger(__name__)

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

async def main():
    # Read environment variables or secrets
    openai_api_key = st.secrets["openai_api_key"]
    pinecone_api_key = st.secrets["pinecone_api_key"]
    pinecone_index_name = st.secrets["pinecone_index_name"]
    galileo_api_key = st.secrets["galileo_api_key"]
    galileo_project = st.secrets["galileo_project"]
    galileo_log_stream = st.secrets["galileo_log_stream"]
    galileo_console_url = st.secrets["galileo_console_url"]

    # Initialize ChatCore configuration
    config = ChatConfig(
        openai_api_key=openai_api_key,
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=pinecone_index_name,
        galileo_api_key=galileo_api_key,
        galileo_project=galileo_project,
        galileo_log_stream=galileo_log_stream,
        galileo_console_url=galileo_console_url
    )
    
    # Initialize ChatCore
    chat_core = ChatCore(config)
    
    st.title("RAG Chat Application")
    logger_debug.info("Starting Streamlit application")
    
    # Get query parameters
    default_project = unquote(st.query_params.get("project", galileo_project))
    default_log_stream = unquote(st.query_params.get("log_stream", galileo_log_stream))
    
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
                st.session_state.session_active = True
                st.session_state.messages = []  # Clear any previous messages
                
                # Start a new Galileo session
                logger_debug.info("Starting new Galileo session")
                try:
                    # start_session doesn't return a session ID
                    chat_core.galileo_logger.start_session(
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
        
        hallucination_button = st.button(
            "Log Sample Hallucination", 
            type="primary", 
        )

        if hallucination_button:
            log_hallucination(galileo_project, galileo_log_stream)

        # Add a danger zone section at the bottom
        st.markdown("---")
        st.subheader("⚠️ Danger Zone")
        
        # Collapsible section to avoid accidental clicks
        with st.expander("Experiment Management"):
            # First show the current experiments
            if st.button("List Experiments"):
                api_key = galileo_api_key
                project_id = get_galileo_project_id(api_key, galileo_project)
                
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
                api_key = galileo_api_key
                project_id = get_galileo_project_id(api_key, galileo_project)
                
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
                    st.error(f"Project '{galileo_project}' not found.")
    
    # Display session status
    if not st.session_state.session_active:
        st.info("⏸️ No active session. Click 'Start New Session' in the sidebar to begin.")
    else:
        st.success(f"✅ Session active - ID: {st.session_state.galileo_session_id}")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(escape_dollar_signs(message["content"]))
    
    # Only show chat input when session is active
    if st.session_state.session_active:
        # Chat input
        if prompt := st.chat_input("What would you like to know?"):
            logger_debug.info(f"Received user input: {prompt}")
            
            # Add user message to chat history
            st.session_state.messages.append(format_message("user", prompt))
            logger_debug.debug(f"Current message history length: {len(st.session_state.messages)}")
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(escape_dollar_signs(prompt))
            
            # Convert session messages to format for ChatCore
            messages_for_api = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    messages_for_api.append({"role": "user", "content": msg["content"]})
                elif msg["role"] == "assistant":
                    if "tool_calls" in msg:
                        messages_for_api.append({
                            "role": "assistant", 
                            "content": None,
                            "tool_calls": msg["tool_calls"]
                        })
                    else:
                        messages_for_api.append({"role": "assistant", "content": msg["content"]})
                elif msg["role"] == "tool":
                    messages_for_api.append({
                        "role": "tool", 
                        "tool_call_id": msg["tool_call_id"],
                        "content": msg["content"]
                    })
            
            try:
                # Process chat request using ChatCore
                response = await chat_core.process_chat_request(
                    messages=messages_for_api,
                    system_prompt=system_prompt,
                    use_rag=use_rag,
                    namespace=namespace,
                    top_k=top_k,
                    model=model_option
                )
                
                # Handle tool calls if present
                if response.get('tool_calls'):
                    for tool_call in response['tool_calls']:
                        function_name = tool_call.get("function", {}).get("name", "")
                        function_args = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                        
                        if function_name == "getTickerSymbol":
                            company = function_args.get("company")
                            result = get_ticker_symbol(company, chat_core.galileo_logger)
                            description = f"Looking up ticker symbol for {company}..."
                            
                            # Handle tool call in UI
                            handle_tool_call(
                                tool_call=type('obj', (object,), {
                                    'id': tool_call['id'],
                                    'function': type('obj', (object,), {
                                        'name': function_name,
                                        'arguments': tool_call['function']['arguments']
                                    })
                                }),
                                tool_result=result,
                                description=description,
                                messages_to_use=messages_for_api,
                                logger=chat_core.galileo_logger
                            )
                                
                        elif function_name == "getStockPrice":
                            ticker = function_args.get("ticker")
                            result = get_stock_price(ticker, chat_core.galileo_logger)
                            description = f"Getting current price for {ticker}..."
                            
                            # Handle tool call in UI
                            handle_tool_call(
                                tool_call=type('obj', (object,), {
                                    'id': tool_call['id'],
                                    'function': type('obj', (object,), {
                                        'name': function_name,
                                        'arguments': tool_call['function']['arguments']
                                    })
                                }),
                                tool_result=result,
                                description=description,
                                messages_to_use=messages_for_api,
                                logger=chat_core.galileo_logger
                            )
                            
                        elif function_name == "purchaseStocks":
                            args = function_args
                            result = purchase_stocks(
                                ticker=args["ticker"],
                                quantity=args["quantity"],
                                price=args["price"],
                                galileo_logger=chat_core.galileo_logger
                            )
                            description = f"Processing purchase of {args['quantity']} shares of {args['ticker']}..."
                            
                            # Handle tool call in UI
                            handle_tool_call(
                                tool_call=type('obj', (object,), {
                                    'id': tool_call['id'],
                                    'function': type('obj', (object,), {
                                        'name': function_name,
                                        'arguments': tool_call['function']['arguments']
                                    })
                                }),
                                tool_result=result,
                                description=description,
                                messages_to_use=messages_for_api,
                                logger=chat_core.galileo_logger
                            )
                    
                    # Get a new response from OpenAI with the tool results
                    follow_up_response = await chat_core.process_chat_request(
                        messages=messages_for_api,
                        system_prompt=system_prompt,
                        use_rag=False,  # Don't need RAG for follow-up
                        model=model_option
                    )
                    
                    response = follow_up_response
                
                # Add assistant message to chat history if it has content
                if response.get('content'):
                    st.session_state.messages.append(format_message("assistant", response['content']))
                    
                    # Display assistant message
                    with st.chat_message("assistant"):
                        st.markdown(escape_dollar_signs(response['content']))
            
            except Exception as e:
                logger_debug.error(f"Error occurred: {str(e)}", exc_info=True)
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 