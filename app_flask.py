import os
import json
import asyncio
from typing import Dict, Any, List
from flask import Flask, request, jsonify
from chat_lib.chat_core import ChatConfig, ChatCore
import logging
from tools.get_ticker_symbol import get_ticker_symbol
from tools.get_stock_price import get_stock_price
from tools.purchase_stocks import purchase_stocks
from galileo import GalileoLogger
from chat_lib.galileo_logger import initialize_galileo_logger

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger_debug = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Get required environment variables
openai_api_key = os.environ.get("OPENAI_API_KEY", "")
pinecone_api_key = os.environ.get("PINECONE_API_KEY", "")
pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME", "")
galileo_api_key = os.environ.get("GALILEO_API_KEY", "")
galileo_project = os.environ.get("GALILEO_PROJECT", "")
galileo_log_stream = os.environ.get("GALILEO_LOG_STREAM", "default")

# Validate required environment variables
if not all([openai_api_key, pinecone_api_key, pinecone_index_name, galileo_api_key, galileo_project]):
    missing_vars = []
    if not openai_api_key: missing_vars.append("OPENAI_API_KEY")
    if not pinecone_api_key: missing_vars.append("PINECONE_API_KEY")
    if not pinecone_index_name: missing_vars.append("PINECONE_INDEX_NAME")
    if not galileo_api_key: missing_vars.append("GALILEO_API_KEY")
    if not galileo_project: missing_vars.append("GALILEO_PROJECT")
    
    logger_debug.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    
# Load configuration
config = ChatConfig(
    openai_api_key=openai_api_key,
    pinecone_api_key=pinecone_api_key,
    pinecone_index_name=pinecone_index_name,
    galileo_api_key=galileo_api_key,
    galileo_project=galileo_project,
    galileo_log_stream=galileo_log_stream
)

# Initialize ChatCore
chat_core = ChatCore(config)

# Log configuration (mask sensitive values)
logger_debug.info(f"Configuration loaded - Project: {galileo_project}, Log Stream: {galileo_log_stream}")
logger_debug.info(f"OpenAI API Key: {'*' * 8}{openai_api_key[-4:] if openai_api_key else 'Not set'}")
logger_debug.info(f"Pinecone API Key: {'*' * 8}{pinecone_api_key[-4:] if pinecone_api_key else 'Not set'}")
logger_debug.info(f"Pinecone Index: {pinecone_index_name}")

# Create a dict to store conversation history for each session
conversations = {}

def handle_tool_call(tool_call: Dict[str, Any], messages_to_use: List[Dict[str, Any]]) -> None:
    """Handle tool calls and their responses.
    
    Args:
        tool_call: The tool call from OpenAI
        messages_to_use: Current conversation history to update
    """
    try:
        function_name = tool_call.get("function", {}).get("name", "")
        function_args = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
        
        if function_name == "getTickerSymbol":
            company = function_args.get("company")
            result = get_ticker_symbol(company, chat_core.galileo_logger)
            logger_debug.info(f"Got ticker symbol for {company}: {result}")
            
        elif function_name == "getStockPrice":
            ticker = function_args.get("ticker")
            result = get_stock_price(ticker, chat_core.galileo_logger)
            logger_debug.info(f"Got stock price for {ticker}: {result}")
            
        elif function_name == "purchaseStocks":
            ticker = function_args.get("ticker")
            quantity = function_args.get("quantity")
            price = function_args.get("price")
            result = purchase_stocks(ticker, quantity, price, chat_core.galileo_logger)
            logger_debug.info(f"Purchased {quantity} shares of {ticker} at ${price}: {result}")
            
        else:
            result = f"Unknown function: {function_name}"
            logger_debug.error(f"Unknown function call: {function_name}")
            
        # Add assistant's tool call to messages
        messages_to_use.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call]
        })
            
        # Add tool response to messages
        messages_to_use.append({
            "role": "tool",
            "tool_call_id": tool_call.get("id"),
            "content": result
        })
            
        return result
    except Exception as e:
        logger_debug.error(f"Error handling tool call: {str(e)}")
        return f"Error: {str(e)}"

@app.route('/api/chat', methods=['POST'])
async def chat():
    """API endpoint for chat requests"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        message = data.get('message')
        system_prompt = data.get('system_prompt', """You are a stock market analyst and trading assistant. You help users analyze stocks and execute trades. Follow these guidelines:

1. For analysis questions, first use the provided context to answer. Only use tools if the context doesn't contain the information needed.
2. For purchase requests:
   - First get the ticker symbol using getTickerSymbol
   - Then get the current stock price using getStockPrice
   - Finally, execute the purchase using purchaseStocks with the current price
3. Format all monetary values with dollar signs and two decimal places.""")
        use_rag = data.get('use_rag', True)
        namespace = data.get('namespace', 'sp500-qa-demo')
        top_k = data.get('top_k', 10)
        model = data.get('model', 'gpt-4')
        
        # Check if client provided Galileo project and log stream
        request_galileo_project = data.get('galileo_project')
        request_galileo_log_stream = data.get('galileo_log_stream')
        
        # If client provided Galileo project or log stream, create a new config and ChatCore
        if request_galileo_project or request_galileo_log_stream:
            custom_config = ChatConfig(
                openai_api_key=openai_api_key,
                pinecone_api_key=pinecone_api_key,
                pinecone_index_name=pinecone_index_name,
                galileo_api_key=galileo_api_key,
                galileo_project=request_galileo_project or galileo_project,
                galileo_log_stream=request_galileo_log_stream or galileo_log_stream
            )
            
            # Create a temporary ChatCore with the custom configuration
            current_chat_core = ChatCore(custom_config)
            logger_debug.info(f"Using custom Galileo configuration - Project: {current_chat_core.config.galileo_project}, Log Stream: {current_chat_core.config.galileo_log_stream}")
        else:
            # Use the default ChatCore
            current_chat_core = chat_core
        
        # Initialize conversation if not exists
        if session_id not in conversations:
            conversations[session_id] = []
            
        # Add user message to conversation
        conversations[session_id].append({
            "role": "user",
            "content": message
        })
        
        # Process chat request with the appropriate ChatCore
        response = await current_chat_core.process_chat_request(
            messages=conversations[session_id],
            system_prompt=system_prompt,
            use_rag=use_rag,
            namespace=namespace,
            top_k=top_k,
            model=model
        )
        
        # Handle tool calls if present
        tool_calls = response.get('tool_calls')
        if tool_calls:
            tool_results = []
            # Save a copy of the current messages
            messages_to_use = conversations[session_id].copy()
            
            continue_conversation = True
            while continue_conversation and tool_calls:
                for tool_call in tool_calls:
                    result = handle_tool_call(tool_call, messages_to_use)
                    tool_results.append({
                        "tool": tool_call.get("function", {}).get("name"),
                        "result": result
                    })
                
                # Get follow-up response
                follow_up_response = await current_chat_core.process_chat_request(
                    messages=messages_to_use,
                    system_prompt=system_prompt,
                    use_rag=False,  # Don't need RAG for follow-up
                    model=model
                )
                
                # Update response with follow-up
                response = follow_up_response
                tool_calls = response.get('tool_calls')
                
                # If no more tool calls, end the conversation
                if not tool_calls:
                    continue_conversation = False
            
            # Update conversation with final state
            conversations[session_id] = messages_to_use
            
            # Add assistant's final response
            if response.get('content'):
                conversations[session_id].append({
                    "role": "assistant",
                    "content": response.get('content')
                })
                
            return jsonify({
                "session_id": session_id,
                "response": response.get('content'),
                "tool_results": tool_results,
                "conversation": conversations[session_id]
            })
        
        # If no tool calls, just add the response and return
        conversations[session_id].append({
            "role": "assistant",
            "content": response.get('content')
        })
        
        return jsonify({
            "session_id": session_id,
            "response": response.get('content'),
            "conversation": conversations[session_id]
        })
        
    except Exception as e:
        logger_debug.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """List all active chat sessions"""
    return jsonify({
        "sessions": list(conversations.keys())
    })

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a specific chat session"""
    if session_id in conversations:
        del conversations[session_id]
        return jsonify({
            "success": True,
            "message": f"Session {session_id} deleted"
        })
    else:
        return jsonify({
            "success": False,
            "message": f"Session {session_id} not found"
        }), 404

@app.route('/api/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get conversation history for a specific session"""
    if session_id in conversations:
        return jsonify({
            "session_id": session_id,
            "conversation": conversations[session_id]
        })
    else:
        return jsonify({
            "success": False,
            "message": f"Session {session_id} not found"
        }), 404

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    # Use Gunicorn for production
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True) 