import asyncio
import json
import logging
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from app import process_chat_message, initialize_galileo_logger

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # Initialize OpenAI key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        return
    
    # Set up Galileo logger (optional)
    galileo_logger = None
    if os.getenv("GALILEO_API_KEY") and os.getenv("GALILEO_PROJECT_NAME") and os.getenv("GALILEO_LOG_STREAM_NAME"):
        try:
            galileo_logger = initialize_galileo_logger(
                os.getenv("GALILEO_PROJECT_NAME"),
                os.getenv("GALILEO_LOG_STREAM_NAME")
            )
            galileo_logger.start_session(name="Example Script Session")
        except Exception as e:
            logger.error(f"Failed to initialize Galileo logger: {e}")
            galileo_logger = None
    
    # Configuration options
    model = "gpt-4"
    system_prompt = """You are a stock market analyst and trading assistant. You help users analyze stocks and execute trades."""
    use_rag = True
    namespace = "sp500-qa-demo"
    top_k = 5
    
    # Initialize message history
    message_history = []
    
    # Example conversation
    questions = [
        "Tell me about Apple's financial performance last quarter",
        "What's the current stock price of Microsoft?",
        "I want to buy 10 shares of Tesla"
    ]
    
    for question in questions:
        print(f"\nUser: {question}")
        
        # Process the message
        try:
            result = await process_chat_message(
                prompt=question,
                message_history=message_history,
                model=model,
                system_prompt=system_prompt,
                use_rag=use_rag,
                namespace=namespace,
                top_k=top_k,
                galileo_logger=galileo_logger
            )
            
            # Update message history with the result
            message_history = result["updated_history"]
            
            # Print the response
            response_message = result["response_message"]
            print(f"Assistant: {response_message.content}")
            
            # Display any tool usage
            if result["tool_results"]:
                print("\nTools used:")
                for tool in result["tool_results"]:
                    print(f"- {tool['name']} with args: {json.dumps(tool['arguments'])}")
                    print(f"  Result: {tool['result']}")
            
            # Show number of RAG documents if available
            if result["rag_documents"]:
                print(f"\nRAG: Found {len(result['rag_documents'])} relevant documents")
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            print(f"Error: {str(e)}")
    
    # Clean up Galileo logger if used
    if galileo_logger:
        try:
            galileo_logger.end_session()
        except Exception as e:
            logger.error(f"Error ending Galileo session: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 