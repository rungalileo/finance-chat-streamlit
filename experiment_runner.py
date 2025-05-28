import os
import asyncio
import json
import logging
import datetime
from dotenv import load_dotenv
from app import process_chat_message
from galileo.datasets import get_dataset
from galileo.experiments import run_experiment
from galileo import galileo_context

# Configure logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def process_trade_prompt(example):
    """
    Custom function to process a trade prompt using process_chat_message.
    This is an async wrapper around our process_chat_message function.
    
    Args:
        example: A dataset example with 'input' field
        
    Returns:
        The model's response to the prompt
    """
    model = "gpt-4"
    system_prompt = """You are a stock market analyst and trading assistant. You help users analyze stocks and execute trades."""
    message_history = []

    # Define an async function to process the message
    async def process_async():
        # Process the chat message
        result = await process_chat_message(
            prompt=example,
            message_history=message_history,
            model=model,
            system_prompt=system_prompt,
            use_rag=True,
            namespace="sp500-qa-demo",
            top_k=5,
            galileo_logger=galileo_context.get_logger_instance(),
            ambiguous_tool_names=True,
            is_streamlit=False
        )
        return result

    # Process the chat message
    try:
        # Run the async function
        result = asyncio.run(process_async())
        
        # Extract and return the response content
        response = result["response_message"].content
        
        # Add metadata about tool usage if available
        metadata = {}
        if result.get("tool_results"):
            metadata["tools_used"] = [tool["name"] for tool in result["tool_results"]]
        
        if result.get("rag_documents"):
            metadata["rag_documents_count"] = len(result["rag_documents"])
        
        # Log the metadata separately
        logger.info(f"Response metadata: {json.dumps(metadata)}")
        
        return response  # Return just the response string for Galileo logging
    except Exception as e:
        logger.error(f"Error processing prompt: {e}")
        return {"response": f"Error: {str(e)}", "metadata": {"error": str(e)}}

def main():
    # Ensure required environment variables are set
    required_vars = ["GALILEO_API_KEY", "GALILEO_PROJECT_NAME", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return
    
    # Get project name from environment
    galileo_project = os.getenv("GALILEO_PROJECT_NAME")
    
    try:
        # Get the dataset
        try:
            dataset = get_dataset(name="trades")
        except Exception as e:
            logger.error(f"Error getting dataset 'trades': {e}")
            return
        
        # Create a unique experiment name with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"trade_tool_selection_{timestamp}"
        
        logger.info(f"Starting experiment: {experiment_name}")
        
        # Run the experiment with our custom function
        results = run_experiment(
            experiment_name,
            dataset=dataset,
            function=process_trade_prompt,
            metrics=["correctness"],
            project=galileo_project
        )
        
        logger.info(f"Experiment completed: {experiment_name}")
        
    except Exception as e:
        logger.error(f"Error running experiment: {e}")


if __name__ == "__main__":
    main()