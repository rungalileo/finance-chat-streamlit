import streamlit as st
import os
import logging
from datetime import datetime
from galileo import galileo_context
from galileo.datasets import get_dataset
from galileo.experiments import run_experiment
from app import process_chat_message
import asyncio

# Configure logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Initialize session state
if "current_experiment" not in st.session_state:
    st.session_state.current_experiment = None

# Page title
st.title("Run Experiment")

# Sidebar with configuration
with st.sidebar:
    st.header("Configuration")
    
    st.subheader("Galileo Configuration")
    galileo_project = st.text_input(
        "Galileo Project",
        value=os.getenv("GALILEO_PROJECT_NAME", ""),
        help="The name of your Galileo project"
    )
    galileo_log_stream = st.text_input(
        "Galileo Log Stream",
        value=os.getenv("GALILEO_LOG_STREAM_NAME", ""),
        help="The name of your Galileo log stream"
    )

    galileo_api_key = st.text_input(
        "Galileo API Key",
        value=os.getenv("GALILEO_API_KEY", ""),
        help="The API key for your Galileo project"
    )

    galileo_console_url = st.text_input(
        "Galileo Console URL",
        value=os.getenv("GALILEO_CONSOLE_URL", ""),
        help="The URL of your Galileo console"
    )
    

# Main content
st.header("Experiment Configuration")
# Experiment name
experiment_name = st.text_input(
    "Experiment Name",
    value=f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    help="A unique name for this experiment"
)

# Dataset selection
selected_dataset = st.text_input(
    "Dataset Name",
    value="trades",
    help="Choose the dataset to run the experiment on"
)

# Tool configuration
st.subheader("App Configuration")
ambiguous_tool_names = st.checkbox(
    "Use Ambiguous Tool Names",
    value=False,
    help="Makes sell / buy functions ambiguous to induce poor tool selection"
)

# Model selection
st.subheader("Model Configuration")
model = st.selectbox(
    "Select GPT Model",
    options=["gpt-4", "gpt-3.5-turbo"],
    index=0,
    format_func=lambda x: "GPT-4" if x == "gpt-4" else "GPT-3.5 Turbo"
)

st.subheader("RAG Configuration")
# System prompt
system_prompt = st.text_area(
    "System Prompt",
    value="""You are a stock market analyst and trading assistant. You help users analyze stocks and execute trades.""",
    height=100
)

# RAG configuration
use_rag = st.checkbox("Use RAG", value=True)
namespace = st.text_input("Pinecone Namespace", value=st.secrets["pinecone_namespace"])
top_k = st.number_input("Top K", min_value=1, max_value=20, value=10)

# Run experiment button
if st.button("Run Experiment", type="primary"):
    
    try:
        # Get the dataset
        dataset = get_dataset(name=selected_dataset)
        if not dataset:
            st.error(f"Dataset '{selected_dataset}' not found")
            st.stop()

        def process_trade_prompt(example):
            
            async def process_async(prompt):
                result = await process_chat_message(
                prompt=prompt,
                message_history=[],
                model=model,
                system_prompt=system_prompt,
                use_rag=use_rag,
                namespace=namespace,
                top_k=top_k,
                galileo_logger=galileo_context.get_logger_instance(),
                ambiguous_tool_names=ambiguous_tool_names,
                is_streamlit=True
                )
                return result["response_message"].content

            try:

                # Run the async function
                result = asyncio.run(process_async(example))
                
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
        
        # Run the experiment
        st.info(f"Starting experiment: {experiment_name}")
        
        dataset = get_dataset(name=selected_dataset)
        if not dataset:
            st.error(f"Dataset '{selected_dataset}' not found")
            
        
        results = run_experiment(
            experiment_name=experiment_name,
            dataset=dataset,
            function=process_trade_prompt,
            metrics=["correctness"],
            project=galileo_project
        )
        
        # Store results in session state
        st.session_state.current_experiment = {
            "name": experiment_name,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        st.success(f"Experiment completed: {experiment_name}")
        
    except Exception as e:
        st.error(f"Error running experiment: {str(e)}")
        logger.error(f"Error running experiment: {e}", exc_info=True)

# Display current experiment results
if st.session_state.current_experiment:
    st.header("Current Experiment Results")
    st.write(f"Experiment: {st.session_state.current_experiment['name']}")
    st.write(f"Timestamp: {st.session_state.current_experiment['timestamp']}")
    st.json(st.session_state.current_experiment['results'])
