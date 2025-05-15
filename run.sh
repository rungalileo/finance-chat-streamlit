#!/bin/bash

# Generate .env file from .streamlit/secrets.toml if it exists
if [ -f .streamlit/secrets.toml ]; then
    echo "Generating .env from .streamlit/secrets.toml"
    python generate_env.py
    
    # Check if generation was successful
    if [ ! -f .env ]; then
        echo "Error: Failed to generate .env file"
        exit 1
    fi
fi

# Load environment variables if .env file exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    export $(cat .env | grep -v '#' | xargs)
else
    echo "Warning: .env file not found"
fi

# Function to check if a required environment variable is set
check_env_var() {
    if [ -z "${!1}" ]; then
        echo "Error: $1 environment variable is not set"
        exit 1
    fi
}

# Check required environment variables
check_env_var "OPENAI_API_KEY"
check_env_var "PINECONE_API_KEY"
check_env_var "PINECONE_INDEX_NAME"
check_env_var "GALILEO_API_KEY"
check_env_var "GALILEO_PROJECT"
check_env_var "GALILEO_LOG_STREAM"

# Check command line arguments
if [ "$1" == "streamlit" ]; then
    echo "Starting Streamlit application..."
    streamlit run app_streamlit.py
elif [ "$1" == "flask" ]; then
    echo "Starting Flask API..."
    python app_flask.py
elif [ "$1" == "both" ]; then
    echo "Starting both applications in parallel..."
    # Start Flask in the background
    python app_flask.py &
    FLASK_PID=$!
    echo "Flask API running with PID: $FLASK_PID"
    
    # Start Streamlit
    streamlit run app_streamlit.py
    
    # When Streamlit exits, kill Flask process
    kill $FLASK_PID
else
    echo "Usage: ./run.sh [streamlit|flask|both]"
    echo ""
    echo "  streamlit  - Start the Streamlit UI only"
    echo "  flask      - Start the Flask API only"
    echo "  both       - Start both applications in parallel"
    exit 1
fi 