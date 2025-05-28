# Finance Transcripts RAG Chat Application

This application provides both a Streamlit web interface and a Flask REST API for interacting with a RAG system focused on financial transcripts.

## Features

- Question answering using Retrieval Augmented Generation (RAG) over financial transcripts
- Tools for stock lookup and trading simulation
- Galileo observability built-in
- Available as both an interactive web UI and a REST API
- Configurable model selection (GPT-4 or GPT-3.5 Turbo)

## Architecture

The application is structured as follows:

- `chat_lib/` - Shared library for both Streamlit and Flask applications
  - `chat_core.py` - Core chat functionality including RAG, OpenAI integration, and tool handling
- `app_streamlit.py` - Streamlit web interface
- `app_flask.py` - Flask REST API
- `tools/` - Financial tools for ticker lookup, price checking, and trading
- `galileo_api_helper.py` - Helper functions for Galileo observability

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up required environment variables or use a `.streamlit/secrets.toml` file:

```toml
openai_api_key = "your-openai-api-key"
pinecone_api_key = "your-pinecone-api-key"
pinecone_index_name = "your-pinecone-index-name"
galileo_api_key = "your-galileo-api-key"
galileo_project = "your-galileo-project-name"
galileo_log_stream = "your-galileo-log-stream-name"
alpha_vantage_api_key = "your-alpha-vantage-api-key"
admin_key = "your-admin-key" 
galileo_console_url = "https://app.galileo.ai"
```

## Running the Applications

### Running the Streamlit Web UI

```bash
streamlit run app_streamlit.py
```

The application will be available at http://localhost:8501.

### Running the Flask API

#### Development Mode

```bash
python app_flask.py
```

The API will be available at http://localhost:5000.

#### Production Mode

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app_flask:app
```

## API Endpoints

### POST /api/chat

Send a chat message and receive a response.

**Request:**

```json
{
  "session_id": "string",
  "message": "string",
  "system_prompt": "string",
  "use_rag": true,
  "namespace": "sp500-qa-demo",
  "top_k": 10,
  "model": "gpt-4",
  "galileo_project": "optional-project-name",
  "galileo_log_stream": "optional-log-stream-name"
}
```

Only `message` is required; all other fields have defaults. If `galileo_project` and `galileo_log_stream` are provided, they will override the environment variables set in the `.env` file.

**Response:**

```json
{
  "session_id": "string",
  "response": "string",
  "tool_results": [
    {
      "tool": "string",
      "result": "string"
    }
  ],
  "conversation": [
    {
      "role": "string",
      "content": "string"
    }
  ]
}
```

### GET /api/sessions

List all active sessions.

**Response:**

```json
{
  "sessions": ["session-id-1", "session-id-2"]
}
```

### GET /api/sessions/{session_id}

Get the conversation history for a specific session.

**Response:**

```json
{
  "session_id": "string",
  "conversation": [
    {
      "role": "string",
      "content": "string"
    }
  ]
}
```

### DELETE /api/sessions/{session_id}

Delete a specific session.

**Response:**

```json
{
  "success": true,
  "message": "Session {session_id} deleted"
}
```

### GET /health

Health check endpoint.

**Response:**

```json
{
  "status": "healthy"
}
```

## License

[MIT License](LICENSE)

## New Feature: Standalone Chat Function

The core chat functionality has been extracted into a separate function called `process_chat_message`, which can be used independently of the Streamlit UI. This allows you to:

1. Integrate the chat functionality into other applications
2. Run the chat in a standalone script
3. Use the chat in an API or service

## Using the Standalone Chat Function

### Environment Variables

Create a `.env` file with the following variables:

```
# Core API keys
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=your_pinecone_index_name_here

# Galileo configuration (optional)
GALILEO_API_KEY=your_galileo_api_key_here
GALILEO_PROJECT_NAME=your_galileo_project_name_here
GALILEO_LOG_STREAM_NAME=your_galileo_log_stream_name_here
GALILEO_CONSOLE_URL=your_galileo_console_url_here
```

### Running the Example

1. Install dependencies: `pip install -r requirements.txt`
2. Run the example script: `python example.py`

### Using in Your Own Code

```python
import asyncio
from app import process_chat_message

async def chat_example():
    # Initialize message history
    message_history = []
    
    # Process a message
    result = await process_chat_message(
        prompt="Tell me about Apple's performance",
        message_history=message_history,
        model="gpt-4",
        system_prompt="You are a financial assistant",
        use_rag=True,
        namespace="sp500-qa-demo",
        top_k=5,
        galileo_logger=None  # Optional
    )
    
    # Update message history
    message_history = result["updated_history"]
    
    # Get the response
    response = result["response_message"].content
    print(f"Assistant: {response}")

# Run the async function
asyncio.run(chat_example())
```

## Function Parameters

The `process_chat_message` function takes the following parameters:

- `prompt` (str): The user's message/prompt
- `message_history` (List[Dict]): Previous message history
- `model` (str, optional): The OpenAI model to use. Defaults to "gpt-4".
- `system_prompt` (str, optional): System prompt to use. Defaults to None.
- `use_rag` (bool, optional): Whether to use RAG for context. Defaults to True.
- `namespace` (str, optional): Namespace for RAG. Defaults to "sp500-qa-demo".
- `top_k` (int, optional): Number of top documents to retrieve for RAG. Defaults to 10.
- `galileo_logger` (optional): Optional Galileo logger for observability.

## Return Value

The function returns a dictionary containing:

- `response_message`: The final response message from the model
- `updated_history`: The updated message history
- `rag_documents`: Any RAG documents retrieved (if RAG was used)
- `tool_results`: Results from any tools used during the conversation
- `total_tokens`: Total token count for the conversation

## Original Streamlit App

The original Streamlit app is still available and now uses the extracted function internally.

To run the Streamlit app:

```
streamlit run app.py
``` 