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