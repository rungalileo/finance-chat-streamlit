# RAG Chat Application (Streamlit)

This is a Streamlit version of the RAG Chat Application that provides a user-friendly interface for interacting with the AI model.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the `streamlit_app` directory with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
GALILEO_PROJECT_NAME=your_galileo_project_name
GALILEO_LOG_STREAM_NAME=your_galileo_log_stream_name
```

## Running the Application

Run the Streamlit app:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501` by default.

## Features

- Chat interface with message history
- Configuration sidebar for:
  - RAG toggle
  - Namespace selection
  - Top K value adjustment
  - System prompt customization
- Galileo logging integration
- Tool support (e.g., getTickerSymbol)
- Error handling and display

## Usage

1. Open the application in your browser
2. Configure settings in the sidebar if needed
3. Type your message in the chat input
4. View the AI's response in the chat interface
5. Continue the conversation as needed

## Notes

- The application uses GPT-4 by default
- All interactions are logged to Galileo for monitoring and analysis
- Tool calls are supported and will be displayed in the chat 