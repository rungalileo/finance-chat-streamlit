# Finance Transcripts Bot

A Streamlit-based chat application that uses Retrieval Augmented Generation (RAG) to analyze financial transcripts and provide intelligent responses. The application includes tools for stock trading simulation and observability through Galileo.

## Key Features

- RAG-powered question answering over financial transcripts
- Stock trading simulation with purchase/sell functionality
- Galileo observability integration
- Configurable AI models (GPT-4 or GPT-3.5 Turbo)
- Experiment runner for batch processing
- Streamlit web interface with real-time chat

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── experiment_runner.py   # (Unused) Script for experiment runner 
├── generate_env.py        # (Unused) Script to convert secrets.toml to .env
├── galileo_api_helper.py  # Helper functions for Galileo API
├── pages/                 # Streamlit page components
│   └── 2_run_experiment.py
├── tools/                 # Financial tools
│   ├── purchase_stocks.py
│   ├── sell_stocks.py
│   └── get_stock_price.py
│   └── get_ticker_symbol.py
└── log_hallucination.py  # Hallucination logging utility
```

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up secrets in .streamlit/secrets.toml

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
galileo_console_url = your_galileo_console_url
pinecone_index_name = "galileo-demo"
pinecone_namespace = "sp500-qa-demo"
GALILEO_API_KEY=your_galileo_api_key
GALILEO_PROJECT=your_galileo_project
GALILEO_LOG_STREAM=your_galileo_log_stream
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
```

## Running the Application

### Streamlit UI

```bash
streamlit run app.py
```

The application will be available at http://localhost:8501

## Usage

The application provides:

1. Interactive chat interface for financial queries
2. Stock trading simulation tools
3. Batch experiment processing
4. Galileo observability for monitoring and debugging

## Tools

- Stock Purchase Simulation: Simulates buying stocks with specified ticker, quantity, and price
- Stock Sale Simulation: Simulates selling stocks with specified ticker, quantity, and price
- RAG System: Uses Pinecone vector store for semantic search over financial transcripts

## Observability

The application uses Galileo for:
- Logging chat interactions
- Tracking tool usage
- Monitoring RAG performance
- Debugging and error tracking

## License

[MIT License](LICENSE)

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Support

For support, please open an issue in the repository.

## Acknowledgments

- OpenAI for GPT models
- Pinecone for vector search
- Galileo for observability
- Streamlit for the web interface
- Alpha Vantage for stock data
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
