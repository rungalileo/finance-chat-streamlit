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
openai_api_key = "your_openai_api_key"
pinecone_api_key = "your_pinecone_api_key"
galileo_console_url = "your_galileo_console_url"
pinecone_index_name = "galileo-demo"
pinecone_namespace = "sp500-qa-demo"
galileo_api_key = "your_galileo_api_key"
galileo_project = "your_galileo_project"
galileo_log_stream = "your_galileo_log_stream"
ALPHA_VANTAGE_API_KEY= "your_alpha_vantage_key"
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
- Ticker symbol lookup: Using the company name (uses AlphaVantage API key or mocks the data if over API limit)
- Get Stock price: Using the ticker symbol (uses AlphaVantage API key or mocks the data if over API limit) 
- RAG System: Uses Pinecone vector store for semantic search over financial transcripts

## Observability

The application uses Galileo for:
- Logging chat interactions
- Tracking tool usage
- Monitoring RAG performance
- Debugging and error tracking
