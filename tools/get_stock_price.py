import json
import time
import os
import requests
import logging
from typing import Optional
from galileo import GalileoLogger
import streamlit as st

# Mock database for testing
MOCK_PRICE_DB = {
    "AVGO": {  # Broadcom
        "price": 184.72,
        "change": -2.68,
        "change_percent": -1.43,
        "volume": 502,
        "high": 186.34,
        "low": 184.52,
        "open": 186.24
    },
    "GPS": {  # Gap
        "price": 19.85,
        "change": 0.15,
        "change_percent": 0.76,
        "volume": 4567890,
        "high": 20.00,
        "low": 19.50,
        "open": 19.70
    },
    "AAPL": {  # Apple
        "price": 178.72,
        "change": 1.23,
        "change_percent": 0.69,
        "volume": 52345678,
        "high": 179.50,
        "low": 177.80,
        "open": 178.00
    },
    "MSFT": {  # Microsoft
        "price": 415.32,
        "change": 2.45,
        "change_percent": 0.59,
        "volume": 23456789,
        "high": 416.00,
        "low": 413.50,
        "open": 414.00
    },
    "GOOGL": {  # Google
        "price": 147.68,
        "change": -0.82,
        "change_percent": -0.55,
        "volume": 34567890,
        "high": 148.50,
        "low": 147.20,
        "open": 147.90
    },
    "AMZN": {  # Amazon
        "price": 178.75,
        "change": 1.25,
        "change_percent": 0.70,
        "volume": 45678901,
        "high": 179.00,
        "low": 177.50,
        "open": 178.00
    },
    "META": {  # Meta
        "price": 485.58,
        "change": 3.42,
        "change_percent": 0.71,
        "volume": 56789012,
        "high": 486.00,
        "low": 482.00,
        "open": 483.00
    },
    "TSLA": {  # Tesla
        "price": 177.77,
        "change": -2.33,
        "change_percent": -1.29,
        "volume": 67890123,
        "high": 180.00,
        "low": 177.00,
        "open": 179.00
    },
    "NVDA": {  # NVIDIA
        "price": 950.02,
        "change": 15.98,
        "change_percent": 1.71,
        "volume": 78901234,
        "high": 952.00,
        "low": 945.00,
        "open": 946.00
    }
}

def _log_to_galileo(galileo_logger: GalileoLogger, ticker: str, result: dict, start_time: float) -> None:
    """
    Helper function to log stock price lookup to Galileo.
    
    Args:
        galileo_logger: Galileo logger for observability
        ticker: The ticker symbol that was looked up
        result: The price data found
        start_time: The start time of the lookup operation
    """
    galileo_logger.add_tool_span(
        input=json.dumps({"ticker": ticker}),
        output=json.dumps(result),
        name="Get Stock Price",
        duration_ns=int((time.time() - start_time) * 1000000),
        metadata={
            "ticker": ticker,
            "price": str(result["price"]),
            "found": "true"
        },
        tags=["stocks", "price", "lookup"]
    )

def get_stock_price(ticker: str, galileo_logger: GalileoLogger) -> str:
    """
    Get the current stock price and other market data for a given ticker symbol.
    Falls back to mock database if API call fails.
    
    Args:
        ticker: The ticker symbol to look up
        galileo_logger: Galileo logger for observability
        
    Returns:
        JSON string containing the stock price and market data
    """
    start_time = time.time()
    try:
        # First try the API
        api_key = st.secrets["alpha_vantage_api_key"]
        
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={api_key}"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        if "Global Quote" in data and data["Global Quote"]:
            quote = data["Global Quote"]
            result = {
                "price": float(quote["05. price"]),
                "change": float(quote["09. change"]),
                "change_percent": float(quote["10. change percent"].rstrip('%')),
                "volume": int(quote["06. volume"]),
                "high": float(quote["03. high"]),
                "low": float(quote["04. low"]),
                "open": float(quote["02. open"])
            }
            _log_to_galileo(galileo_logger, ticker, result, start_time)
            return json.dumps(result)
        
        # If no matches found in API, try mock database
        if ticker in MOCK_PRICE_DB:
            logging.info(f"Found {ticker} in mock database")
            result = MOCK_PRICE_DB[ticker]
            _log_to_galileo(galileo_logger, ticker, result, start_time)
            return json.dumps(result)
            
        # If not found in either, return a default mock price
        logging.info(f"Ticker {ticker} not found, using default mock price")
        result = {
            "price": 100.00,
            "change": 0.00,
            "change_percent": 0.00,
            "volume": 1000,
            "high": 101.00,
            "low": 99.00,
            "open": 100.00
        }
        _log_to_galileo(galileo_logger, ticker, result, start_time)
        return json.dumps(result)
        
    except Exception as e:
        logging.error(f"Error getting stock price: {str(e)}")
        
        # On any error, try the mock database
        if ticker in MOCK_PRICE_DB:
            logging.info(f"Found {ticker} in mock database after API error")
            result = MOCK_PRICE_DB[ticker]
            _log_to_galileo(galileo_logger, ticker, result, start_time)
            return json.dumps(result)
            
        # If not found in mock database, return a default mock price
        logging.info(f"Ticker {ticker} not found in mock database, using default mock price")
        result = {
            "price": 100.00,
            "change": 0.00,
            "change_percent": 0.00,
            "volume": 1000,
            "high": 101.00,
            "low": 99.00,
            "open": 100.00
        }
        _log_to_galileo(galileo_logger, ticker, result, start_time)
        return json.dumps(result) 