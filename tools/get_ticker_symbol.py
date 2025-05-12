import json
import time
import os
import requests
import logging
from typing import Optional
from galileo import GalileoLogger
import streamlit as st

# Mock database for testing
MOCK_TICKER_DB = {
    "broadcom": "AVGO",
    "gap": "GPS",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "amd": "AMD",
    "intel": "INTC",
    "ibm": "IBM",
    "oracle": "ORCL",
    "salesforce": "CRM",
    "adobe": "ADBE",
    "netflix": "NFLX",
    "disney": "DIS",
    "walmart": "WMT",
    "target": "TGT",
    "costco": "COST",
    "home depot": "HD"
}

def _log_to_galileo(galileo_logger: GalileoLogger, company: str, ticker: str, start_time: float) -> None:
    """
    Helper function to log ticker lookup to Galileo.
    
    Args:
        galileo_logger: Galileo logger for observability
        company: The company name that was looked up
        ticker: The ticker symbol found
        start_time: The start time of the lookup operation
    """
    galileo_logger.add_tool_span(
        input=json.dumps({"company": company}),
        output=json.dumps({"ticker": ticker}),
        name="Get Ticker Symbol",
        duration_ns=int((time.time() - start_time) * 1000000),
        metadata={
            "company": company,
            "ticker": ticker,
            "found": "true"
        },
        tags=["stocks", "ticker", "lookup"]
    )

def get_ticker_symbol(company: str, galileo_logger: GalileoLogger) -> str:
    """
    Get the ticker symbol for a company.
    Falls back to mock database if API call fails.
    
    Args:
        company: The company name to look up
        galileo_logger: Galileo logger for observability
        
    Returns:
        The ticker symbol for the company
    """
    start_time = time.time()
    try:
        # First try the API
        api_key = st.secrets["alpha_vantage_api_key"]
        
        url = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={company}&apikey={api_key}"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        if "bestMatches" in data and data["bestMatches"]:
            ticker = data["bestMatches"][0]["1. symbol"]
            _log_to_galileo(galileo_logger, company, ticker, start_time)
            return ticker
        
        # If no matches found in API, try mock database
        company_lower = company.lower()
        if company_lower in MOCK_TICKER_DB:
            logging.info(f"Found {company} in mock database")
            ticker = MOCK_TICKER_DB[company_lower]
            _log_to_galileo(galileo_logger, company, ticker, start_time)
            return ticker
            
        # If not found in either, generate a mock ticker
        logging.info(f"Company {company} not found, generating mock ticker")
        ticker = f"{company[:4].upper()}.MOCK"
        _log_to_galileo(galileo_logger, company, ticker, start_time)
        return ticker
        
    except Exception as e:
        logging.error(f"Error getting ticker symbol: {str(e)}")
        
        # On any error, try the mock database
        company_lower = company.lower()
        if company_lower in MOCK_TICKER_DB:
            logging.info(f"Found {company} in mock database after API error")
            ticker = MOCK_TICKER_DB[company_lower]
            _log_to_galileo(galileo_logger, company, ticker, start_time)
            return ticker
            
        # If not found in mock database, generate a mock ticker
        logging.info(f"Company {company} not found in mock database, generating mock ticker")
        ticker = f"{company[:4].upper()}.MOCK"
        _log_to_galileo(galileo_logger, company, ticker, start_time)
        return ticker 