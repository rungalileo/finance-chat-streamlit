import json
import time
import random
import logging
from typing import Optional
from galileo import GalileoLogger

def _log_to_galileo(galileo_logger: GalileoLogger, ticker: str, quantity: int, price: float, order_id: str, start_time: float) -> None:
    """
    Helper function to log stock purchase to Galileo.
    
    Args:
        galileo_logger: Galileo logger for observability
        ticker: The ticker symbol being purchased
        quantity: Number of shares being purchased
        price: Price per share
        order_id: The generated order ID
        start_time: The start time of the purchase operation
    """
    galileo_logger.add_tool_span(
        input=json.dumps({
            "ticker": ticker,
            "quantity": quantity,
            "price": price
        }),
        output=json.dumps({
            "order_id": order_id,
            "total_cost": quantity * price,
            "fees": 10.00
        }),
        name="Purchase Stocks",
        duration_ns=int((time.time() - start_time) * 1000000),
        metadata={
            "ticker": ticker,
            "quantity": str(quantity),
            "price": str(price),
            "order_id": order_id
        },
        tags=["stocks", "purchase", "trade"]
    )

def purchase_stocks(ticker: str, quantity: int, price: float, galileo_logger: GalileoLogger) -> str:
    """
    Simulate purchasing stocks with a given ticker symbol, quantity, and price.
    
    Args:
        ticker: The ticker symbol to purchase
        quantity: The number of shares to purchase
        price: The price per share
        galileo_logger: Galileo logger for observability
        
    Returns:
        JSON string containing the order confirmation
    """
    start_time = time.time()
    try:
        # Generate a random order ID
        order_id = f"ORD-{random.randint(100000, 999999)}"
        
        # Calculate total cost including fees
        total_cost = quantity * price
        fees = 10.00  # Fixed fee for simplicity
        total_with_fees = total_cost + fees
        
        # Create order confirmation
        result = {
            "order_id": order_id,
            "ticker": ticker,
            "quantity": quantity,
            "price": price,
            "total_cost": total_cost,
            "fees": fees,
            "total_with_fees": total_with_fees,
            "status": "completed",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Purchase of stocks completed successfully"
        }
        
        _log_to_galileo(galileo_logger, ticker, quantity, price, order_id, start_time)
        return json.dumps(result)
        
    except Exception as e:
        logging.error(f"Error purchasing stocks: {str(e)}")
        raise 