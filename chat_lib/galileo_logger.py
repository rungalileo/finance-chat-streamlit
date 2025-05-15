"""
Shared Galileo logger utilities for both Streamlit and Flask applications.
"""
import logging
from galileo import GalileoLogger

logger_debug = logging.getLogger(__name__)

def initialize_galileo_logger(project_name: str, log_stream: str) -> GalileoLogger:
    """
    Initialize a Galileo logger with the specified project and log stream.
    
    Args:
        project_name: The Galileo project name
        log_stream: The Galileo log stream name
        
    Returns:
        An initialized GalileoLogger instance
    """

    # Log the initialization
    logger_debug.info(f"Initializing Galileo logger - Project: {project_name}, Log Stream: {log_stream}")
    
    # Create and return a new logger
    return GalileoLogger(
        project=project_name,
        log_stream=log_stream
    ) 