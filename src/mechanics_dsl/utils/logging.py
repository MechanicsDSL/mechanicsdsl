"""
Logging utilities for MechanicsDSL
"""
import logging
from typing import Optional

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def setup_logging(level: int = logging.INFO, 
                 log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        level: Logging level
        log_file: Optional file to write logs to
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger('MechanicsDSL')
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


# Global logger instance
logger = setup_logging()
