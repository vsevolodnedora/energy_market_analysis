import logging
import os
import sys

# Get log level from environment variable (default to INFO)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Validate log level (fallback to INFO if invalid)
VALID_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
if LOG_LEVEL not in VALID_LOG_LEVELS:
    LOG_LEVEL = "INFO"

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log to console
    ],
)

def get_logger(name: str):
    """Returns a configured logger for the given module."""
    return logging.getLogger(name)
