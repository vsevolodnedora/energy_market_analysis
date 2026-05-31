import logging
import os
import sys
from dotenv import load_dotenv


load_dotenv()  # Load environment variables from .env

# Detect if running inside GitHub Actions
GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

# Get log level from environment variable (default to INFO)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Validate log level (fallback to INFO if invalid)
VALID_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
if LOG_LEVEL not in VALID_LOG_LEVELS:
    LOG_LEVEL = "INFO"

# ANSI color codes for terminal output
class ANSIColors:
    DEBUG = "\033[36m"    # Cyan
    INFO = "\033[32m"     # Green
    WARNING = "\033[33m"  # Yellow
    ERROR = "\033[31m"    # Red
    CRITICAL = "\033[1;31m"  # Bold Red
    RESET = "\033[0m"     # Reset color

# Custom formatter with ANSI colors (if not in GitHub Actions)
class CustomFormatter(logging.Formatter):
    def format(self, record):
        log_message = super().format(record)

        if GITHUB_ACTIONS:
            # Use GitHub Actions logging syntax
            if record.levelno == logging.DEBUG:
                return f"::debug::{log_message}"
            elif record.levelno == logging.WARNING:
                return f"::warning::{log_message}"
            elif record.levelno >= logging.ERROR:
                return f"::error::{log_message}"
            return log_message  # INFO level remains unchanged

        # Apply ANSI colors for local terminal
        log_color = {
            logging.DEBUG: ANSIColors.DEBUG,
            logging.INFO: ANSIColors.INFO,
            logging.WARNING: ANSIColors.WARNING,
            logging.ERROR: ANSIColors.ERROR,
            logging.CRITICAL: ANSIColors.CRITICAL,
        }.get(record.levelno, ANSIColors.RESET)

        return f"{log_color}{log_message}{ANSIColors.RESET}"

# Configure the handler
console_handler = logging.StreamHandler(sys.stdout)
formatter = CustomFormatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
console_handler.setFormatter(formatter)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    handlers=[console_handler],
)

def get_logger(name: str):
    """Returns a configured logger for the given module."""
    return logging.getLogger(name.split(".")[-1])
