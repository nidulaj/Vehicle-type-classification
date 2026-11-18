"""Centralized logging configuration for consistent output across all modules.

Provides a unified logging format with timestamp, log level, module name, and message,
supporting both console output and optional rotating/persisted file logs.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

# Default log format used across the entire project
DEFAULT_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "vehicle_classification",
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> logging.Logger:
    """Configure and return a standardized logger instance.

    If the logger already has configured handlers, existing handlers are preserved
    to prevent duplicate log message emissions.

    Args:
        name: Name of the logger (typically __name__ of the calling module).
        level: Logging level (e.g., logging.INFO, "DEBUG", "INFO", "WARNING").
        log_file: Optional path to a file where logs should be appended.
        log_format: Format string for log messages.
        date_format: Format string for timestamps.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Normalize log level string if passed as string
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger.setLevel(level)

    # Avoid adding duplicate handlers if the logger was already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    # Console Handler (writes to sys.stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional File Handler
    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(file_path), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent logs from propagating up to root logger if root has separate handlers
    logger.propagate = False

    return logger


def get_logger(name: str = "vehicle_classification") -> logging.Logger:
    """Convenience helper to retrieve an existing logger or create one with defaults.

    Args:
        name: Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    return setup_logger(name=name)

