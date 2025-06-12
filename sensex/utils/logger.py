import logging
import os
from logging.handlers import RotatingFileHandler


def get_logger(name: str, log_file: str, level: int=logging.INFO) -> logging.Logger:
    """
    Returns a configured logger.

    Parameters
    ----------
    name : str
        Logger name (usually `__name__`)
    log_file : str
        Optional path to a log file.
    level : int
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns
    -------
        logger : logging.Logger
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times in Jupyter or re-imported modules
    if logger.handlers:
        return logger

    # Formatter
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
