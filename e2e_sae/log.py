"""Setup a logger to be used in all modules in the library.

To use the logger, import it in any module and use it as follows:

    ```
    from e2e_sae.log import logger
    logger.info("Info message")
    logger.warning("Warning message")
    ```
"""

import logging
from logging.config import dictConfig
from pathlib import Path

DEFAULT_LOGFILE = Path(__file__).resolve().parent.parent / "logs" / "logs.log"


def setup_logger(logfile: Path = DEFAULT_LOGFILE) -> logging.Logger:
    """Setup a logger to be used in all modules in the library.

    Sets up logging configuration with a console handler and a file handler.
    Console handler logs messages with INFO level, file handler logs WARNING level.
    The root logger is configured to use both handlers.

    Returns:
        logging.Logger: A configured logger object.

    Example:
        >>> logger = setup_logger()
        >>> logger.debug("Debug message")
        >>> logger.info("Info message")
        >>> logger.warning("Warning message")
    """
    if not logfile.parent.exists():
        logfile.parent.mkdir(parents=True, exist_ok=True)

    logging_config = {
        "version": 1,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": "INFO",
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": str(logfile),
                "formatter": "default",
                "level": "WARNING",
            },
        },
        "root": {
            "handlers": ["console", "file"],
            "level": "INFO",
        },
    }

    dictConfig(logging_config)
    return logging.getLogger()


logger = setup_logger()
