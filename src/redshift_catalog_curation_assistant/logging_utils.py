from __future__ import annotations

import logging
import sys

PACKAGE_LOGGER_NAME = "redshift_catalog_curation_assistant"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
TERMINAL_HANDLER_NAME = "rcca-terminal"


def configure_logging(level: str | int = logging.INFO) -> None:
    """Configure concise terminal logging for package commands."""
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    for handler in list(logger.handlers):
        if handler.get_name() == TERMINAL_HANDLER_NAME:
            logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    handler.set_name(TERMINAL_HANDLER_NAME)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
