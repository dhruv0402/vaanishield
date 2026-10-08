"""Project-wide logger factory so log format is consistent across scripts/ and
vaanishield/. Prefer this over bare print() statements everywhere, especially in the
dataset-construction scripts where reproducibility of what ran matters."""

import logging


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
