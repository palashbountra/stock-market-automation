"""
utils/logger.py
---------------
Centralised logging setup. Every module calls get_logger(__name__).
Logs go to both console and logs/trading.log.
"""

import logging
import sys
from pathlib import Path
from utils.config_loader import cfg

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_LOG_LEVEL = getattr(logging, cfg.get("system", {}).get("log_level", "INFO").upper(), logging.INFO)

_fmt = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# File handler
_fh = logging.FileHandler(_LOG_DIR / "trading.log", encoding="utf-8")
_fh.setFormatter(_fmt)

# Console handler
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(_fmt)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(_LOG_LEVEL)
        logger.addHandler(_fh)
        logger.addHandler(_ch)
        logger.propagate = False
    return logger
