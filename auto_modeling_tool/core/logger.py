"""
Logging configuration for AutoModelTool.

Provides a centralized, colored logging setup for the entire package.
"""

import logging
import sys
from typing import Optional

# Try to import colorlog for colored output
try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False


# Global logger instance
class EncodingSafeStreamHandler(logging.StreamHandler):
    """Keep logs usable on terminals whose encoding cannot represent emoji."""

    def emit(self, record: logging.LogRecord) -> None:
        # StreamHandler.emit handles UnicodeEncodeError internally via
        # handleError(), so format/write explicitly to make replacement work.
        try:
            message = self.format(record)
            self.stream.write(message + self.terminator)
            self.flush()
        except UnicodeEncodeError:
            try:
                encoding = getattr(self.stream, "encoding", None) or "utf-8"
                safe_message = message.encode(encoding, errors="replace").decode(
                    encoding, errors="replace"
                )
                self.stream.write(safe_message + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)
        except Exception:
            self.handleError(record)


logger = logging.getLogger("AutoModel")
logger.setLevel(logging.INFO)
logger.propagate = False

# Prevent duplicate handlers
if not logger.handlers:
    handler = EncodingSafeStreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    if COLORLOG_AVAILABLE:
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s[AutoModel] %(asctime)s - %(levelname)s - %(message)s%(reset)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            }
        )
    else:
        formatter = logging.Formatter(
            "[AutoModel] %(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)


def set_log_level(level: str) -> None:
    """
    Set the logging level for AutoModel logger.

    Parameters
    ----------
    level : str
        Log level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'

    Example
    -------
    >>> set_log_level("DEBUG")
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    for handler in logger.handlers:
        handler.setLevel(numeric_level)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a child logger for a specific module.

    Parameters
    ----------
    name : str, optional
        Module name for the child logger.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    if name:
        return logger.getChild(name)
    return logger
