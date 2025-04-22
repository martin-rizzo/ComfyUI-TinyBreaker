"""
File    : system.py
Purpose : System-level functions for the ComfyUI-TinyBreaker project.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Nov 14, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import sys
import logging
# ANSI colors
_GREEN     = "\033[0;32m"
_BLUE      = "\033[0;34m"
_YELLOW    = "\033[0;33m"
_RED       = "\033[0;31m"
_LIGHT_RED = "\033[1;31m"
_RESET     = "\033[0m"   # reset to default color

class _CustomFormatter(logging.Formatter):
    """Custom formatter for the logger."""
    EMOJI        = "\U0001F4AA"  # emoji shown before the log message
    NAME_COLOR   = _YELLOW       # logger name color
    LEVEL_COLORS = {
        logging.INFO    : _GREEN,
        logging.DEBUG   : _BLUE,
        logging.WARNING : _YELLOW,
        logging.ERROR   : _RED,
        logging.CRITICAL: _LIGHT_RED,
    }

    def format(self, record):
        """Override the default format method."""
        # set color based on the log level and add an emoji before the logger name
        level_color = self.LEVEL_COLORS.get(record.levelno, _RESET)
        record.name      = f"{self.EMOJI}{self.NAME_COLOR}{record.name}{_RESET}"
        record.levelname = f"{level_color}{record.levelname}{_RESET}"
        return super().format(record)


#======================= THE MAIN TINYBREAKER LOGGER =======================#

logger: logging.Logger = None

def setup_logger(name      : str  = "TB",
                 log_level : str  = "INFO",
                 use_stdout: bool = False
                 ):
    global logger
    if logger is not None:
        logger.warning("Logger already set up. Skipping setup_logger().")

    logger = logging.getLogger(name)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.strout if use_stdout else sys.stderr)
        handler.setFormatter(_CustomFormatter("[%(name)s %(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(log_level)
    if log_level=="DEBUG" or log_level==logging.DEBUG:
        logger.debug("Debug logging enabled.")

