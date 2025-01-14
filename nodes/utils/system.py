"""
File    : system.py
Purpose : System-level functions for the ConfyUI-TinyBreaker project.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Nov 14, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ConfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import sys
import logging

class _CustomFormatter(logging.Formatter):
    """Custom formatter for the logger."""
    EMOJI  = "\U0001F4AA" # the emoji show before logger name
    COLOR  = "\033[0;33m" # yellow for the logger name color
    RESET  = "\033[0m"    # reset to default color
    COLORS = {
        logging.INFO    : "\033[0;32m",  # GREEN
        logging.DEBUG   : "\033[0;34m",  # BLUE
        logging.WARNING : "\033[0;33m",  # YELLOW
        logging.ERROR   : "\033[0;31m",  # RED
        logging.CRITICAL: "\033[1;31m",  # BOLD RED
    }

    def format(self, record):
        """Override the default format method."""
        # set color based on the log level and add an emoji before the logger name
        color = self.COLORS.get(record.levelno, self.RESET)
        record.name      = f"{self.EMOJI}{self.COLOR}{record.name}{self.RESET}"
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


# Create a logger instance and set the custom formatter.
logger = logging.getLogger("TinyBreaker")
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_CustomFormatter("[%(name)s %(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
