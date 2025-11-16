"""
Some utility functions for the project
"""

import logging
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_agent_save_path(agent_name):
    return os.path.join(ROOT_DIR, 'agents', agent_name)

def set_logger(logger_name, debug_level=logging.DEBUG):
    
    # class logger configurations
    # https://stackoverflow.com/a/56944256
    import logging

    class ColoredFormatter(logging.Formatter):
        """
        A custom logging formatter that adds color to log messages based on their level.
        """
        # Define ANSI escape codes for different colors
        GREY = "\x1b[38;20m"
        YELLOW = "\x1b[33;20m"
        RED = "\x1b[31;20m"
        BOLD_RED = "\x1b[31;1m"
        RESET = "\x1b[0m"

        FORMAT = "%(asctime)s - %(name)s - %(levelname)s: %(message)s"

        FORMATS = {
            logging.DEBUG: GREY + FORMAT + RESET,
            logging.INFO: GREY + FORMAT + RESET,
            logging.WARNING: YELLOW + FORMAT + RESET,
            logging.ERROR: RED + FORMAT + RESET,
            logging.CRITICAL: BOLD_RED + FORMAT + RESET
        }

        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt)
            return formatter.format(record)

    # Configure the logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(debug_level)

    # Create a stream handler and set the custom formatter
    ch = logging.StreamHandler()
    # ch.setLevel(logging.DEBUG)
    ch.setFormatter(ColoredFormatter())

    # Add the handler to the logger
    logger.addHandler(ch)
    return logger
    # class logger ends

