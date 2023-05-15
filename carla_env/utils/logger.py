"""Logging utilities.

This module provides a simple logging utility that can be used to log to a file
and stdout. It is based on the logging module from the standard library.

Example:
    >>> from carla_env.utils.logger import Logging
    >>> Logging.setup(filepath="outputs.log", level=logging.INFO)
    >>> logger = Logging.get_logger("my_logger")
    >>> logger.info("Hello world!")
    >>> logger.debug("This will not be logged.")

"""

import abc
import logging
import sys
from pathlib import Path
from typing import Dict, Union


class Logging(abc.ABC):
    """Logging utility. This class is a singleton.
    
    It can be used to log to a file and stdout. It is based on the logging module
    from the standard library. It is a singleton class, so it can be used as a base
    class for other classes that need to log.
    
    Attributes:
        _initialized_ (bool): Whether the logger has been initialized.
        _loggers_ (Dict[str, logging.Logger]): Dictionary of loggers.
        
    """

    _initialized_: bool = False
    _loggers_: Dict[str, logging.Logger] = {}

    @classmethod
    def setup(
        cls,
        filepath: Union[str, Path] = "outputs.log",
        level=logging.DEBUG,
        formatter="(%(asctime)s) [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ):
        """Setup the logger. This method should be called before using the logger.
        
        Args:
            filepath (Union[str, Path]): Path to the log file.
            level (int): Logging level.
            formatter (str): Logging formatter.
            datefmt (str): Logging date format.
            
        """
        cls.__filepath = Path(filepath) if isinstance(filepath, str) else filepath
        cls.__level = level
        cls.__format = formatter
        cls.__datefmt = datefmt

        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

        for logger in cls._loggers_.values():
            cls.set_logger(logger)

        cls._initialized_ = True

    @classmethod
    def get_logger(cls, name: str):
        """Get a logger with the given name.
        
        Args:
            name (str): Name of the logger.
            
        Returns:
            logging.Logger: Logger with the given name.
            
        """
        logger = logging.getLogger(name)
        if cls._initialized_:
            cls.set_logger(logger)

        cls._loggers_[name] = logger
        return logger

    @classmethod
    def set_logger(cls, logger: logging.Logger):
        """Set the logger to use the given logging level and formatter.
        
        Args:
            logger (logging.Logger): Logger to set.
            
        """
        logger.setLevel(cls.__level)

        for handler in logger.handlers:
            if (
                isinstance(handler, logging.StreamHandler)
                and handler.stream == sys.stdout
            ):
                logger.removeHandler(handler)

        cls.__filepath.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(cls.__filepath)
        handler.setLevel(cls.__level)

        formatter = logging.Formatter(cls.__format, datefmt=cls.__datefmt)
        handler.setFormatter(formatter)

        logger.addHandler(handler)
