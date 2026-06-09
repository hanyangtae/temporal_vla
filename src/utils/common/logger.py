import datetime
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from colorlog import ColoredFormatter

LOG_PATH = "/temporal_vla/outputs/logs"

LOG_FORMAT = "%(asctime)s - %(process)d - %(levelname)s - %(message)s"
COLOR_LOG_FORMAT = "%(log_color)s%(levelname)-8s%(reset)s %(log_color)s%(message)s"


def _repo_log_path() -> Path:
    return Path(__file__).resolve().parents[3] / "outputs" / "logs"


def _get_console_handler():
    handler = logging.StreamHandler()
    formatter = ColoredFormatter(
        COLOR_LOG_FORMAT,
        log_colors={
            "DEBUG":    "cyan",
            "INFO":     "green",
            "WARNING":  "yellow",
            "ERROR":    "red",
            "CRITICAL": "bold_red",
        },
    )
    handler.setFormatter(formatter)
    return handler


def _get_file_handler(filepath, mode="a"):
    handler = logging.FileHandler(filepath, mode=mode)
    formatter = logging.Formatter(LOG_FORMAT)
    handler.setFormatter(formatter)
    return handler


def create_module_logger(
    module_name,
    module_log=False,
    level=logging.DEBUG,
    log_file_path: Optional[Path] = None,
    run_timestamp: Optional[str] = None,
):
    """
    Creates and configures a logger instance.

    This function is designed to work correctly in both single-process and multi-process
    (parallel) environments.

    - In a standard single-process context (when `log_file_path` is not provided),
      it returns a standard singleton logger based on `module_name`, ensuring
      consistency across the application.
    - In a multi-process context (when a specific `log_file_path` is provided,
      typically for a worker process), it creates a logger with a unique name
      to prevent handler conflicts (race conditions) between different processes.

    Args:
        module_name (str): The name of the module for the logger.
        module_log (bool): If True, creates a separate log file for the module.
        level (int): The logging level.
        log_file_path (Optional[Path]): A specific path for the log file. If provided,
            it signals a multi-process worker context.
        run_timestamp (Optional[str]): A fixed timestamp string (e.g., "20251031_2230")
            to use for naming log files, ensuring consistency.

    Returns:
        logging.Logger: The configured logger instance.
    """
    if log_file_path:
        # Worker process in a parallel run: create a unique logger to avoid conflicts.
        logger_name = f"{module_name}-{uuid.uuid4()}"
    else:
        # Standard context: use the module name for a shared, singleton logger.
        logger_name = module_name

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid adding handlers if they already exist, especially for the singleton logger.
    if logger.hasHandlers():
        return logger

    logger.addHandler(_get_console_handler())

    uses_default_log_path = False
    if log_file_path:
        log_file = log_file_path
    elif os.environ.get("PDK_LOG_FILE"):
        log_file = Path(os.environ["PDK_LOG_FILE"])
    else:
        # Default log file for general (non-worker) logging.
        timestamp = (
            run_timestamp
            if run_timestamp
            else datetime.datetime.now().strftime("%Y%m%d_%H%M")
        )
        log_file = Path(LOG_PATH) / "all_log" / f"{timestamp}.log"
        uses_default_log_path = True

    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        if not uses_default_log_path:
            raise
        log_file = _repo_log_path() / "all_log" / log_file.name
        log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.addHandler(_get_file_handler(log_file, mode="a"))

    if False:
        # Append process ID to module log file name to prevent mixing logs
        # from different parallel runs.
        pid = os.getpid()
        module_log_file = Path(LOG_PATH) / f"{module_name}-{pid}.log"
        module_log_file.parent.mkdir(parents=True, exist_ok=True)
        # Always use append mode to prevent data loss.
        logger.addHandler(_get_file_handler(module_log_file, mode="a"))

    return logger
