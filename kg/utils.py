import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

LOGS_DIRECTORY_PATH = Path('../logs')
# Number of processes to use when executing functions in parallel.
MAX_PROCESS_COUNT = 20


def set_up_logging(log_name):
    """Set up a logger that logs to both the console and a file."""
    log_path = LOGS_DIRECTORY_PATH / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    # Remove the default handler.
    logger.handlers.clear()
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    return logger


def strip_and_remove_empty_strings(strings):
    """Strip a list of strings and remove empty strings."""
    strings = [string.strip() for string in strings]
    strings = [string for string in strings if string]
    return strings


def execute_function_in_parallel(function: Callable, *argument_lists,
                                 logger=None):
    """Execute a function in parallel using multiple processes."""
    with ProcessPoolExecutor(max_workers=MAX_PROCESS_COUNT) as executor:
        futures = [executor.submit(function, *arguments)
                   for arguments in zip(*argument_lists)]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exception:
                if logger:
                    logger.exception('Exception')
                raise exception
