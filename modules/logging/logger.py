import enum
import logging
import os


class LoggerFormat(enum.Enum):
    CAMERA = "%(id)s, %(x_coordinate)s, %(y_coordinate)s, %(camera_name)s, %(day)s, %(time)s, %(frame)s, %(score)s"
    SYSTEM = "%(asctime)s - %(levelname)s - %(message)s"


def setup_logger(filename, format=LoggerFormat.CAMERA):
    # Create a logger
    logger = logging.getLogger(format.name)

    # Set the logging level
    logger.setLevel(logging.DEBUG)

    # Create a file handler with the given filename
    file_handler = logging.FileHandler(filename, encoding="utf-8")

    # Create a formatter
    formatter = logging.Formatter(format.value)

    # Add the formatter to the file handler
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger


def log_file_size(input_path, output_path, logger):
    input_file_size = os.path.getsize(input_path) / (1024 * 1024)
    output_file_size = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(
        f"Input file size: {input_file_size:.2f}MB\t\
        Output file size: {output_file_size:.2f}MB\t\
        Difference: {(output_file_size - input_file_size):.2f}MB"
    )