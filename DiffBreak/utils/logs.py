import logging
from tqdm.contrib.logging import logging_redirect_tqdm

BAR_LEVEL_NUM = 11
logging.addLevelName(BAR_LEVEL_NUM, "BAR")


class CustomFormatter(logging.Formatter):

    grey = "\x1b[36m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + "%(name)s - %(levelname)s - %(message)s" + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
        BAR_LEVEL_NUM: grey + "%(message)s" + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_handler():
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    return ch


class DiffBreakLogger(logging.Logger):
    def info(self, msg, *args, **kwargs):
        with logging_redirect_tqdm(loggers=[self]):
            super().info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        with logging_redirect_tqdm(loggers=[self]):
            super().warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        with logging_redirect_tqdm(loggers=[self]):
            super().error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        with logging_redirect_tqdm(loggers=[self]):
            super().critical(msg, *args, **kwargs)

    def bar(self, msg, *args, **kwargs):
        with logging_redirect_tqdm(loggers=[self]):
            self._log(BAR_LEVEL_NUM, msg, args, **kwargs)


def get_logger():
    logging.setLoggerClass(DiffBreakLogger)
    logger = logging.getLogger("DiffBreak")
    logger.setLevel(logging.DEBUG)
    ch = get_handler()
    if not logger.hasHandlers():
        logger.addHandler(ch)
    logger.propagate = False
    return logger
