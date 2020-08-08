import colorlog
from tqdm import tqdm
import logging


class TqdmHandler(logging.StreamHandler):
    """
    Makes log messages work through tqdm.write().
    From https://github.com/tqdm/tqdm/issues/193#issuecomment-233212170
    """

    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


class DuplicateFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.msgs = set()

    def filter(self, record):
        rv = record.msg not in self.msgs
        self.msgs.add(record.msg)
        return rv


def setup_custom_logger(name):
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(name)s | %(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "white",
            "SUCCESS:": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )

    handler = TqdmHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    dup_filter = DuplicateFilter()
    logger.addFilter(dup_filter)
    return logger
