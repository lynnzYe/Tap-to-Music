"""
Author: Lynn Ye
Created on: 2025/11/12
Brief: 
"""
import logging
import sys

from ttm.config import LOG_LEVEL


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[90m[Debug]\033[0m",
        logging.INFO: "\033[34m[Info]\033[0m",
        logging.WARNING: "\033[33m[Warning]\033[0m",
        logging.ERROR: "\033[31m[Error]\033[0m",
        logging.CRITICAL: "\033[1;31m[CRITICAL]\033[0m",
    }

    def format(self, record):
        level_color = self.COLORS.get(record.levelno, "")
        log_fmt = f"{level_color} %(asctime)s | %(message)s"
        formatter = logging.Formatter(log_fmt, "%H:%M:%S")
        return formatter.format(record)


def formatargs(*args):
    """Join multiple args into a space-separated string."""
    return " ".join(map(str, args))


class Log:
    def __init__(self, name="app", level=logging.DEBUG):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Only add handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(ColorFormatter())
            self.logger.addHandler(handler)

    def _log(self, level, *args):
        msg = formatargs(*args)
        self.logger.log(level, msg)

    def set_level(self, level): self.logger.setLevel(level)

    def debug(self, *args): self._log(logging.DEBUG, *args)

    def info(self, *args): self._log(logging.INFO, *args)

    def warn(self, *args): self._log(logging.WARNING, *args)

    def error(self, *args): self._log(logging.ERROR, *args)

    def crit(self, *args): self._log(logging.CRITICAL, *args)


log = Log("color_log")


def main():
    log.info('hello', 'world', 1 + 1)
    log.set_level(LOG_LEVEL)
    log.debug('this is a debug message')


if __name__ == "__main__":
    main()
