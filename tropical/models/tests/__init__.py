"""Unit test base class for Feersum NLP Engine Python Module models."""
import unittest
import logging


def reset_logging():
    """ Remove the handlers and filters from the logger to prevent, for example,
    handlers stacking up over multiple runs."""
    root = logging.getLogger()

    for handler in list(root.handlers):  # list(...) makes a copy of the handlers list.
        root.removeHandler(handler)
        handler.close()

    for filter in list(root.filters):  # list(...) makes a copy of the handlers list.
        root.removeFilter(filter)


def setup_logging():
    logger = logging.getLogger()  # Root logger.

    # Log Levels!:
    # CRITICAL 50
    # ERROR    40
    # WARNING  30
    # INFO     20
    # DEBUG    10
    # NOTSET    0

    logger.setLevel(logging.DEBUG)

    # Create console handler.
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter and add it to the handlers.
    formatter = logging.Formatter('%(asctime)s, %(name)s, %(levelname)s, %(message)s')
    ch.setFormatter(formatter)

    # Add the handlers to logger.
    logger.addHandler(ch)


class BaseTestCase(unittest.TestCase):
    def setUp(self):
        setup_logging()

    def tearDown(self):
        reset_logging()
