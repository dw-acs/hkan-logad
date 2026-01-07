import logging
import sys

from loguru import logger

third_party_libs = ["langfuse"]

# Disable logs for third party libs
for lib in third_party_libs:
    log = logging.getLogger(lib)
    log.setLevel(logging.CRITICAL)
    log.handlers = [logging.NullHandler()]

# Customize loguru logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>[{time:YYYY-MM-DD HH:mm:ss.SSS}]</green> | <level>{level: <8}</level> | <level>{message}</level>",
    colorize=True,
)
