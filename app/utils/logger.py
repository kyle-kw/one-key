
import sys
from loguru import logger

logger.remove(0)

logger.add("logs/info.log", rotation="10 MB", compression="zip")
logger.add(sys.stdout, level="INFO")
