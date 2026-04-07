from tensorflow.python.platform import tf_logging
import json, logging, os, time, sys, datetime


# define log format
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(filename)-16s:%(lineno)-3d pid:%(process)s %(threadName)s -- %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S ')
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(formatter)
logger = tf_logging._get_logger()
logger.handlers.pop()
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False
tflogger = logger
tflogger.info("hello")
