# -*- coding: utf-8 -*-
import os
import logging
import logging.handlers
import time


def init_logger(log_file):
    dir_path = os.path.dirname(log_file)
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except Exception as e:
        pass
    
    handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=30 * 1024 * 1024, backupCount=10)
    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    #logger_instance = logging.getLogger('logs')
    logger_instance = logging.getLogger(log_file.split("/")[-1])
    logger_instance.addHandler(handler)
    #logger_instance.setLevel(logging.DEBUG)
    logger_instance.setLevel(logging.INFO)
    return logger_instance
