import os
import sys
import time
import logging
from logging.handlers import RotatingFileHandler


def setup_logger(log_level=logging.INFO):
    # 初始化logger对象
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # 日志目录
    log_file_folder = os.path.join("logs", "inference")
    os.makedirs(log_file_folder, exist_ok=True)
    
    # 文件名，以日期作为文件名
    log_file_name = 'inference-logger-' + time.strftime('%Y-%m-%d', time.localtime(time.time())) + '.log'
    log_file_str = log_file_folder + os.sep + log_file_name

    # 创建日志记录器，指明日志保存路径,每个日志的大小，保存日志的上限
    # 文件日志
    file_log_handler = RotatingFileHandler(log_file_str, maxBytes=1024 * 1024, backupCount=10, encoding='UTF-8')
    # 控制台日志
    console_log_handler = logging.StreamHandler(stream=sys.stdout)
    
    # 设置日志的格式                   发生时间       日志等级      日志信息文件名     函数名          行数        日志信息
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s')

    # 将日志记录器指定日志的格式
    file_log_handler.setFormatter(formatter)
    console_log_handler.setFormatter(formatter)

    # 日志记录器
    logger.addHandler(file_log_handler)
    logger.addHandler(console_log_handler)

    return logger


logger = setup_logger()
