import logging , sys 

def get_logger(name, flog_name, stdout=True):

    logger = logging.getLogger(name)
    h_file = logging.FileHandler(flog_name)
    h_file.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(h_file)
    logger.setLevel(logging.INFO)
    if stdout:
        h_stdout = logging.StreamHandler(sys.stdout)
        logger.addHandler(h_stdout)
    return logger 


if __name__ == '__main__':
    logger1 = get_logger('log1', 'log1.log')
    logger2 = get_logger('log2', 'log2.log', False)
    logger1.info('test')
    logger2.info('test')