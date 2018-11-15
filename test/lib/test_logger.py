from slm_lab.lib import logger


def test_logger(test_str):
    logger.critical(test_str)
    logger.debug(test_str)
    logger.error(test_str)
    logger.exception(test_str)
    logger.info(test_str)
    logger.warn(test_str)
