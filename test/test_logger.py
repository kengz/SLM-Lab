from unity_lab.lib import logger


def test_logger(test_multiline_str):
    logger.critical(test_multiline_str)
    logger.debug(test_multiline_str)
    logger.error(test_multiline_str)
    logger.exception(test_multiline_str)
    logger.info(test_multiline_str)
    logger.warn(test_multiline_str)
