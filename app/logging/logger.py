import logging
import sys

def get_logger(name: str):
    logger = logging.getLogger(name)

    # 이미 같은 스트림핸들러가 있으면 재등록 안 함
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    logger.propagate = False  # root(Uvicorn)로 전달 막기

    # 불필요한 라이브러리 로그 끄기
    for noisy in ["uvicorn", "uvicorn.access", "uvicorn.error", "sqlalchemy.engine", "sqlalchemy.pool"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
        logging.getLogger(noisy).propagate = False

    return logger
