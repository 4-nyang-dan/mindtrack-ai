# app/db.py
import os
import time
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
import logging

DB_USER = os.getenv("POSTGRES_USER", "4nyangdan")
DB_PASS_RAW = os.getenv("POSTGRES_PASSWORD", "")
DB_PASS = quote_plus(DB_PASS_RAW)  # 특수문자 안전 인코딩
DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "postgres")

# sqlalchemy 디버그 끄기
for name in ("sqlalchemy.engine", "sqlalchemy.pool", "sqlalchemy.orm", "sqlalchemy.dialects"):
    logging.getLogger(name).setLevel(logging.WARNING)

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    f"?connect_timeout=5"
)
print("DB URL:", DATABASE_URL)

engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,  # 죽은 커넥션 자동감지
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def wait_for_db(max_attempts: int = 20, delay: float = 1.5) -> None:
    """DB가 실제로 응답할 때까지 재시도"""
    last_err = None
    for _ in range(max_attempts):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return
        except Exception as e:
            last_err = e
            time.sleep(delay)
    raise last_err
