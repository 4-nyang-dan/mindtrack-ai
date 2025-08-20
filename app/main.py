import os, sys, logging, io, threading, redis
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from .db import Base, engine, wait_for_db
from app.db import Base, engine
from app.worker import run_forever
import logging
from app.logging.logger import get_logger

logger = get_logger("mindtrack.fastapi")


app = FastAPI(title="mind-track AI", version="1.0.0")

r = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    decode_responses=False,
)

def _orig_key(uid: int, img_id: int) -> str:
    return f"user:{uid}:img:{img_id}"

@app.on_event("startup")
def on_startup():
    wait_for_db() # DB 가 열릴때까지 대기
    # 1) 테이블 생성
    Base.metadata.create_all(bind=engine)
    logger.info("[startup] DB 테이블 생성(필요 시) 완료")

    # 2) 워커 기동 - Redis+DB를 감시하며 분석 진행
    t = threading.Thread(target=run_forever, daemon=True, name="worker-thread")
    t.start()
    logger.info("[startup] 워커 스레드 시작")

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/inspect/original/{user_id}/{image_id}")
def inspect_original(user_id: int, image_id: int):
    key = _orig_key(user_id, image_id)
    raw = r.get(key)
    if not raw:
        logger.warning(f"[/inspect] 원본 없음 or TTL 만료 | key={key}")
        return JSONResponse(status_code=404, content={"error": "original not found or expired"})
    try:
        im = Image.open(io.BytesIO(raw))
        return {"width": im.width, "height": im.height, "mode": im.mode, "format": im.format}
    except Exception as e:
        logger.exception(f"[/inspect] 이미지 오픈 실패 | key={key}")
        return JSONResponse(status_code=500, content={"error": f"cannot open image: {e}"})

