import os, sys, logging, io, threading, redis
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from app.db import Base, engine, wait_for_db
from app.worker import run_forever
from app.logging.logger import get_logger
from app.integration_service import IntegrationService
from app.common import _orig_key
from pydantic import BaseModel
from typing import Optional

# sys.path 보정
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


# 프로세스 내 중복 방지(전역 플래그 + Lock)
WORKER_STARTED = False
WORKER_LOCK = threading.Lock()

# FastAPI app
logger = get_logger("mindtrack.fastapi")
service = IntegrationService()
app = FastAPI(title="mind-track AI", version="1.0.0")


# Redis setup for session and image handling
r = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    decode_responses=False,
)

# 세션 키를 확인하는 함수
def check_session(user_id: int, session_id: str) -> bool:
    session_key = f"user:{user_id}:session"
    stored_session_id = r.get(session_key)
    return stored_session_id == session_id  # 세션 ID 일치 여부 반환

# 세션 확인 및 유효성 검사
def get_current_user(session_id: str, user_id: int) -> int:
    if not check_session(user_id, session_id):
        raise HTTPException(status_code=401, detail="Session expired or invalid")
    return user_id

# session_id를 받는 의존성 함수
def get_session_id(session_id: str):
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required")
    return session_id


# 애플리케이션 스타트업 훅
#  - DB 대기 → 스키마 생성 → (옵션) 워커 스레드 1회 기동
#  - --reload 환경에서도 중복 기동 방지
@app.on_event("startup")
def on_startup():
    global WORKER_STARTED


    wait_for_db()  # DB가 열릴 때까지 대기 . 모델 테이블 생성(이미 있으면 no-op)
    Base.metadata.create_all(bind=engine)
    logger.info("[startup] DB 테이블 생성(필요 시) 완료")

    # --- in-process guard: start worker only once per process ---
    with WORKER_LOCK:
        if WORKER_STARTED:
            logger.info("[startup] 워커 이미 기동되어 skip")
            return
        
        # 워커 기동 - Redis+DB를 감시하며 분석 진행
        t = threading.Thread(target=run_forever, daemon=True, name="worker-thread")
        t.start()
        WORKER_STARTED = True
        logger.info("[startup] 워커 스레드 시작")



@app.get("/health")
def health():
    return {"ok": True}










# ─────────────────────────────────────────────────────────────────────────────
# 디버그용 원본 이미지 검사 엔드포인트
@app.get("/inspect/original/{user_id}/{image_id}")
def inspect_original(user_id: int, image_id: int, session_id: str = Depends(get_session_id)):
    """
    주어진 user_id와 image_id로 Redis에서 이미지를 가져와 반환하는 API
    세션이 유효한지 확인 후 이미지 처리
    """
    # 세션 확인
    get_current_user(session_id, user_id)

    key = _orig_key(user_id, image_id)  # Redis 키 생성
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

# 수동 질의 응답용
class AnswerResponse(BaseModel):
    answer: str

@app.post("/answer-question", response_model=AnswerResponse)
async def answer_question(current_context: str,
                          recent_context: str,
                          similar_context: str,
                          user_question: str):
    try:
        answer = service.answer_question(
            current_context, recent_context, similar_context, user_question
        )
        return AnswerResponse(answer=answer)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
