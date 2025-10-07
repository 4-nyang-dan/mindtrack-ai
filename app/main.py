import os, sys, logging, io, threading, redis
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from PIL import Image
from app.logging.logger import get_logger  # 이거만 import, run_forever는 나중에 lazy import
from app.integration_service import IntegrationService
from fastapi.middleware.cors import CORSMiddleware

logger = get_logger("mindtrack.fastapi")

startup_lock = threading.Lock()
startup_done = False

service = IntegrationService()

# ====== FastAPI 앱 ======
app = FastAPI(title="mind-track AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 또는 ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Redis 클라이언트 (bytes 유지) ======
r = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    decode_responses=False,
)


# ====== Startup 이벤트 ======
@app.on_event("startup")
def on_startup():
    global startup_done
    with startup_lock:
        if startup_done:
            logger.warning("[startup] 이미 워커가 실행 중이므로 재기동 생략")
            return
        startup_done = True
        
         # 워커 스레드 시작
        from app.worker import run_forever
        t = threading.Thread(target=run_forever, daemon=True, name="worker-thread")
        t.start()
        logger.info("[startup] 워커 스레드 시작")


# ====== 헬스체크 ======
@app.get("/health")
def health():
    return {"ok": True}

# ====== (옵션) 원본 이미지 점검 API ======
"""
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
"""

@app.post("/api/qa/answer")
def answer_question(data: dict = Body(...)):
    """
    프론트에서 클릭된 질문을 받아 AI 응답을 반환
    """
    question = data.get("question")
    if not question:
        return {"answer": "질문이 비어 있습니다."}

    print(f"[QA 요청 수신] question={question}")
    answer = service.answer_question(question)
    return {"answer": answer}
