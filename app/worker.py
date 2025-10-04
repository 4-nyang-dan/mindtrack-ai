"""
fastapi_worker.py
- 30초 주기로 Redis pending 큐에서 이미지 가져와 분석
- pending → processing 원자적 이동
- 임시 폴더로 이미지 다운로드 → AI 분석 → Spring 콜백
- 처리 완료 후 Redis 정리
"""

import os
import time
import shutil
import tempfile
import requests
import redis
from typing import Optional

# 내부 서비스 (AI 분석 파이프라인)
from app.integration_service import IntegrationService
from app.logging.logger import get_logger

log = get_logger("mindtrack.worker")

# ====== 환경 변수 ======
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
SPRING_BASE = os.getenv("SPRING_CALLBACK_BASE", "http://localhost:8080")
POLL_INTERVAL_SEC = int(os.getenv("POLL_INTERVAL_SEC", "30"))

# ====== Redis 연결 ======
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ====== 서비스 초기화 ======
_service = IntegrationService()


# ------------------------------------------------------------------
# Redis 키 유틸 (Spring과 동일)
# ------------------------------------------------------------------
def k_img(uid: int, img_id: int) -> str:
    return f"user:{uid}:img:{img_id}"

def k_pending(uid: int) -> str:
    return f"pending:{uid}"

def k_processing(uid: int) -> str:
    return f"processing:{uid}"

def k_status(uid: int, img_id: int) -> str:
    return f"screenshot:status:{uid}:{img_id}"


# ------------------------------------------------------------------
# 임시 디렉토리 유틸
# ------------------------------------------------------------------
def _ensure_tmp_dir(prefix="mt_job_") -> str:
    return tempfile.mkdtemp(prefix=prefix)

def _write_bytes(path: str, raw: bytes) -> None:
    with open(path, "wb") as f:
        f.write(raw)


# ------------------------------------------------------------------
# 배치 실행
# ------------------------------------------------------------------
def process_one(user_id: int) -> bool:
    """
    - pending → processing 원자적 이동
    - 이미지 다운로드 → 분석 → Spring 콜백
    - 처리 완료 시 Redis 정리
    """
    # 1) pending → processing 이동
    image_id = r.rpoplpush(k_pending(user_id), k_processing(user_id))
    if image_id is None:
        return False
    image_id = int(image_id)

    log.info(f"[WORKER] user={user_id}, image={image_id} 처리 시작")
    r.set(k_status(user_id, image_id), "PROCESSING")

    # 2) 원본 이미지 로드
    raw = r.get(k_img(user_id, image_id))
    if not raw:
        log.warning(f"[WORKER] 원본 이미지 없음 user={user_id}, image={image_id}")
        r.lrem(k_processing(user_id), 0, image_id)
        r.delete(k_status(user_id, image_id))
        return True

    # 3) 임시 디렉토리에 저장
    tmpdir = _ensure_tmp_dir()
    try:
        img_path = os.path.join(tmpdir, f"{image_id}.png")
        _write_bytes(img_path, raw.encode() if isinstance(raw, str) else raw)

        # 4) AI 분석
        result = _service.run_image_cycle(tmpdir)
        description = (result.get("description") or "").strip()
        p_actions   = result.get("predicted_actions")   or []
        p_questions = result.get("predicted_questions") or []

        # 5) Spring 콜백
        payload = {
            "userId": str(user_id),
            "imageId": image_id,
            "suggestion": {
                "description": description,
                "predicted_actions": p_actions
            },
            "predicted_questions": [{"text": q} for q in p_questions[:3]]
        }
        try:
            resp = requests.post(f"{SPRING_BASE}/analysis/result", json=payload, timeout=12)
            resp.raise_for_status()
            log.info(f"[WORKER] Spring 콜백 성공 user={user_id}, image={image_id}")
        except Exception as e:
            log.exception(f"[WORKER] Spring 콜백 실패 user={user_id}, image={image_id}: {e}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

        # 6) 처리 완료 → Redis 정리
        r.lrem(k_processing(user_id), 0, image_id)
        r.delete(k_img(user_id, image_id))
        r.delete(k_status(user_id, image_id))

    return True


# ------------------------------------------------------------------
# 메인 루프
# ------------------------------------------------------------------
def run_forever(user_id: int) -> None:
    log.info("[worker] start (windowed 30s batch)")
    while True:
        window_start = time.time()
        processed_any = False
        processed_count = 0;
        # 30초 윈도우 동안 pending에서 가능한 만큼 꺼내서 처리
        while time.time() - window_start < POLL_INTERVAL_SEC:
            try:
                processed = process_one(user_id)
                if not processed:
                    # 큐 비어있으면 짧게 쉬었다가 윈도우 기간 내 재확인
                    time.sleep(0.2)
                else:
                    processed_any = True
                    processed_count +=1; # 윈도우 내 수집 이미지 개수 카운트
            except Exception as e:
                log.exception("process_one error: %s", e)
                time.sleep(1.0)
    
     # 윈도우 종료 — 처리 여부를 로깅
        if processed_any:
            log.info("[worker] window finished: processed %d items", processed_count)
        else:
            log.debug("[worker] window finished: no items processed")

if __name__ == "__main__":
    USER_ID = int(os.getenv("WORKER_USER_ID", "1"))  # 단일 사용자 기준
    run_forever(USER_ID)
