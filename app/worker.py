"""
fastapi_worker_clean.py
- Redis에 쌓인 이미지를 30초 단위로 모아 분석(batch window)
- Redis 구조: pending:{uid}, user:{uid}:img:{img_id}
- 처리: pending → 폴더 저장 → AI 분석 → Spring 콜백 → 정리
- 상태키/processing 제거, 구조 단순화
"""
import re
import os
import time
import shutil
import tempfile
import requests
import redis
import threading
from typing import List
from app.integration_service import IntegrationService
from app.logging.logger import get_logger

log = get_logger("mindtrack.worker")

# ===== 환경 설정 =====
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
SPRING_BASE = os.getenv("SPRING_CALLBACK_BASE", "http://spring-backend:8080")
WINDOW_SEC = int(os.getenv("WINDOW_SEC", "15"))  # 윈도우 기간 (초)

#  이미지 바이너리를 안전하게 다루기 위해 decode_responses=False 유지
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
_service = IntegrationService()

# 현재 처리 중인 유저 (중복 방지용)
active_users = set()
lock = threading.Lock()

# ------------------------------------------------------------------
# Redis 키 유틸
# ------------------------------------------------------------------
def k_img(uid: int, img_id: int) -> bytes:
    """Redis 이미지 키 (bytes로 반환)"""
    return f"user:{uid}:img:{img_id}".encode("utf-8")


def k_pending(uid: int) -> bytes:
    """Redis 큐 키 (bytes로 반환)"""
    return f"pending:{uid}".encode("utf-8")


# ------------------------------------------------------------------
# 유저별 윈도우(batch window) 처리
# ------------------------------------------------------------------
def process_user_window(user_id: int):
    """한 유저의 30초 윈도우(batch window) 동안 이미지 수집 후 분석"""
    start_time = time.time()
    collected_ids: List[int] = []
    tmpdir = tempfile.mkdtemp(prefix=f"user{user_id}_")

    log.info(f"[WORKER] user={user_id} 윈도우 시작 ({WINDOW_SEC}초 동안 수집 중...)")

    try:
        t_collect_start = time.time()
        # 1. {WINDOW_SEC}초 동안 이미지 수집
        while time.time() - start_time < WINDOW_SEC:
            img_id = r.lpop(k_pending(user_id))
            if not img_id:
                time.sleep(0.2)
                continue

            try:
                img_int = int(img_id.decode("utf-8") if isinstance(img_id, bytes) else img_id)
            except ValueError:
                continue

            raw = r.get(k_img(user_id, img_int))
            if not raw:
                continue

            img_path = os.path.join(tmpdir, f"{img_int}.png")
            with open(img_path, "wb") as f:
                f.write(raw)
            collected_ids.append(img_int)
        t_collect_end = time.time()
        if not collected_ids:
            log.info(f"[WORKER] user={user_id} 수집된 이미지 없음. 종료.")
            return

        log.info(f"[WORKER] user={user_id} 총 {len(collected_ids)}장 수집 완료. 분석 시작.")
        log.info(f"[PERF] 수집 시간: {t_collect_end - t_collect_start:.2f}s")
        
        # 2.  AI 분석
        t_ai_start = time.time()
        result = {}
        try:
            result = _service.run_image_cycle(tmpdir) or {}
        except Exception as e:
            log.exception(f"[WORKER] AI 분석 중 오류 user={user_id}: {e}")
        t_ai_end = time.time()
        
        log.info(f"[PERF] AI 분석 소요시간: {t_ai_end - t_ai_start:.2f}s")
        log.info(f"[WORKER] 분석 결과: {result}")

        # 2.2 대표 이미지 ID 추출 및 결과 추출
        rep_img_path = result.get("representative_image", "")
        # 기존 폴더 경로로 반환하던걸 파일명으로 변경 
        rep_img_name = os.path.basename(rep_img_path)  # ex) "000172_blurred.png"

        match = re.search(r"(\d+)", rep_img_name)
        representative_id = int(match.group(1)) if match else (collected_ids[0] if collected_ids else -1)


        desc = result.get("description", "")
        actions = result.get("predicted_actions", [])
        questions = result.get("predicted_questions", [])



        # 3. Spring 콜백
        payload = {
            "userId": str(user_id),
            "imageIds": representative_id, # 이미지 리스트 대신 대표 이미지 id 만 전송
            "suggestion": {
                "representative_image": rep_img_name,
                "description": desc,
                "predicted_actions": actions,
            },
            "predicted_questions": [{"text": q} for q in questions[:3]],
        }
        t_callback_start = time.time()
        try:
            requests.post(f"{SPRING_BASE}/analysis/result", json=payload, timeout=10)
            log.info(f"[WORKER] ✅ Spring 콜백 성공 user={user_id} (이미지 {len(collected_ids)}장)")
        except Exception as e:
            log.exception(f"[WORKER] 에러!- Spring 콜백 실패 user={user_id}: {e}")
        t_callback_end = time.time()
        log.info(f"[PERF] Spring 콜백 소요시간: {t_callback_end - t_callback_start:.2f}s")

    finally:
        # 4. 정리
        total_time = time.time() - start_time
        log.info(f"[PERF] 전체 윈도우 처리 총 시간: {total_time:.2f}s")
        shutil.rmtree(tmpdir, ignore_errors=True)
        for img_id in collected_ids:
            r.delete(k_img(user_id, img_id))
        
        with lock:
            active_users.discard(user_id)
        
        log.info(f"[WORKER] DONE! user={user_id} 윈도우 종료 (처리된 {len(collected_ids)}장)")


# ------------------------------------------------------------------
# 메인 루프 (모든 유저 감시)
# ------------------------------------------------------------------
def run_forever():
    """Redis에서 모든 pending:* 큐를 감시하면서 각 유저별로 30초 윈도우 스레드 실행"""
    log.info(f"[WORKER] start clean mode (window={WINDOW_SEC}s)")
    while True:
        try:
            for key in r.scan_iter(b"pending:*"):  # bytes 단위 key
                key_str = key.decode("utf-8")
                user_id = int(key_str.split(":")[1])
                
                with lock:

                    # 이미 윈도우 실행 중인 유저는 스킵
                    if user_id in active_users:
                        continue

                    # 큐에 데이터가 있으면 새 윈도우 시작
                    if r.llen(k_pending(user_id)) > 0:
                        t = threading.Thread(target=process_user_window, args=(user_id,), daemon=True)
                        t.start()
                        active_users.add(user_id)
                        log.info(f"[WORKER] ▶️ user={user_id} 윈도우 스레드 시작 (active={len(active_users)})")

            time.sleep(1)

        except Exception as e:
            log.exception(f"[WORKER] 루프 오류: {e}")
            time.sleep(2)


#---main.py에서만 워커 시작하도록 유지 , 아래 코드는 주석 처리함
#if __name__ == "__main__":
#    run_forever()
