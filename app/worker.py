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
from queue import Queue # 병렬 분석 파이프라인용 큐 추가

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

# 분석을 비동기적으로 수행하기 위한 큐 생성
analysis_queue = Queue()

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
    """
    한 유저의 15초 윈도우(batch window) 동안 Redis 큐에서 이미지를 수집하고
    새 tmpdir에 저장한 뒤, 분석 큐(analysis_queue)에 전달한다.
    이후 즉시 다음 윈도우 수집으로 넘어가며, 분석은 별도 스레드에서 수행된다.
    """

    while True:
            start_time = time.time()
            collected_ids: List[int] = []
            tmpdir = tempfile.mkdtemp(prefix=f"user{user_id}_")

            #log.info(f"[WORKER] user={user_id} 윈도우 시작 ({WINDOW_SEC}초 동안 수집 중...)")
            log.info(f"[COLLECT] user={user_id} 수집 스레드 시작 (window={WINDOW_SEC}s)")

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
            
            # 2. 수집된 이미지가 없으면 tmpdir 삭제 후 다음 루프로 이동
            if not collected_ids:
                log.info(f"[COLLECT] user={user_id} 수집된 이미지 없음. 다음 윈도우 대기.")
                shutil.rmtree(tmpdir, ignore_errors=True)
                continue
            
            # 3. 수집 완료 → 분석 큐에 전
            log.info(f"[WORKER] user={user_id} 총 {len(collected_ids)}장 수집 완료. 분석 큐에 전달.")
            analysis_queue.put((user_id, tmpdir, collected_ids))

            # 4. 즉시 다음 윈도우 수집 시작 (분석은 별도 스레드에서 수행)
            log.info(f"[COLLECT] user={user_id} 다음 윈도우로 이동.")


def analyze_worker():
    """
    process_user_window()가 analysis_queue에 넣은 작업을 소비한다.
    각 폴더(tmpdir)에 대해 AI 분석을 수행하고, Spring 콜백 전송 및 데이터 정리를 담당한다.
    """
    while True:
        try:
            user_id, tmpdir, collected_ids = analysis_queue.get()
            log.info(f"[ANALYZE] user={user_id} 폴더={tmpdir} 분석 시작 ({len(collected_ids)}장)")
            # 1. AI 분석 실행
            t_ai_start = time.time()
            result = {}
            try:
                result = _service.run_image_cycle(tmpdir) or {}
            except Exception as e:
                log.exception(f"[ANALYZE] AI 분석 오류 user={user_id}: {e}")
            t_ai_end = time.time()

            log.info(f"[PERF] AI 분석 소요시간: {t_ai_end - t_ai_start:.2f}s")
            log.info(f"[ANALYZE] 분석 결과: {result}")

            # 2. 대표 이미지 및 결과 추출
            rep_img_path = result.get("representative_image", "")
            rep_img_name = os.path.basename(rep_img_path)
            match = re.search(r"(\d+)", rep_img_name)
            representative_id = int(match.group(1)) if match else (collected_ids[0] if collected_ids else -1)

            desc = result.get("description", "")
            actions = result.get("predicted_actions", [])
            questions = result.get("predicted_questions", [])

            payload = {
                "user_id": user_id,
                "image_id": representative_id,
                "suggestion": {
                    "representative_image": rep_img_name,
                    "description": desc,
                    "predicted_actions": actions,
                },
                "predicted_questions": [{"question": q} for q in questions[:3]],
            }

            t_callback_start = time.time()
            try:
                requests.post(f"{SPRING_BASE}/analysis/result", json=payload, timeout=10)
                log.info(f"[CALLBACK TEST] posting to {SPRING_BASE}/analysis/result")
                log.info(f"[CALLBACK PAYLOAD] {payload}")
                log.info(f"[ANALYZE] ✅ Spring 콜백 성공 user={user_id}")
            except Exception as e:
                log.exception(f"[ANALYZE] Spring 콜백 실패 user={user_id}: {e}")
            t_callback_end = time.time()

            log.info(f"[PERF] Spring 콜백 소요시간: {t_callback_end - t_callback_start:.2f}s")

        finally:
            # 정리: temp 폴더 및 redis 데이터 정리
            for img_id in collected_ids:
                r.delete(k_img(user_id, img_id))  #분석 끝난 원본 이미지를 정리해줘야 메모리 누수 방지
            shutil.rmtree(tmpdir, ignore_errors=True)
            analysis_queue.task_done()
            log.info(f"[ANALYZE] user={user_id} 분석 완료 및 정리 완료 ({len(collected_ids)}장)")

# ------------------------------------------------------------------
# 메인 루프 (모든 유저 감시)
# ------------------------------------------------------------------
def run_forever():
    """
    Redis의 pending:* 큐를 주기적으로 감시하여,
    유저별로 수집 스레드를 실행한다.
    동시에 하나의 분석 스레드(analyze_worker)가 큐를 소비하며 병렬로 동작한다.
    """
    
    log.info(f"[WORKER] start pipeline mode (window={WINDOW_SEC}s)")
    
    # 분석 전용 스레드 시작 (큐 소비자)
    threading.Thread(target=analyze_worker, daemon=True).start()

    while True:
        try:
            for key in r.scan_iter(b"pending:*"):  # bytes 단위 key -> Redis 내 모든 유저 큐 탐색
                key_str = key.decode("utf-8")
                user_id = int(key_str.split(":")[1])
                
                with lock:

                    # 이미 윈도우 실행 중인 유저(수집 스레드가 실행 중)는 스킵
                    if user_id in active_users:
                        continue

                    # 큐에 데이터가 있으면 새 윈도우 시작
                    if r.llen(k_pending(user_id)) > 0:
                        t = threading.Thread(target=process_user_window, args=(user_id,), daemon=True)
                        t.start()
                        active_users.add(user_id)
                        log.info(f"[WORKER] ▶ user={user_id} 스레드 시작 (active={len(active_users)})")

            time.sleep(1)

        except Exception as e:
            log.exception(f"[WORKER] 루프 오류: {e}")
            time.sleep(2)


#---main.py에서만 워커 시작하도록 유지 , 아래 코드는 주석 처리함
#if __name__ == "__main__":
#    run_forever()
