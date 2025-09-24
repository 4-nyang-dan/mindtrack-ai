# app/worker.py
# 배치 워커: 30초 윈도우로 선점 → 임시폴더에 저장 → 폴더 단일 호출로 분석 → 결과/상태 저장

import os, time, logging, redis, tempfile, shutil, json, sys
from sqlalchemy import text
from app.db import SessionLocal
from app.logging.logger import get_logger

# 패키지 경로 보정 (컨테이너 내 상대경로용)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 폴더 단위 파이프라인
from app.integration_service import IntegrationService
# 키 규칙은 공용 모듈의 것을 사용 (중복 정의 금지)
from app.common import _orig_key

# 로거/Redis/서비스는 1회 초기화
log = get_logger("mindtrack.worker")


r = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    decode_responses=False,  # bytes 유지
)
_service = IntegrationService()

# ---------- 유틸 ----------
def _ensure_tmp_dir(prefix="mt_batch_"):
    """배치 분석용 임시 디렉토리 생성"""
    return tempfile.mkdtemp(prefix=prefix)

def _write_bytes(path: str, raw: bytes):
    """원본 바이트를 파일로 저장 (경로 생성 포함)"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(raw)



def claim_window(session, window_size: int, max_images: int):
    """
    가장 오래된 PENDING 1장을 '피벗'으로 삼아,
    동일 사용자 + [pivot, pivot+window] 구간의 PENDING을 선점(IN_PROGRESS)하고 반환.
    다중 워커 환경을 위해 일단 SKIP LOCKED로 중복 선점 방지.
    """
    sql = text("""
        WITH first_pending AS (
            SELECT id, user_id, captured_at
              FROM screenshot_image
             WHERE analysis_status = 'PENDING'
             ORDER BY captured_at
             LIMIT 1
             FOR UPDATE SKIP LOCKED
        ),
        picked AS (
            SELECT si.id, si.user_id, si.captured_at
              FROM screenshot_image si
              JOIN first_pending fp ON si.user_id = fp.user_id
             WHERE si.analysis_status = 'PENDING'
               AND si.captured_at >= fp.captured_at
               AND si.captured_at <  fp.captured_at + (:window_size || ' seconds')::interval
             ORDER BY si.captured_at
             LIMIT :max_images
             FOR UPDATE SKIP LOCKED
        )
        UPDATE screenshot_image si
           SET analysis_status = 'IN_PROGRESS',
               updated_at = NOW()
          FROM picked
         WHERE si.id = picked.id
     RETURNING si.id, si.user_id, si.captured_at
    """)
    return session.execute(
        sql, {"window_size": window_size, "max_images": max_images}
    ).mappings().all()




# ---------- 메인 루프 ----------
def run_forever():

    log.info(f"[worker] module file = {__file__}")

    """
    배치 워커 메인 루프.
    - 30초 윈도우로 이미지들을 선점(IN_PROGRESS)
    - Redis에서 원본을 모아 임시폴더에 저장
    - 폴더 단일 호출(IntegrationService.run_image_cycle)로 분석
    - 결과는 피l벗 이미지에 suggestions 1건 + suggestion_items(질문/답변) 저장
    - 성공한 이미지 전부 DONE, 실패건 FAILED(reason) 기록
    """
    log.info("[worker] 배치 워커 시작: 30s 윈도우 + 폴더 단일 호출")

    WINDOW_SEC = int(os.getenv("WORKER_WINDOW_SEC", "30"))
    MAX_IMAGES = int(os.getenv("WORKER_MAX_IMAGES", "50"))

    while True:
        s = SessionLocal()
        try:
            s.begin()
            jobs = claim_window(s, window_size=WINDOW_SEC, max_images=MAX_IMAGES)
            s.commit()
        except Exception as e:
            s.rollback(); s.close()
            log.exception("[worker] claim_window 실패: %s", e)
            time.sleep(1.0)
            continue

        if not jobs:
            s.close()
            time.sleep(0.5)
            continue

        user_id = int(jobs[0]["user_id"])
        ids_in_window = [int(r["id"]) for r in jobs]
        pivot_image_id = ids_in_window[0]  # 대표 suggestion 연결 대상

        tmpdir = _ensure_tmp_dir()
        saved_ids = []

        try:
            # 1) Redis에서 원본 로드 → 임시폴더 저장
            for img_id in ids_in_window:
                key = _orig_key(user_id, img_id)
                raw = r.get(key)
                if not raw:
                    log.warning("[worker] 원본 없음(TTL 만료?) key=%s", key)
                    s2 = SessionLocal(); s2.begin()
                    try:
                        mark_failed(s2, img_id, "original_expired_or_missing")
                        s2.commit()
                    except Exception as e2:
                        s2.rollback(); log.exception("[worker] FAILED 기록 실패: %s", e2)
                    finally:
                        s2.close()
                    continue

                fpath = os.path.join(tmpdir, f"{img_id:06d}.png")
                try:
                    _write_bytes(fpath, raw)
                    saved_ids.append(img_id)
                except Exception as e:
                    log.exception("[worker] 파일 저장 실패 image=%s err=%s", img_id, e)
                    s2 = SessionLocal(); s2.begin()
                    try:
                        mark_failed(s2, img_id, f"file_write_error: {e}")
                        s2.commit()
                    except Exception as e2:
                        s2.rollback(); log.exception("[worker] FAILED 기록 실패(2): %s", e2)
                    finally:
                        s2.close()

            if not saved_ids:
                shutil.rmtree(tmpdir, ignore_errors=True)
                continue

            # 2) 폴더 단일 호출로 분석
            log.info("[worker] 모델 분석 시작(배치) user=%s count=%s dir=%s", user_id, len(saved_ids), tmpdir)
            try:
                result = _service.run_image_cycle(tmpdir)
            except Exception as e:
                log.exception("[worker] 모델 분석 실패(배치): %s", e)
                s3 = SessionLocal(); s3.begin()
                try:
                    for img_id in saved_ids:
                        mark_failed(s3, img_id, f"batch_analyze_error: {e}")
                    s3.commit()
                except Exception as e2:
                    s3.rollback(); log.exception("[worker] FAILED 기록 실패(배치): %s", e2)
                finally:
                    s3.close()
                shutil.rmtree(tmpdir, ignore_errors=True)
                continue

            # 3) 결과 저장
            analysis_result = {
                "description": result.get("description", ""),
                "predicted_actions": result.get("predicted_actions", []),
                "predicted_questions": result.get("predicted_questions", []),
            }

            s4 = SessionLocal(); s4.begin()
            try:
                suggestion_id = insert_suggestions(s4, user_id, pivot_image_id)
                insert_suggestion_items(
                    s4,
                    suggestion_id,
                    analysis_result.get("predicted_questions") or [],
                    analysis_result["description"],
                )
                # 원본 읽기/저장에 성공했던 것들은 DONE
                mark_done_many(s4, saved_ids)
                s4.commit()
                log.info("[worker] DONE(batch) user=%s pivot=%s saved=%s sid=%s",
                         user_id, pivot_image_id, len(saved_ids), suggestion_id)
            except Exception as e:
                s4.rollback()
                log.exception("[worker] persist error(batch): %s", e)
                s4.begin()
                try:
                    for img_id in saved_ids:
                        mark_failed(s4, img_id, f"persist_error: {e}")
                    s4.commit()
                except Exception as e2:
                    s4.rollback(); log.exception("[worker] FAILED 기록 실패(persist): %s", e2)
                finally:
                    s4.close()
                shutil.rmtree(tmpdir, ignore_errors=True)
                continue

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)









def mark_failed(session, image_id: int, reason: str):
    """해당 이미지 처리 실패로 마킹 (트랜잭션 내 호출)"""
    session.execute(
        text("""
            UPDATE screenshot_image
               SET analysis_status='FAILED',
                   fail_reason=:reason,
                   updated_at=NOW()
             WHERE id=:id
        """),
        {"id": int(image_id), "reason": str(reason)[:500]},
    )


def mark_done_many(session, image_ids: list[int]):
    """여러 이미지를 성공(DONE) 마킹 (트랜잭션 내 호출)"""
    if not image_ids:
        return
    session.execute(
        text("""
            UPDATE screenshot_image
               SET analysis_status='DONE',
                   updated_at=NOW()
             WHERE id = ANY(:ids)
        """),
        {"ids": [int(x) for x in image_ids]},
    )

def insert_suggestions(session, user_id: int, image_id: int) -> int:
    """
    대표 제안을 suggestions에 1건 저장 (피벗 이미지에 연결)
    """
    res = session.execute(
        text("""
            INSERT INTO suggestions(user_id, image_id, created_at)
            VALUES (:u, :img,  NOW())
            RETURNING id
        """),
        {"u": str(user_id), "img": int(image_id)},
    )
    return res.scalar()




def insert_suggestion_items(session, suggestion_id: int, questions: list[str], description: str):
    """
    예측된 질문 상위 3개에 대해 answer 생성 후 suggestion_items에 저장
    """
    if not questions:
        return
    params = []
    for i, q in enumerate(questions[:3], start=1):
        ans = _service.answer_question(
            current_context=description,
            recent_context="",
            similar_context="",
            user_question=q,
        )
        params.append({
            "sid": suggestion_id,
            "q": q,
            "a": ans if isinstance(ans, str) else str(ans),
            "c": None,
            "r": i,
        })
    session.execute(
        text("""
            INSERT INTO suggestion_items(suggestion_id, question, answer, confidence, rank)
            VALUES (:sid, :q, :a, :c, :r)
        """),
        params,
    )

