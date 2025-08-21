import io, os, time, logging, redis, tempfile, shutil, json
from PIL import Image
from sqlalchemy import text
from app.db import SessionLocal
import logging
import sys
from app.logging.logger import get_logger

# sys.path 보정
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.integration_service import IntegrationService  # 모델/파이프라인 서비스

# 워커 로거
log = get_logger("mindtrack.worker")

r = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    decode_responses=False,  # bytes 유지
)

#  무거운 초기화 1회만 (모델/벡터DB 로딩 비용 절감)
_service = IntegrationService()

def _orig_key(uid: int, img_id: int) -> str:
    return f"user:{uid}:img:{img_id}"

def _ensure_tmp_dir(prefix="mt_job_"):
    return tempfile.mkdtemp(prefix=prefix)

def _write_bytes_to_file(raw: bytes, path: str):
    with open(path, "wb") as f:
        f.write(raw)

def analyze_image_with_pipeline(raw: bytes) -> dict:
    """
    - 임시 폴더를 만들고
    - raw 바이트를 PNG 파일로 저장
    - IntegrationService.run_image_cycle(임시폴더)를 호출
    - 결과(dict)를 그대로 반환
    """
    tmpdir = _ensure_tmp_dir()
    img_path = os.path.join(tmpdir, "input.png")
    try:
        _write_bytes_to_file(raw, img_path)
        # 폴더 단위로 대표 이미지 뽑고 → OCR/PII → 설명 → 임베딩/검색 → 행동/질문 예측까지 수행
        result = _service.run_image_cycle(tmpdir)
        return result  # { representative_image, description, predicted_actions, predicted_questions }
    finally:
        # 임시 파일/폴더 정리
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

def claim_batch(session, batch_size: int = 16):
    # 🔁 ScreenshotImage 테이블에서 PENDING 선점 → IN_PROGRESS 전환
    sql = text("""
    WITH cte AS (
      SELECT id, user_id
      FROM screenshot_image
      WHERE analysis_status = 'PENDING'
      ORDER BY captured_at
      FOR UPDATE SKIP LOCKED
      LIMIT :batch_size
    )
    UPDATE screenshot_image si
    SET analysis_status = 'IN_PROGRESS'
    FROM cte
    WHERE si.id = cte.id
    RETURNING si.id, si.user_id
    """)
    return session.execute(sql, {"batch_size": batch_size}).mappings().all()

def mark_done(session, rec_id: int, result_text: str):
    session.execute(
        text("""
        UPDATE screenshot_image
        SET analysis_status='DONE', analysis_result=:r
        WHERE id=:id
        """),
        {"r": result_text, "id": rec_id}
    )

def mark_failed(session, rec_id: int, error: str):
    session.execute(
        text("""
        UPDATE screenshot_image
        SET analysis_status='FAILED', analysis_result=:e
        WHERE id=:id
        """),
        {"e": error[:2000], "id": rec_id}
    )

# --------------------------- 핵심 수정: insert/upsert 시그니처 정리 ---------------------------

def insert_suggestions(
    session,
    user_id: int | str,
    image_id: int,
    questions: list[str],
    description: str,  # ✅ 답변 생성 컨텍스트로 사용
) -> int | None:
    """
    suggestions 헤더 1건 + suggestion_items Top3 insert.
    질문 + 자동 답변까지 저장.
    """
    res = session.execute(
        text("""
            INSERT INTO suggestions(user_id, image_id)
            VALUES (:u, :img)
            RETURNING id
        """),
        {"u": str(user_id), "img": image_id}
    )
    suggestion_id = res.scalar()

    if not suggestion_id:
        return None

    if not questions:
        return suggestion_id

    top3 = questions[:3]
    params = []
    for i, q in enumerate(top3, start=1):
        # 🔽 자동 답변 생성 (에러가 나도 다른 항목은 계속 진행)
        try:
            ans = _service.answer_question(
                current_context=description or "",
                recent_context="",
                similar_context="",
                user_question=q
            )
        except Exception as e:
            log.exception("[worker] answer_question 실패(user=%s, image=%s, rank=%s): %s",
                          user_id, image_id, i, e)
            ans = ""

        params.append({
            "sid": suggestion_id,
            "q": q,
            "a": ans if isinstance(ans, str) else str(ans),
            "c": None,   # confidence 없으면 NULL, 필요시 0.0 등으로 변경
            "r": i
        })

    # executemany
    session.execute(
        text("""
            INSERT INTO suggestion_items(suggestion_id, question, answer, confidence, rank)
            VALUES (:sid, :q, :a, :c, :r)
        """),
        params
    )
    return suggestion_id


def upsert_suggestions_for_image(
    session,
    user_id: int | str,
    image_id: int,
    questions: list[str],
    description: str,  # ✅ 반드시 전달받아 insert로 넘김
) -> int | None:
    """
    유니크 인덱스(ux_suggestions_user_image)가 걸려있으면
    동일 (user_id, image_id) 기존 행 삭제 후 재삽입.
    """
    # 있으면 삭제 후 재삽입 (가장 단순)
    session.execute(
        text("DELETE FROM suggestions WHERE user_id=:u AND image_id=:img"),
        {"u": str(user_id), "img": image_id}
    )
    return insert_suggestions(session, user_id, image_id, questions, description)

# ---------------------------------------------------------------------------------------------

# 일단 워커 자체는 단일 프로세스로 설정
def run_forever():
    log.info("[worker] 시작 (DB 상태머신 + 기존 파이프라인 재사용)")
    while True:
        s = SessionLocal()
        try:
            s.begin()
            jobs = claim_batch(s, batch_size=8)  # 최대 8개의 작업만 가져와서 처리
            s.commit()
        except Exception as e:
            s.rollback()
            log.exception("[worker] claim_batch 실패: %s", e)
            s.close()
            time.sleep(1.0)
            continue

        if not jobs:
            s.close()
            time.sleep(0.5)
            continue

        for row in jobs:
            rec_id  = int(row["id"])
            user_id = int(row["user_id"])
            key = _orig_key(user_id, rec_id)

            log.info("[worker] job 시작 user=%s image=%s key=%s", user_id, rec_id, key)

            raw = r.get(key)  # bytes
            if not raw:
                log.warning("[worker] 원본 없음(TTL 만료?) key=%s", key)
                s2 = SessionLocal(); s2.begin()
                try:
                    mark_failed(s2, rec_id, "original_expired_or_missing")
                    s2.commit()
                except Exception as e:
                    s2.rollback(); log.exception("[worker] FAILED 기록 실패: %s", e)
                finally:
                    s2.close()
                continue

            try:
                log.info("[worker] 모델 분석 호출 시작 image=%s", rec_id)
                # 1) 실제 분석
                result = analyze_image_with_pipeline(raw)
                log.info("[worker] 모델 분석 완료 image=%s result.keys=%s",
                         rec_id, list(result.keys()) if result else None)

                description: str = (result.get("description") or "").strip()
                predicted_questions: list[str] = result.get("predicted_questions") or []
                predicted_actions: list[str] = result.get("predicted_actions") or []

                payload = {
                    "description": description,
                    "predicted_actions": predicted_actions,
                    "predicted_questions": predicted_questions,
                }
                result_text = json.dumps(payload, ensure_ascii=False)

                # 2) 같은 트랜잭션에서 DONE 저장 + suggestions 적재
                s2 = SessionLocal(); s2.begin()
                try:
                    # 2-1) DONE + analysis_result 저장
                    mark_done(s2, rec_id, result_text)

                    # 2-2) Top3 질문 추출 후 suggestions/suggestion_items 적재
                    pq = predicted_questions
                    suggestion_id = upsert_suggestions_for_image(
                        s2, user_id, rec_id, pq, description  # ✅ description 전달
                    )

                    # 2-3) 커밋
                    s2.commit()
                    log.info("[worker] DONE + suggestions inserted user=%s image=%s sid=%s",
                             user_id, rec_id, suggestion_id)

                except Exception as e:
                    s2.rollback()
                    log.exception("[worker] persist error: %s", e)
                    # DONE 롤백됐을 수 있으니 FAILED로 남겨 트러블슈팅
                    try:
                        s2.begin(); mark_failed(s2, rec_id, f"persist_error: {e}"); s2.commit()
                    except Exception as e2:
                        s2.rollback()
                        log.exception("[worker] FAILED 기록 실패(rollback): %s", e2)
                finally:
                    s2.close()

            except Exception as e:
                log.exception("[worker] 분석 실패: %s", e)
                s2 = SessionLocal(); s2.begin()
                try:
                    mark_failed(s2, rec_id, str(e))
                    s2.commit()
                except Exception as e2:
                    s2.rollback(); log.exception("[worker] FAILED 기록 실패(2): %s", e2)
                finally:
                    s2.close()
