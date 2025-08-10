# app/app.py
import sys
import os
import shutil
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # 프로젝트 루트 경로 추가

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config_loader import config
from integration_service import IntegrationService


# ==========================
# 경로 및 서비스 초기화
# ==========================
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="mind-track API",
    description="이미지 기반 작업 분석 및 행동 예측 API",
    version="1.0.0"
)

service = IntegrationService()


# ==========================
# 응답 모델 정의
# ==========================
class UploadProcessResponse(BaseModel):
    representative_image: str
    description: str
    predicted_actions: List[str]
    predicted_questions: List[str]


class AnswerResponse(BaseModel):
    answer: str


# ==========================
# 엔드포인트
# ==========================
@app.post(
    "/upload-and-process",
    response_model=UploadProcessResponse,
    summary="이미지 업로드 및 처리",
    description="""
    여러 장의 이미지를 업로드하면:
    1. 대표 이미지 선정
    2. OCR 및 PII 처리
    3. 이미지 설명 생성
    4. 벡터 임베딩 생성 및 저장
    5. 최근/유사 작업 컨텍스트 분석
    6. 다음 행동 및 예상 질문 예측
    """
)
async def upload_and_process(files: List[UploadFile] = File(...)):
    try:
        # 업로드된 파일 저장
        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        # 저장된 모든 이미지로 프로세스 실행
        result = service.run_image_cycle(UPLOAD_DIR)
        return UploadProcessResponse(**result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post(
    "/answer-question",
    response_model=AnswerResponse,
    summary="질문에 대한 답변 생성",
    description="""
    현재 작업, 최근 작업, 유사 작업, 그리고 사용자의 질문을 기반으로
    심층 QA 응답을 생성합니다.
    """
)
async def answer_question(
    current_context: str,
    recent_context: str,
    similar_context: str,
    user_question: str
):
    try:
        answer = service.answer_question(
            current_context, recent_context, similar_context, user_question
        )
        return AnswerResponse(answer=answer)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ==========================
# 개발 환경에서 실행
# ==========================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=config.get("app_host", "0.0.0.0"),
        port=config.get("app_port", 8000),
        reload=True
    )
