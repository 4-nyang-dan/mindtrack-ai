# app/app.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # 프로젝트 루트 경로 추가

from typing import List
from fastapi import FastAPI, UploadFile, File
import shutil
from fastapi.responses import JSONResponse

from config_loader import config
from integration_service import IntegrationService

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="mind-track API")
service = IntegrationService()

@app.post("/upload-and-process")
async def upload_and_process(files: List[UploadFile] = File(...)):
    """
    여러 이미지 업로드 후 run_image_cycle 실행
    """
    try:
        # 업로드된 파일 저장
        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        # 저장된 모든 이미지로 프로세스 실행
        result = service.run_image_cycle(UPLOAD_DIR)
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/answer-question")
async def answer_question(
    current_context: str,
    recent_context: str,
    similar_context: str,
    user_question: str
):
    """
    History QA 기능
    """
    try:
        answer = service.answer_question(
            current_context, recent_context, similar_context, user_question
        )
        return {"answer": answer}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=config.get("app_host", "0.0.0.0"),
        port=config.get("app_port", 8000),
        reload=True
    )
