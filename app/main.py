from fastapi import FastAPI, UploadFile
from app.schemas import OCRResult, RepresentativeImageResult, DescriptionResult
from app.config import settings
from app.logging.logger import get_logger

logger = get_logger(__name__)
app = FastAPI(title="AI Agent API")

@app.on_event("startup")
async def startup_event():
    logger.info("Application starting...")

@app.get("/")
async def health_check():
    return {"status": "ok", "env": settings.APP_ENV}

@app.post("/process-image", response_model=OCRResult)
async def process_image(file: UploadFile):
    logger.info(f"Processing image: {file.filename}")
    return OCRResult(text="Sample text", pii_found=False)

@app.post("/select-representative", response_model=RepresentativeImageResult)
async def select_representative(files: list[UploadFile]):
    logger.info(f"Selecting representative image from {len(files)} files")
    return RepresentativeImageResult(filename="image_1.jpg")

@app.post("/generate-description", response_model=DescriptionResult)
async def generate_description(file: UploadFile):
    logger.info(f"Generating description for {file.filename}")
    return DescriptionResult(description="A sample description", embedding=[0.1, 0.2, 0.3])
