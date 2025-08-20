# app/models.py
from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from app.db import Base

# 분석 결과를 저장할 테이블 모델
class ScreenshotAnalysis(Base):
    __tablename__ = "screenshot_analysis"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    image_id = Column(Integer, index=True)
    status = Column(String, default="PENDING")
    result = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
