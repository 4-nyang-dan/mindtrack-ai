FROM python:3.10-slim
WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    libgl1 \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# pip 패키지 먼저 설치 (캐시 유지)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# 앱 코드 복사
COPY . .

ENV PYTHONPATH=/app
EXPOSE 8000

# 개발용
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# 운영용 (원할 때만)
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]