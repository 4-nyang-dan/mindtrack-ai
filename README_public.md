# 🧠 MindTrack AI

AI 기반 이미지 분석 및 행동 예측 시스템입니다.<br>
이미지 업로드 후 대표 이미지를 선택하고, OCR과 PII 마스킹, 이미지 설명 생성, 벡터 임베딩, 과거 데이터 기반 행동 예측과 질문 응답 기능을 제공합니다.

## 🚀 주요 기능

- **📷 이미지 업로드 및 대표 이미지 선택**  
  업로드된 여러 이미지 중 가장 대표적인 이미지를 자동 선정합니다.

- **🔍 OCR + PII(개인정보) 탐지 및 마스킹**  
  Tesseract OCR과 Microsoft Presidio를 활용하여 개인정보를 탐지하고 블러 처리합니다.

- **📝 이미지 설명 및 임베딩 생성**  
  OpenAI GPT 모델로 이미지 설명을 생성하고, 해당 설명을 벡터 임베딩으로 변환합니다.

- **📂 벡터 데이터베이스 관리**  
  FAISS 기반으로 이미지 임베딩과 메타데이터를 저장/검색합니다.

- **🤖 행동 및 질문 예측**  
  현재/최근/유사 작업 컨텍스트를 기반으로 다음에 할 행동과 예상 질문을 예측합니다.

- **💬 과거 작업 기반 QA**  
  현재 상황과 과거 데이터를 바탕으로 사용자의 질문에 답변합니다.

## 📂 프로젝트 구조
```
mindtrack-ai/
│
├── app/
│   ├── app.py                 # FastAPI 엔드포인트
│   ├── config_loader.py       # YAML 설정 로더
│   ├── uploads/               # 업로드된 이미지
│
├── integration_service.py     # 통합 파이프라인
├── config.yaml                # 환경설정 파일
├── modules/
│   ├── image_selector/        # 대표 이미지 선택
│   ├── ocr_pii/               # OCR + 민감정보 마스킹
│   ├── image_description/     # 이미지 설명/임베딩
│   ├── action_predictor/      # 행동/질문 예측
│   ├── history_qa/            # 과거 QA
│
└── README.md
```

## ⚙️ 설정 (config.yaml 예시)
```yaml
app_host: "0.0.0.0"
app_port: 8000

openai:
  image_description_model: "gpt-4.1-mini"
  embedding_model: "text-embedding-3-small"
  action_predictor_model: "gpt-4.1-mini"
  history_qa_model: "gpt-5-mini"

vectordb:
  path: "./vectorstore/description_index.faiss"
  dim: 1536
  recent_k: 3
  search_top_k: 2

image_selector:
  n_clusters: null
  random_state: 42

integration:
  sample_dir: "./app/sample/uploads"
```

## ▶️ 실행 방법

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 환경 설정
`.env` 파일에 OpenAI API 키 및 Tesseract 경로를 설정합니다.
```
OPENAI_API_KEY=your_openai_api_key
TESSERACT_PATH=/usr/bin/tesseract
```

### 3. 서버 실행
```bash
uvicorn app.app:app --reload --host 0.0.0.0 --port 8000
```

## 📡 API 엔드포인트

### **1. /upload-and-process**
- 설명: 여러 이미지를 업로드하고, 대표 이미지 선택 → OCR/PII → 설명 → 임베딩 → 행동/질문 예측까지 수행
- 요청: `multipart/form-data` (파일 여러 개)
- 응답:
```json
{
  "representative_image": "uploads/sample.png",
  "description": "{...}",
  "predicted_actions": ["..."],
  "predicted_questions": ["..."]
}
```

### **2. /answer-question**
- 설명: 현재/최근/유사 컨텍스트와 질문을 기반으로 답변 생성
- 요청:
```json
{
  "current_context": "...",
  "recent_context": "...",
  "similar_context": "...",
  "user_question": "..."
}
```
- 응답:
```json
{
  "answer": "..."
}
```

## 🧪 샘플 데이터
`app/sample/` 경로에 테스트용 이미지, 설명, 임베딩 파일이 포함되어 있습니다.

## 📜 라이선스
MIT License
