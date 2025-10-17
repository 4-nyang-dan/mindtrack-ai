# mindtrack-ai (for Developer)

## 📌 프로젝트 개요
`mindtrack-ai`는 이미지 기반 데이터 처리, 개인정보(PII) 마스킹, 이미지 설명 생성, 벡터 임베딩 저장 및 검색, 그리고 AI 기반 행동 예측 및 질의응답(QA)을 통합적으로 제공하는 **AI 기반 분석 파이프라인**입니다.

이 프로젝트는 다음과 같은 기능을 제공합니다:
1. **이미지 업로드 및 대표 이미지 선정**
2. **OCR을 통한 텍스트 추출 및 PII 마스킹 처리**
3. **이미지 설명 자동 생성**
4. **텍스트 임베딩 생성 및 벡터 DB 저장**
5. **최근/유사 컨텍스트 기반 행동 및 질문 예측**
6. **현재 및 과거 컨텍스트 기반 QA 응답**

---

## 📂 프로젝트 구조

```
mindtrack-ai/
├── app/
│   ├── app.py                     # FastAPI 엔드포인트 정의
│   ├── integration_service.py     # 통합 서비스 로직
│   ├── config_loader.py           # config.yaml 로드 유틸
│   ├── logging/
│   │   └── logger.py              # 로깅 유틸
│   └── uploads/                   # 업로드 이미지 저장 폴더
│
├── modules/
│   ├── __init__.py
│   ├── action_predictor/          # 행동 및 질문 예측 모듈
│   ├── history_qa/                # 과거+현재 컨텍스트 기반 QA
│   ├── image_description/         # 이미지 설명 및 임베딩
│   ├── image_selector/            # 대표 이미지 선택
│   └── ocr_pii/                    # OCR + PII 마스킹
│
├── vectorstore/                   # 벡터 DB 저장
│
├── tests/                         # 유닛 테스트 (Placeholder)
│
├── config.yaml                    # 전역 설정 파일
└── requirements.txt               # Python 의존성
```

---

## ⚙️ 실행 흐름 (Integration Pipeline)

### 1. `/upload-and-process` API 호출
- **다중 이미지 업로드**
- 업로드된 이미지 `app/uploads`에 저장
- `IntegrationService.run_image_cycle()` 호출

### 2. 대표 이미지 선정 (ImageClusterSelector)
- 업로드된 모든 이미지 임베딩 추출
- KMeans 기반 클러스터링
- 가장 큰 클러스터에서 메도이드 이미지 선택

### 3. OCR + PII 처리
- `pytesseract`를 사용하여 텍스트 추출
- `Presidio Analyzer`로 개인정보 탐지
- 탐지된 PII 영역 블러 처리

### 4. 이미지 설명 생성 (ImageDescription)
- OpenAI API(`gpt-4.1-mini`)에 이미지와 프롬프트 전달
- 이미지에 대한 상세 JSON 기반 설명 생성

### 5. 텍스트 임베딩 생성 (EmbeddingGenerator)
- OpenAI Embeddings API(`text-embedding-3-small`) 사용
- FAISS 기반 벡터DB에 저장

### 6. 컨텍스트 분석
- 최근 N개(`recent_k`) 기록 불러오기
- 현재 설명과 유사한 상위 M개(`search_top_k`) 검색

### 7. 행동 및 질문 예측 (ActionPredictor)
- OpenAI API(`gpt-4.1-mini`)로 행동 및 예상 질문 예측

### 8. 최종 응답
- 대표 이미지 경로
- 설명 텍스트
- 예측 행동 리스트
- 예측 질문 리스트

---

## 🗂 config.yaml 설정

| 섹션 | 키 | 설명 | 기본값 |
|------|----|------|--------|
| `image_selector` | `n_clusters` | KMeans 클러스터 개수(자동 결정시 None) | `None` |
| `image_selector` | `random_state` | KMeans 랜덤 시드 | `42` |
| `openai` | `image_description_model` | 이미지 설명 모델명 | `gpt-4.1-mini` |
| `openai` | `embedding_model` | 임베딩 모델명 | `text-embedding-3-small` |
| `openai` | `action_predictor_model` | 행동 예측 모델명 | `gpt-4.1-mini` |
| `openai` | `history_qa_model` | QA 모델명 | `gpt-5-mini` |
| `vectordb` | `path` | 벡터DB 저장 경로 | `./vectorstore/description_index.meta` |
| `vectordb` | `dim` | 벡터 차원 수 | `1536` |
| `vectordb` | `recent_k` | 최근 검색 개수 | `3` |
| `vectordb` | `search_top_k` | 유사 검색 개수 | `2` |
| `integration` | `sample_dir` | 샘플 업로드 경로 | `app/sample/uploads` |
| `app` | `app_host` | 서버 호스트 | `0.0.0.0` |
| `app` | `app_port` | 서버 포트 | `8000` |

---

## 📌 주요 클래스 및 함수 설명

### 1. `IntegrationService`
**위치:** `app/integration_service.py`  
**설명:** 전체 파이프라인 실행을 담당하는 메인 서비스

#### `__init__(self)`
- 설정(`config.yaml`)을 로드하여 각 모듈 초기화

#### `run_image_cycle(self, upload_dir: str) -> dict`
- 전체 이미지 처리 파이프라인 실행
- **파라미터:** `upload_dir` (이미지 디렉토리 경로)
- **리턴:** `dict` (대표 이미지, 설명, 예측 행동, 예측 질문)

#### `answer_question(self, current_context, recent_context, similar_context, user_question)`
- 컨텍스트 + 질문 기반 QA 실행

---

### 2. `ImageClusterSelector`
**위치:** `modules/image_selector/selector.py`  
**역할:** 대표 이미지 자동 선택

#### `select(self, directory: str) -> Tuple[str, List[str]]`
- **입력:** 이미지 폴더 경로
- **출력:** `(대표 이미지 경로, 전체 이미지 경로 리스트)`
- **처리:**  
  1. 이미지 로드 & 전처리
  2. CNN(ResNet-18) 특징 추출
  3. KMeans로 클러스터링
  4. 가장 큰 클러스터의 중심에 가까운 이미지 선택

---

### 3. `OCR & PII Detection`
**위치:** `modules/ocr_pii`  
- `initialize_tesseract()` : Tesseract OCR 초기화
- `extract_text_data(image)` : 이미지 → 텍스트 DataFrame 추출
- `initialize_analyzer()` : Presidio Analyzer 초기화
- `analyze_and_blur_image(image_path, analyzer)` : 이미지에서 PII 탐지 후 블러 처리

---

### 4. `ImageDescription`
**위치:** `modules/image_description/description.py`  
- `_encode_image(image_path)` : 이미지 Base64 인코딩
- `generate_description(image_path)` : OpenAI API로 설명 생성

---

### 5. `EmbeddingGenerator`
**위치:** `modules/image_description/embedding.py`  
- `generate_embedding(text)` : OpenAI API로 임베딩 생성

---

### 6. `VectorDBStorage`
**위치:** `modules/image_description/storage.py`  
- `add_vector(embedding, metadata)` : 벡터 + 메타데이터 저장
- `search_vector(query_embedding, top_k)` : 유사 벡터 검색
- `get_recent(k)` : 최근 k개 메타데이터 반환
- `save()` : 인덱스 저장

---

### 7. `ActionPredictor`
**위치:** `modules/action_predictor/predictor.py`  
- `predict(current_context, recent_context, similar_context)`  
  → 행동 및 예상 질문 예측

---

### 8. `HistoryQA`
**위치:** `modules/history_qa/qa.py`  
- `answer(current_context, recent_context, similar_context, user_question)`  
  → 컨텍스트 기반 QA 응답

---

## 🌐 API 명세

### `POST /upload-and-process`
- **설명:** 이미지 업로드 후 전체 파이프라인 실행
- **입력:** `multipart/form-data` → `files` (다중 이미지)
- **출력:**
```json
{
  "representative_image": "path/to/image.png",
  "description": "...",
  "predicted_actions": ["..."],
  "predicted_questions": ["..."]
}
```

### `POST /answer-question`
- **설명:** 컨텍스트 + 질문 기반 QA
- **입력:**
```json
{
  "current_context": "...",
  "recent_context": "...",
  "similar_context": "...",
  "user_question": "..."
}
```
- **출력:**
```json
{
  "answer": "..."
}
```

---

## 🚀 실행 방법

```bash
# 1. 환경 설치
pip install -r requirements.txt

# 2. 환경 변수 설정
export OPENAI_API_KEY="your_api_key"
export TESSERACT_PATH="/usr/bin/tesseract"

# 3. 서버 실행
python app/app.py

# 또는 uvicorn 사용
uvicorn app.app:app --reload
```

---

## 📌 확장 가이드
- 새로운 PII 패턴 추가 → `pii_detection.py`에 `PatternRecognizer` 추가
- 이미지 설명 모델 변경 → `config.yaml`의 `openai.image_description_model` 수정
- 벡터DB 변경(Faiss → 다른 DB) → `VectorDBStorage` 클래스 교체
- 다른 임베딩 모델 사용 → `EmbeddingGenerator` 수정

---

## 🛠 개발자 참고
- 모든 주요 파라미터는 `config.yaml`에서 관리
- 각 모듈은 독립적으로 실행 가능 (예: `python modules/image_selector/selector.py`)
- OpenAI API 호출 시 요금이 발생하므로 개발 시 `max_output_tokens` 조정 권장
