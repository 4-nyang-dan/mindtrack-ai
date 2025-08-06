# AI Agent Project

This project provides OCR, PII detection, image selection, and GPT-based image description with vector storage using FAISS.



### Tesseract OCR 엔진 설치
이미지에서 텍스트를 인식(OCR)하기 위해 Tesseract 엔진이 필요합니다. 운영체제에 맞게 설치해주세요.

* 🖥️ **Windows**
    * [Windows용 Tesseract 설치 프로그램](https://github.com/UB-Mannheim/tesseract/wiki)을 다운로드하여 설치합니다.
    * ⚠️ **중요**: 설치 과정에서 **"Add Tesseract to system PATH"** 옵션을 반드시 체크해주세요.

* 🍏 **macOS**
    ```bash
    brew install tesseract tesseract-lang
    ```

* 🐧 **Linux (Ubuntu)**
    ```bash
    sudo apt update && sudo apt install tesseract-ocr tesseract-ocr-kor
    ```

---

### Spacy 언어 모델 설치
개인정보 탐지(PII Detection)에 필요한 한국어, 영어 언어 모델을 다운로드합니다.

```bash
python -m spacy download ko_core_news_sm
python -m spacy download en_core_web_lg