# ocr_pii/pii_detection.py
import cv2
import argparse
import re
import os
import spacy
import matplotlib.pyplot as plt
from spacy import cli as spacy_cli

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider, SpacyNlpEngine


from .ocr import initialize_tesseract, extract_text_data
from .blur import union_boxes, blur_area


def ensure_spacy_model(model_name="ko_core_news_sm"):
    try:
        spacy.load(model_name)
    except OSError:
        print(f"[spaCy] '{model_name}' 모델이 없어 설치를 시도합니다...")
        spacy_cli.download(model_name)

def initialize_analyzer():
    print("Presidio Analyzer Engine을 초기화하는 중입니다...")
    ensure_spacy_model("ko_core_news_sm")
    ensure_spacy_model("en_core_web_lg")

    # --- 기존 패턴들 ---
    rrn_pattern = Pattern(name="RRN Pattern", regex=r'\d{6}[-\s\.]?\d{7}', score=1.0)
    api_key_pattern = Pattern(name="API Key Pattern", regex=r'[A-Za-z0-9_=-]{20,}', score=0.8)
    phone_pattern_kr = Pattern(name="Phone Number KR", regex=r'((010|011|016|017|018|019|02|0[3-6][1-4])[-\s\.]?\d{3,4}[-\s\.]?\d{4})', score=1.0)
    bank_account_pattern_kr = Pattern(name="Bank Account KR", regex=r'\b[\d-]{10,18}\b', score=0.75)

    imperfect_email_pattern = Pattern(
        name="Imperfect Email Pattern",
        regex=r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9-]+(com|net|org|kr|co|ac|io|dev)\b',
        score=0.6 
    )

    rrn_recognizer = PatternRecognizer(supported_entity="KR_RRN", patterns=[rrn_pattern], supported_language="ko")
    api_key_recognizer = PatternRecognizer(supported_entity="API_KEY", patterns=[api_key_pattern], supported_language="en")
    phone_recognizer_kr = PatternRecognizer(supported_entity="PHONE_NUMBER_KR", patterns=[phone_pattern_kr], supported_language="ko")
    bank_recognizer_kr = PatternRecognizer(supported_entity="BANK_ACCOUNT_KR", patterns=[bank_account_pattern_kr], supported_language="ko")
    imperfect_email_recognizer = PatternRecognizer(
        supported_entity="EMAIL_ADDRESS", 
        patterns=[imperfect_email_pattern]
    )

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers() 
    registry.add_recognizer(rrn_recognizer)
    registry.add_recognizer(api_key_recognizer)
    registry.add_recognizer(phone_recognizer_kr)
    registry.add_recognizer(bank_recognizer_kr)
    registry.add_recognizer(imperfect_email_recognizer)

    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}, {"lang_code": "ko", "model_name": "ko_core_news_sm"}]
    })
    nlp_engine = provider.create_engine()

    analyzer = AnalyzerEngine(registry=registry, nlp_engine=nlp_engine, default_score_threshold=0.4)
    print("Engine 초기화 완료.")
    return analyzer

def detect_pii(text, analyzer):
    korean_pii = ["KR_RRN", "PHONE_NUMBER_KR", "BANK_ACCOUNT_KR"]
    english_pii = ["EMAIL_ADDRESS", "API_KEY", "CREDIT_CARD_NUMBER"]
    
    ko_results = analyzer.analyze(text=text, language="ko", entities=korean_pii)
    en_results = analyzer.analyze(text=text, language="en", entities=english_pii)
    
    return ko_results + en_results


def analyze_and_blur_image(image_path, analyzer):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"오류: 이미지 파일을 찾을 수 없거나 손상되었습니다 - {image_path}")
            return None, []
        blurred_image = image.copy()
    except Exception as e:
        print(f"오류: 이미지 로드 실패 - {e}")
        return None, []

    ocr_df = extract_text_data(image)
    if ocr_df.empty:
        return blurred_image, []

    all_pii_boxes = []
    
    for _, line_df in ocr_df.groupby(['block_num', 'par_num', 'line_num']):
        line_text = ' '.join(line_df['text'].astype(str))
        analyzer_results = detect_pii(line_text, analyzer)
        
        if analyzer_results:
            for res in analyzer_results:
                pii_text = line_text[res.start:res.end]
                searchable_pii_text = re.sub(r'[^a-zA-Z0-9]', '', pii_text)
                if not searchable_pii_text: continue
                
                pii_word_boxes = []
                for _, word_row in line_df.iterrows():
                    word_text = str(word_row['text'])
                    cleaned_word_text = re.sub(r'[^a-zA-Z0-9]', '', word_text)
                    if cleaned_word_text and cleaned_word_text in searchable_pii_text:
                        x, y, w, h = int(word_row['left']), int(word_row['top']), int(word_row['width']), int(word_row['height'])
                        pii_word_boxes.append((x, y, w, h))
                
                if pii_word_boxes:
                    all_pii_boxes.append(union_boxes(pii_word_boxes))

    if all_pii_boxes:
        unique_boxes = list(set([tuple(box) for box in all_pii_boxes if box]))
        print(f" -> {len(unique_boxes)}개의 민감정보 영역을 블러 처리합니다.")
        for box in unique_boxes:
            blurred_image = blur_area(blurred_image, box)
            
    return blurred_image, all_pii_boxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR 기반 PII 탐지 및 블러 처리기")
    parser.add_argument("--path", type=str, required=True, help="처리할 이미지 파일의 전체 경로")
    args = parser.parse_args()

    initialize_tesseract()
    analyzer = initialize_analyzer()

    blurred_image, detected_boxes = analyze_and_blur_image(args.path, analyzer)

    if blurred_image is not None:
        output_dir = "blurred_results"
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        filename = os.path.basename(args.path)
        output_path = os.path.join(output_dir, f"blurred_{filename}")
        cv2.imwrite(output_path, blurred_image)

        print("\n" + "="*50)
        print("최종 처리 완료")
        print(f" -> 블러 처리된 이미지가 다음 경로에 저장되었습니다: {output_path}")

        if detected_boxes:
            print("\n탐지된 민감정보 좌표 (x, y, width, height):")
            for i, box in enumerate(detected_boxes):
                print(f"  - Box {i+1}: {box}")
        else:
            print("\n탐지된 민감정보가 없습니다.")

        if not detected_boxes:
            print("\n탐지된 민감정보가 없습니다.")
        
        print("="*50)
        
        try:
            final_blurred_image_rgb = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(12, 10))
            plt.imshow(final_blurred_image_rgb)
            plt.title("Blurred PII Result")
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"\n결과 이미지 표시에 실패했습니다: {e}")
    else:
        print("\n이미지 처리에 실패했습니다. 입력 파일 경로를 확인해주세요.")