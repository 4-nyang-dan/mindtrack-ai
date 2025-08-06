# ocr_pii/ocr.py
import cv2
import pytesseract
import pandas as pd
import os
from dotenv import load_dotenv

def initialize_tesseract():
    load_dotenv()
    tesseract_cmd_path = os.getenv('TESSERACT_PATH')

    if tesseract_cmd_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path

    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        print("오류: Tesseract를 찾을 수 없습니다.")
        print("Tesseract를 설치하고, .env 파일에 TESSERACT_PATH를 설정하거나")
        print("시스템 환경 변수(PATH)에 Tesseract 경로를 추가해주세요.")
        exit()

def extract_text_data(image):
    height, width, _ = image.shape
    upscaled_image = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)
    
    gray = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)
    
    binary_image = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    ocr_df = pytesseract.image_to_data(binary_image, lang='kor+eng', output_type=pytesseract.Output.DATAFRAME)
    
    ocr_df = ocr_df[ocr_df.conf > 30].dropna(subset=['text'])
    
    if not ocr_df.empty:
        ocr_df[['left', 'top', 'width', 'height']] = (ocr_df[['left', 'top', 'width', 'height']] / 2).astype(int)
        
    return ocr_df