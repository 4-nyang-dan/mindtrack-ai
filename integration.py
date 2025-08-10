# app/integration.py

import os
import json

from modules.image_selector import ImageClusterSelector
from modules.ocr_pii import initialize_tesseract, initialize_analyzer, analyze_and_blur_image
from modules.image_description import ImageDescription, EmbeddingGenerator, VectorDBStorage
from modules.action_predictor import ActionPredictor
from modules.history_qa import HistoryQA


class IntegrationService:
    def __init__(self):
        self.selector = ImageClusterSelector()
        self.image_desc = ImageDescription()
        self.embed_gen = EmbeddingGenerator()
        self.db = VectorDBStorage()
        self.action_predictor = ActionPredictor()
        self.history_qa = HistoryQA()

    def run_image_cycle(self, upload_dir: str):
        """
        주기적 이미지 처리:
        1. 대표 이미지 선정
        2. OCR/PII 처리
        3. 설명 생성
        4. 임베딩 생성
        5. VectorDB 저장
        6. 컨텍스트 추출
        7. 다음 행동/예상 질문 예측
        """
        # 1. 대표 이미지 선택
        rep_img_path, all_images = self.selector.select(upload_dir)
        print(f"[대표 이미지] {rep_img_path}")

        # 2. OCR & PII 처리
        initialize_tesseract()
        analyzer = initialize_analyzer()
        blurred_img, pii_boxes = analyze_and_blur_image(rep_img_path, analyzer)
        if blurred_img is None:
            raise ValueError("이미지 처리 실패")

        # 3. 설명 생성
        desc_response = self.image_desc.generate_description(rep_img_path)
        description_text = desc_response.output_text.strip()
        print(f"[이미지 설명] {description_text}")

        # 4. 임베딩 생성
        embedding = self.embed_gen.generate_embedding(description_text)

        # 5. VectorDB 저장
        self.db.add_vector(embedding, {
            "file": os.path.basename(rep_img_path),
            "text": description_text
        })
        self.db.save()

        # 6. 컨텍스트 가져오기 (예외 처리 포함)
        if self.db.metadata:
            recent_items = self.db.get_recent(k=3)
            recent_context = recent_items[0]["text"] if recent_items else ""
            similar_results = self.db.search_vector(embedding, top_k=2)
            similar_context = similar_results[0]["metadata"]["text"] if similar_results else ""
        else:
            recent_context = ""
            similar_context = ""

        # 7. 행동/질문 예측 (JSON 파싱)
        action_prediction_json = self.action_predictor.predict(
            description_text, recent_context, similar_context
        )
        try:
            action_prediction = json.loads(action_prediction_json)
        except Exception as e:
            print(f"[경고] action_prediction JSON 파싱 실패: {e}")
            action_prediction = {
                "predicted_actions": [],
                "predicted_questions": []
            }

        return {
            "representative_image": rep_img_path,
            "description": description_text,
            #"embedding": embedding,
            "predicted_actions": action_prediction.get("predicted_actions", []),
            "predicted_questions": action_prediction.get("predicted_questions", [])
        }

    def answer_question(self, current_context: str, recent_context: str, similar_context: str, user_question: str):
        """사용자 질문 기반 QA"""
        return self.history_qa.answer(
            current_context, recent_context, similar_context, user_question
        )


if __name__ == "__main__":
    service = IntegrationService()

    # === 시연 ===
    sample_dir = os.path.join(os.path.dirname(__file__),"app", "sample", "uploads")
    if not os.path.exists(sample_dir):
        raise FileNotFoundError(f"{sample_dir} 폴더가 없습니다.")

    result = service.run_image_cycle(sample_dir)
    print("\n=== 통합 모듈 결과 ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    