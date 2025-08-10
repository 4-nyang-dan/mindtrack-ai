import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
from config_loader import config

from modules.image_selector import ImageClusterSelector
from modules.ocr_pii import initialize_tesseract, initialize_analyzer, analyze_and_blur_image
from modules.image_description import ImageDescription, EmbeddingGenerator, VectorDBStorage
from modules.action_predictor import ActionPredictor
from modules.history_qa import HistoryQA


class IntegrationService:
    def __init__(self):
        self.selector = ImageClusterSelector(
            n_clusters=config["image_selector"]["n_clusters"],
            random_state=config["image_selector"]["random_state"]
        )
        self.image_desc = ImageDescription(
            model_name=config["openai"]["image_description_model"]
        )
        self.embed_gen = EmbeddingGenerator(
            model_name=config["openai"]["embedding_model"]
        )
        self.db = VectorDBStorage(
            db_dir=os.path.dirname(config["vectordb"]["path"]),
            index_name=os.path.splitext(os.path.basename(config["vectordb"]["path"]))[0],
            dim=config["vectordb"]["dim"]
        )
        self.action_predictor = ActionPredictor(
            model_name=config["openai"]["action_predictor_model"]
        )
        self.history_qa = HistoryQA(
            model_name=config["openai"]["history_qa_model"]
        )

    def run_image_cycle(self, upload_dir: str):
        rep_img_path, _ = self.selector.select(upload_dir)
        print(f"[대표 이미지] {rep_img_path}")

        initialize_tesseract()
        analyzer = initialize_analyzer()
        blurred_img, _ = analyze_and_blur_image(rep_img_path, analyzer)
        if blurred_img is None:
            raise ValueError("이미지 처리 실패")

        desc_response = self.image_desc.generate_description(rep_img_path)
        description_text = desc_response.output_text.strip()
        print(f"[이미지 설명] {description_text}")

        embedding = self.embed_gen.generate_embedding(description_text)
        self.db.add_vector(embedding, {
            "file": os.path.basename(rep_img_path),
            "text": description_text
        })
        self.db.save()

        if self.db.metadata:
            recent_items = self.db.get_recent(k=config["vectordb"]["recent_k"])
            recent_context = recent_items[0]["text"] if recent_items else ""
            similar_results = self.db.search_vector(
                embedding,
                top_k=config["vectordb"]["search_top_k"]
            )
            similar_context = similar_results[0]["metadata"]["text"] if similar_results else ""
        else:
            recent_context = ""
            similar_context = ""

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
            "predicted_actions": action_prediction.get("predicted_actions", []),
            "predicted_questions": action_prediction.get("predicted_questions", [])
        }

    def answer_question(self, current_context, recent_context, similar_context, user_question):
        return self.history_qa.answer(
            current_context, recent_context, similar_context, user_question
        )


if __name__ == "__main__":
    service = IntegrationService()
    sample_dir = os.path.join(config["integration"]["sample_dir"])
    if not os.path.exists(sample_dir):
        raise FileNotFoundError(f"{sample_dir} 폴더가 없습니다.")

    result = service.run_image_cycle(sample_dir)
    print("\n=== 통합 모듈 결과 ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
