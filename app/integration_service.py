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
        # 이미지 클러스터링 설정
        self.selector = ImageClusterSelector(
            n_clusters=config["image_selector"]["n_clusters"],
            random_state=config["image_selector"]["random_state"]
        )
        # 이미지 설명 모델 초기화
        self.image_desc = ImageDescription(
            model_name=config["openai"]["image_description_model"]
        )
        # 텍스트 임베딩 생성기 초기화
        self.embed_gen = EmbeddingGenerator(
            model_name=config["openai"]["embedding_model"]
        )
        # 벡터 DB 저장소 초기화
        self.db = VectorDBStorage(
            db_dir=os.path.dirname(config["vectordb"]["path"]),
            index_name=os.path.splitext(os.path.basename(config["vectordb"]["path"]))[0],
            dim=config["vectordb"]["dim"]
        )
        # 행동 예측 모델 초기화
        self.action_predictor = ActionPredictor(
            model_name=config["openai"]["action_predictor_model"]
        )
        # 질의 응답 모델 초기화
        self.history_qa = HistoryQA(
            model_name=config["openai"]["history_qa_model"]
        )

    def run_image_cycle(self, upload_dir: str):
        # 1. 대표 이미지 선택
        rep_img_path, _ = self.selector.select(upload_dir)
        print(f"[대표 이미지] {rep_img_path}")

        # 2. 이미지 처리 및 분석
        initialize_tesseract()
        analyzer = initialize_analyzer()

        blurred_img, _ = analyze_and_blur_image(rep_img_path, analyzer)
        if blurred_img is None:
            raise ValueError("이미지 처리 실패")

        # 3. 이미지 설명 생성
        desc_response = self.image_desc.generate_description(rep_img_path)
        description_text = desc_response.output_text.strip()
        print(f"[이미지 설명] {description_text}")

        # 4. 이미지 설명을 벡터화하고 벡터 DB에 저장
        embedding = self.embed_gen.generate_embedding(description_text)
        self.db.add_vector(embedding, {
            "file": os.path.basename(rep_img_path),
            "text": description_text
        })
        self.db.save()

        # 5. 벡터 DB에서 최근 항목 및 유사 항목 가져오기
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

        # 6. 행동 예측 모델에 입력값을 전달하고 예측값을 받음
        action_prediction_json = self.action_predictor.predict(
            description_text, recent_context, similar_context
        )

        # 7. 예측 결과가 빈 문자열인 경우, 기본값을 설정
        # 이유: 예측 모델이 빈 값을 반환하는 경우에도 코드가 중단되지 않도록 기본값을 설정.
        if not action_prediction_json.strip():  # 반환값이 빈 문자열일 경우
            print(f"[경고] action_prediction이 비어 있음, 기본값 설정")
            action_prediction = {
                "predicted_actions": [],
                "predicted_questions": []
            }
        else:
            # 8. 예측 결과를 JSON으로 파싱
            try:
                action_prediction = json.loads(action_prediction_json)
            except Exception as e:
                # 예외 발생 시 기본값을 설정하고 경고 메시지 출력
                print(f"[경고] action_prediction JSON 파싱 실패: {e}")
                action_prediction = {
                    "predicted_actions": [],
                    "predicted_questions": []
                }

        # 9. 최종 결과 반환
        return {
            "representative_image": rep_img_path,
            "description": description_text,
            "predicted_actions": action_prediction.get("predicted_actions", []),
            "predicted_questions": action_prediction.get("predicted_questions", [])
        }

    def answer_question(self, current_context, recent_context, similar_context, user_question):
        # 질의 응답 처리
        return self.history_qa.answer(
            current_context, recent_context, similar_context, user_question
        )


if __name__ == "__main__":
    # 10. IntegrationService 인스턴스 생성 및 실행
    service = IntegrationService()
    sample_dir = os.path.join(config["integration"]["sample_dir"])
    if not os.path.exists(sample_dir):
        raise FileNotFoundError(f"{sample_dir} 폴더가 없습니다.")

    # 11. 이미지 처리 및 분석 결과 출력
    result = service.run_image_cycle(sample_dir)
    print("\n=== 통합 모듈 결과 ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
