import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config_loader import config
from modules.image_selector import ImageClusterSelector
from modules.ocr_pii import initialize_tesseract, initialize_analyzer, analyze_and_blur_image
from modules.image_description import ImageDescription, EmbeddingGenerator, VectorDBStorage
from modules.action_predictor import ActionPredictor
from modules.history_qa import HistoryQA


class IntegrationService:
    def __init__(self):
        ## 초기화 (시간 측정 가능)
        initialize_tesseract()
        self.analyzer = initialize_analyzer()

        # 모듈 초기화
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

        ## 초기화 완료 (시간 측정 가능)
        print("[Init 완료] 통합 서비스 초기화 완료")

    def run_image_cycle(self, upload_dir: str):
        print(f"\n전체 이미지 폴더 처리 시작: {upload_dir}\n")

        ## 1. 대표 이미지 선택
        rep_img_path, all_imgs = self.selector.select(upload_dir)
        print(f"대표 이미지: {rep_img_path}")
        print(f"총 {len(all_imgs)}개 이미지 클러스터링 완료")

        ## 2. OCR + PII 분석
        blurred_img, _ = analyze_and_blur_image(rep_img_path, self.analyzer)
        if blurred_img is None:
            raise ValueError("이미지 처리 실패")

        ## 3. 이미지 설명 생성
        desc_response = self.image_desc.generate_description(rep_img_path)
        description_text = desc_response.output_text.strip()
        print(f"이미지 설명 요약: {description_text[:80]}...")

        ## 4. 임베딩 생성 및 저장
        embedding = self.embed_gen.generate_embedding(description_text)
        self.db.add_vector(embedding, {
            "file": os.path.basename(rep_img_path),
            "text": description_text
        })
        self.db.save()

        ## 5. 폴더 컨텍스트 구성
        folder_context = [os.path.basename(p) for p in all_imgs if p != rep_img_path]
        context_text = "\n".join(folder_context)

        ## 6. 벡터 DB 검색
        if self.db.metadata:
            recent_items = self.db.get_recent(k=config["vectordb"]["recent_k"])
            recent_context = recent_items[0]["text"] if recent_items else ""
            similar_results = self.db.search_vector(
                embedding,
                top_k=config["vectordb"]["search_top_k"]
            )
            similar_context = similar_results[0]["metadata"]["text"] if similar_results else ""
        else:
            recent_context, similar_context = "", ""

        ## 7. 행동 예측 (대표 이미지 + 폴더 컨텍스트)
        prompt_context = (
            f"대표 이미지 설명:\n{description_text}\n\n"
            f"폴더 내 다른 이미지들:\n{context_text}"
        )
        action_prediction_json = self.action_predictor.predict(
            prompt_context, recent_context, similar_context
        )

        ## 모델 원본 응답 출력
        print("\n[모델 원본 응답]")
        print(repr(action_prediction_json))
        print("============================\n")

        ## 8. JSON 파싱
        try:
            raw_text = action_prediction_json.strip()
            if not raw_text:
                print("[경고] action_prediction이 비어 있음, 기본값 설정")
                action_prediction = {"predicted_actions": [], "predicted_questions": []}
            else:
                cleaned = (
                    raw_text.replace("```json", "")
                    .replace("```", "")
                    .strip()
                )
                action_prediction = json.loads(cleaned)
        except Exception as e:
            print(f"[경고] JSON 파싱 실패: {e}")
            print("원본 출력:", repr(action_prediction_json))
            action_prediction = {"predicted_actions": [], "predicted_questions": []}

        ## 결과 반환
        return {
            "representative_image": rep_img_path,
            "description": description_text,
            "cluster_size": len(all_imgs),
            "cluster_images": folder_context,
            "predicted_actions": action_prediction.get("predicted_actions", []),
            "predicted_questions": action_prediction.get("predicted_questions", [])
        }

    ## HistoryQA 기능
    def answer_question(self, user_question: str):
        """
        사용자의 질문(user_question)에 대해, 최근 이미지 설명 기반으로 답변 생성.
        VectorDB에서 자동으로 context를 구성합니다.
        """

        #  1. 벡터 DB가 비어있는 경우
        if not self.db.metadata:
            print("[경고] 벡터 DB에 데이터가 없습니다.")
            return "데이터가 충분하지 않아 답변을 생성할 수 없습니다."

        #  2. 최근 설명(현재 컨텍스트)
        current_item = self.db.metadata[-1]
        current_context = current_item["text"]

        #  3. 최근 기록 (최근 k개)
        recent_items = self.db.get_recent(k=config["vectordb"]["recent_k"])
        recent_context = "\n\n".join([item["text"] for item in recent_items if item["id"] != current_item["id"]])

        #  4. 유사 컨텍스트 검색
        #    현재 이미지 설명의 임베딩을 다시 생성 후 벡터 검색
        embedding = self.embed_gen.generate_embedding(current_context)
        similar_results = self.db.search_vector(
            embedding,
            top_k=config["vectordb"]["search_top_k"],
            exclude_id=current_item["id"]
        )
        similar_context = "\n\n".join([r["metadata"]["text"] for r in similar_results])

        #  5. 히스토리 기반 Q&A 수행
        answer = self.history_qa.answer(
            current_context=current_context,
            recent_context=recent_context,
            similar_context=similar_context,
            user_question=user_question
        )

        #  6. 결과 출력
        print("\n[질문]")
        print(user_question)
        print("\n[답변]")
        print(answer)

        return answer


if __name__ == "__main__":
    service = IntegrationService()
    sample_dir = os.path.join(config["integration"]["sample_dir"])

    if not os.path.exists(sample_dir):
        raise FileNotFoundError(f"{sample_dir} 폴더가 없습니다.")

    print(f"이미지 디렉토리: {sample_dir}\n")

    ## 동일 폴더를 3회 반복 처리
    for i in range(3):
        print(f"\n=== [{i+1}회차 이미지 처리 시작] ===")
        result = service.run_image_cycle(sample_dir)
        print(f"\n[{i+1}회차 통합 모듈 결과]")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    ## 모든 사이클 완료 후 QA 테스트 수행
    print("\n============================")
    print("[HistoryQA 테스트 시작]")
    user_question = "현재 나는 어떤 일들을 하고 있는지 설명해줘."
    answer = service.answer_question(user_question)
    print("\n[최종 QA 결과]")
    print(answer)