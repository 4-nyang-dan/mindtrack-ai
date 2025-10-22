import sys
import os
import json
import time  # 🔹 추가: 시간 측정용
import traceback

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
        
        self.db.reset()

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

        start_total = time.perf_counter()  # 🔹 전체 사이클 시작 시간

        ## 1️⃣ 대표 이미지 선택
        t1 = time.perf_counter()
        rep_img_path, all_imgs = self.selector.select(upload_dir)
        print(f"[1] 대표 이미지 선택 완료 ({len(all_imgs)}장) - {time.perf_counter() - t1:.2f}s")

        ## 2️⃣ OCR + PII 분석
        t2 = time.perf_counter()
        blurred_img, _ = analyze_and_blur_image(rep_img_path, self.analyzer)
        print(f"[2] OCR + PII 분석 완료 - {time.perf_counter() - t2:.2f}s")
        if blurred_img is None:
            raise ValueError("이미지 처리 실패")

        ## 3️⃣ 이미지 설명 생성
        t3 = time.perf_counter()
        desc_response = self.image_desc.generate_description(rep_img_path)
        description_text = desc_response.output_text.strip()
        print(f"[3] 이미지 설명 생성 완료 - {time.perf_counter() - t3:.2f}s")
        print(f"    └ 요약: {description_text[:80]}...")

        ## 4️⃣ 임베딩 생성 및 저장
        t4 = time.perf_counter()
        embedding = self.embed_gen.generate_embedding(description_text)
        self.db.add_vector(embedding, {
            "file": os.path.basename(rep_img_path),
            "text": description_text
        })

        print("[FAISS] 인덱스 저장 시도 중...")
        self.db.save()
        print(f"[4] 임베딩 생성 및 저장 완료 - {time.perf_counter() - t4:.2f}s")

        ## 5️⃣ 폴더 컨텍스트 구성
        t5 = time.perf_counter()
        folder_context = [os.path.basename(p) for p in all_imgs if p != rep_img_path]
        context_text = "\n".join(folder_context)
        print(f"[5] 폴더 컨텍스트 구성 완료 - {time.perf_counter() - t5:.2f}s")

        ## 6️⃣ 벡터 DB 검색
        t6 = time.perf_counter()
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
        print(f"[6] 벡터 DB 검색 완료 - {time.perf_counter() - t6:.2f}s")

        ## 7️⃣ 행동 예측
        t7 = time.perf_counter()
        prompt_context = (
            f"대표 이미지 설명:\n{description_text}\n\n"
            f"폴더 내 다른 이미지들:\n{context_text}"
        )
        action_prediction_json = self.action_predictor.predict(
            prompt_context, recent_context, similar_context
        )
        print(f"[7] 행동 예측 완료 - {time.perf_counter() - t7:.2f}s")

        ## 모델 원본 응답 출력
        print("\n[모델 원본 응답]")
        print(repr(action_prediction_json))
        print("============================\n")

        ## 8️⃣ JSON 파싱
        t8 = time.perf_counter()
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
        print(f"[8] JSON 파싱 완료 - {time.perf_counter() - t8:.2f}s")

        ## ✅ 전체 처리시간 출력
        total_time = time.perf_counter() - start_total
        print(f"\n✅ 전체 사이클 완료 - 총 {total_time:.2f}초 소요\n")

        ## 결과 반환
        return {
            "representative_image": rep_img_path,
            "description": description_text,
            "cluster_size": len(all_imgs),
            "cluster_images": folder_context,
            "predicted_actions": action_prediction.get("predicted_actions", []),
            "predicted_questions": action_prediction.get("predicted_questions", [])
        }


    def _format_ai_answer(self, user_question: str, answer: str):
        """
        모델 응답(JSON or 일반 문자열)을 파싱해
        Q / AI의 생각 / A 형태로 반환.
        """
        try:
            cleaned = (
                answer.replace("```json", "")
                .replace("```", "")
                .strip()
            )
            parsed = json.loads(cleaned)
        except Exception:
            # JSON 형태가 아니면 fallback
            parsed = {"reasoning_steps": [], "final_answer": answer.strip()}

        reasoning_steps = parsed.get("reasoning_steps", [])
        final_answer = parsed.get("final_answer", "").strip()

        # 포맷 구성
        if reasoning_steps:
            thoughts = "\n".join([f"{i+1}. {step}" for i, step in enumerate(reasoning_steps)])
        else:
            thoughts = "(AI의 세부 사고 단계 정보가 제공되지 않았습니다.)"

        # 최종 출력 구조
        return {
            "question": user_question.strip(),
            "ai_thoughts": thoughts,
            "answer": final_answer or "답변이 비어 있습니다."
        }

    def answer_question(self, user_question: str):
        """
        사용자의 질문(user_question)에 대해, 최근 이미지 설명 기반으로 답변 생성.
        VectorDB에서 자동으로 context를 구성합니다.
        """

        # === 컨텍스트 기본값 (없을 때도 프롬프트에서 처리 가능하도록 "X"로 설정)
        current_context = "X"
        recent_context = "X"
        similar_context = "X"

        try:
            if self.db.metadata:
                current_item = self.db.metadata[-1]
                current_context = current_item.get("text", "") or "X"

                # 최근 데이터
                recent_items = self.db.get_recent(k=config["vectordb"]["recent_k"])
                recent_context = "\n\n".join(
                    [it.get("text", "") for it in recent_items if it.get("id") != current_item.get("id")]
                ).strip() or "X"

                # 유사 검색
                if current_context and current_context != "X":
                    embedding = self.embed_gen.generate_embedding(current_context)
                    similar_results = self.db.search_vector(
                        embedding,
                        top_k=config["vectordb"]["search_top_k"],
                        exclude_id=current_item["id"]
                    )
                    similar_context = "\n\n".join(
                        [r["metadata"]["text"] for r in similar_results]
                    ).strip() or "X"

            # === 모델 호출
            answer = self.history_qa.answer(
                current_context=current_context,
                recent_context=recent_context,
                similar_context=similar_context,
                user_question=user_question
            )

            print("\n[질문]")
            print(user_question)
            print("\n[모델 RAW 응답]")
            print(answer)

            formatted = self._format_ai_answer(user_question, answer)
            print("\n[포맷팅된 결과]")
            print(json.dumps(formatted, ensure_ascii=False, indent=2))
            return formatted

        except Exception as e:
            print(f"[오류] answer_question 처리 중 예외 발생: {e}")
            traceback.print_exc()
            return {
                "question": user_question,
                "ai_thoughts": "(예외가 발생하여 기본 응답을 반환합니다.)",
                "answer": "답변을 생성하는 중 문제가 발생했습니다."
            }



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
