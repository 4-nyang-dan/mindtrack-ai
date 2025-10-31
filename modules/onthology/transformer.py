import os
import json
from openai import OpenAI


class OntologyTransformer:
    """
    사용자의 화면 분석 문장(current_action)을
    구조화된 Scene Ontology로 변환하는 클래스.
    """
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model_name

        base_dir = os.path.dirname(__file__)
        prompt_path = os.path.join(base_dir, "prompts", "transform.txt")

        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

    def to_scene(self, caption: str) -> dict:
        """current_action 문장을 기반으로 Scene Ontology 생성"""
        prompt = self.prompt_template.replace("{{caption}}", caption.strip())

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "너는 사용자의 행동을 온톨로지 구조로 변환하는 AI 전문가야."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

            raw_output = response.choices[0].message.content.strip()
        except Exception as e:
            return {"error": f"LLM 호출 실패: {e}"}

        # JSON 파싱
        try:
            cleaned = raw_output.replace("```json", "").replace("```", "").strip()
            result = json.loads(cleaned)
        except Exception as e:
            result = {"error": f"JSON 파싱 실패: {e}", "raw_output": raw_output}

        return {"current_action": caption, "ontology": result}


# --------------------------
# 시연용 main
# --------------------------
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    desc_dir = os.path.join(base_dir, "app", "sample", "description")

    print(f"[테스트 시작] 샘플 폴더: {desc_dir}")

    transformer = OntologyTransformer()

    for i in range(1, 6):
        file_path = os.path.join(desc_dir, f"description{i}.txt")
        if not os.path.exists(file_path):
            print(f"⚠️ {file_path} 없음, 건너뜀")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            caption = data.get("current_action", "").strip()

        print(f"\n=== description{i}.txt ===")
        print(f"입력: {caption}\n")

        result = transformer.to_scene(caption)
        print("출력:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("=" * 80)
