import os
from openai import OpenAI

class ActionPredictor:
    def __init__(self, prompt_filename="action_predictor_prompt.txt", model_name="gpt-4.1-mini"):
        self.client = OpenAI()
        self.model_name = model_name
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", prompt_filename)
        self.prompt_template = self._load_prompt(prompt_path)

    def _load_prompt(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def predict(self, current_context: str, recent_context: str, similar_context: str):
        prompt = (
            self.prompt_template
            .replace("{current_context}", current_context)
            .replace("{recent_context}", recent_context)
            .replace("{similar_context}", similar_context)
        )

        resp = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            temperature=0.3
        )
        return resp.output_text


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "../../app/sample/description")

    with open(os.path.join(base_dir, "description4.txt"), "r", encoding="utf-8") as f:
        current_context = f.read().strip()
    with open(os.path.join(base_dir, "description3.txt"), "r", encoding="utf-8") as f:
        recent_context = f.read().strip()
    with open(os.path.join(base_dir, "description1.txt"), "r", encoding="utf-8") as f:
        similar_context = f.read().strip()

    predictor = ActionPredictor()
    result = predictor.predict(current_context, recent_context, similar_context)
    print("\n=== 다음 행동 & 예상 질문 예측 ===\n", result)
