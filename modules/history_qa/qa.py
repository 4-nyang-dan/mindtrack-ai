import os
from openai import OpenAI

class HistoryQA:
    def __init__(self, prompt_filename="history_qa_prompt.txt", model_name="gpt-5-mini"):
        self.client = OpenAI()
        self.model_name = model_name
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", prompt_filename)
        self.prompt_template = self._load_prompt(prompt_path)

    def _load_prompt(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def answer(self, current_context: str, recent_context: str, similar_context: str, user_question: str):
        prompt = (
            self.prompt_template
            .replace("{current_context}", current_context)
            .replace("{recent_context}", recent_context)
            .replace("{similar_context}", similar_context)
            .replace("{user_question}", user_question)
        )

        print("\n--- Prompt ---\n", prompt)

        resp = self.client.responses.create(
            model=self.model_name,
            input=prompt
        )
        return resp.output_text


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "../../app/sample/description")

    # 현재 작업 (description4)
    with open(os.path.join(base_dir, "description4.txt"), "r", encoding="utf-8") as f:
        current_context = f.read().strip()

    # 최근 작업 (description3)
    with open(os.path.join(base_dir, "description3.txt"), "r", encoding="utf-8") as f:
        recent_context = f.read().strip()

    # 유사 작업 (description1)
    with open(os.path.join(base_dir, "description1.txt"), "r", encoding="utf-8") as f:
        similar_context = f.read().strip()

    # 사용자 질문
    user_question = "디지털 포렌식에서의 LLM 활용과 이에 대한 취약점을 설명해줘."

    qa = HistoryQA()
    answer = qa.answer(current_context, recent_context, similar_context, user_question)
    print("\n=== QA 결과 ===\n", answer)
