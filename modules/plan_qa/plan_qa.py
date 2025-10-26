import os
import json
import random
from typing import Dict
from openai import OpenAI


class PlanQAModule:
    """
    plan 전체와 현재 step의 detail을 기반으로
    사용자의 질문에 대한 답변을 생성하는 모듈.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model_name

        base_dir = os.path.dirname(__file__)
        prompt_path = os.path.join(base_dir, "prompts", "qa.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.qa_prompt = f.read()

    def _build_prompt(
        self,
        plan: Dict,
        step_number: int,
        question: str
    ) -> str:
        """LLM 입력용 프롬프트 문자열 구성"""
        goal = plan.get("goal", "")
        plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
        steps = plan.get("steps", [])
        current_step = next((s for s in steps if s.get("step") == step_number), None)

        if not current_step:
            raise ValueError(f"지정한 step {step_number}을(를) 찾을 수 없습니다.")

        step_action = current_step.get("action", "")
        step_detail = current_step.get("detail", "")

        prompt = (
            self.qa_prompt
            .replace("{{goal}}", goal.strip())
            .replace("{{plan_json}}", plan_json)
            .replace("{{step_number}}", str(step_number))
            .replace("{{step_action}}", step_action.strip())
            .replace("{{step_detail}}", step_detail.strip())
            .replace("{{user_question}}", question.strip())
        )
        return prompt

    def _call_llm(self, prompt: str) -> str:
        """LLM 호출하여 답변 생성"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "너는 사용자 질문에 단계별 실행 지침을 근거로 답변하는 조력자다. 답변은 간결하고 실용적으로 작성한다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    def answer_question(self, plan: Dict, step_number: int, question: str, print_to_console: bool = True) -> Dict:
        """주어진 plan, step_number, 사용자 질문에 대한 답변 생성 (JSON 반환)"""
        prompt = self._build_prompt(plan, step_number, question)
        answer = self._call_llm(prompt)

        result = {
            "goal": plan.get("goal", ""),
            "step": step_number,
            "action": next((s["action"] for s in plan["steps"] if s["step"] == step_number), ""),
            "question": question,
            "answer": answer
        }

        if print_to_console:
            print("==== Plan QA ====")
            print(f"목표: {result['goal']}")
            print(f"단계 {result['step']}: {result['action']}")
            print(f"질문: {result['question']}")
            print(f"답변: {result['answer']}")
            print("-----------------")

        return result


# -----------------------------------------------------------
# Main Test
# -----------------------------------------------------------
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    plan_path = os.path.join(base_dir,"app", "sample", "planner", "example1_detailed", "detailed_plan.json")

    if not os.path.exists(plan_path):
        raise FileNotFoundError(f"sample plan 파일을 찾을 수 없습니다: {plan_path}")

    with open(plan_path, "r", encoding="utf-8") as f:
        plan_data = json.load(f)

    qa = PlanQAModule()

    # 테스트할 step 무작위 선택
    step = random.choice(plan_data["steps"])
    step_no = step["step"]
    predicted = step.get("predicted_questions", [])

    if not predicted:
        print(f"⚠️ step {step_no}에 predicted_questions가 없습니다.")
    else:
        print(f"선택된 단계: {step_no} - {step['action']}")
        print(f"예측 질문 후보: {predicted[:2]}")
        print()

        # 1️⃣ predicted_question 중 하나로 테스트
        q1 = random.choice(predicted[:2])
        qa.answer_question(plan_data, step_no, q1)

        # 2️⃣ 새로운(LLM이 예상하지 않은) 질문 테스트
        custom_question = "이 단계를 모바일에서도 할 수 있나요?"
        qa.answer_question(plan_data, step_no, custom_question)
