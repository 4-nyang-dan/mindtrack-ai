import os
import json
from typing import Dict, List, Optional, Union
from openai import OpenAI

class StepDetailer:
    """
    plan(JSON dict 또는 파일)을 받아 steps 개수만큼 LLM을 각 step별로 실행해
    '장황하지 않은' 상세 지침과 예상 질문(predicted_questions)을 생성하고, JSON으로 반환한다.
    """
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model_name

        base_dir = os.path.dirname(__file__)
        prompt_path = os.path.join(base_dir, "prompts", "detail.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.detail_prompt = f.read()

    def _normalize_plan(self, plan: Dict) -> Dict:
        steps = plan.get("steps") or []
        for i, s in enumerate(steps, 1):
            if "step" not in s:
                s["step"] = i
        plan["steps"] = steps
        plan["total_steps"] = len(steps)
        if "required_resources" not in plan:
            plan["required_resources"] = []
        return plan

    def _build_prompt(self, goal: str, plan_json_str: str, step_number: int, step_action: str) -> str:
        """LLM에 넘길 프롬프트 텍스트 구성"""
        return (
            self.detail_prompt
            .replace("{{goal}}", goal.strip())
            .replace("{{plan_json}}", plan_json_str)
            .replace("{{step_number}}", str(step_number))
            .replace("{{step_action}}", step_action.strip())
        )

    def _call_llm(self, prompt: str) -> Dict:
        """LLM 호출 -> JSON 결과(dict) 반환"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "너는 단계별 실행 지침을 작성하는 전문가다. "
                        "출력은 반드시 JSON 형식으로 반환해야 한다. "
                        "형식은 {\"detail\": \"...\", \"predicted_questions\": [\"...\", \"...\", \"...\"]} 이다."
                    )
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        raw = response.choices[0].message.content.strip()
        cleaned = raw.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(cleaned)
        except Exception as e:
            # 파싱 실패 시 fallback 구조 반환
            return {
                "detail": f"[LLM 응답 파싱 실패] {e}\n{cleaned[:200]}...",
                "predicted_questions": []
            }

    def run(
        self,
        plan_input: Union[str, Dict],
        print_to_console: bool = True,
        save_dir: Optional[str] = None
    ) -> Dict:
        """
        plan_input으로 JSON 파일 경로(str) 또는 dict(JSON 객체)를 받아 실행.
        - JSON 파일을 주면 파일을 읽고
        - dict를 주면 그대로 사용
        결과는 JSON(dict) 반환
        """
        # 입력 처리
        if isinstance(plan_input, str):
            if not os.path.exists(plan_input):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {plan_input}")
            with open(plan_input, "r", encoding="utf-8") as f:
                plan = json.load(f)
        elif isinstance(plan_input, dict):
            plan = plan_input
        else:
            raise TypeError("plan_input은 파일 경로(str) 또는 JSON(dict) 이어야 합니다.")

        plan = self._normalize_plan(plan)
        goal = plan.get("goal", "")
        steps = plan["steps"]
        total_steps = plan["total_steps"]
        plan_json_str = json.dumps(plan, ensure_ascii=False, indent=2)

        if print_to_console:
            print("==== 상세 지침 생성 시작 ====")
            print(f"목표: {goal}")
            print(f"총 단계: {total_steps}")
            print("-----------------------------")

        detailed_steps: List[Dict] = []

        for s in steps:
            step_no = s["step"]
            action = s.get("action", "").strip()

            if print_to_console:
                print(f"[{step_no}/{total_steps}] 단계 처리 시작: {action}")

            prompt = self._build_prompt(goal, plan_json_str, step_no, action)
            try:
                llm_result = self._call_llm(prompt)
                detail_text = llm_result.get("detail", "").strip()
                predicted_questions = llm_result.get("predicted_questions", [])
                status = "성공"
            except Exception as e:
                detail_text = f"[LLM 호출 실패] {e}"
                predicted_questions = []
                status = "실패"

            if print_to_console:
                print(f"[{step_no}/{total_steps}] 단계 처리 {status}\n")

            detailed_steps.append({
                "step": step_no,
                "action": action,
                "detail": detail_text,
                "predicted_questions": predicted_questions
            })

        result = {
            "goal": goal,
            "total_steps": total_steps,
            "steps": detailed_steps,
            "required_resources": plan.get("required_resources", [])
        }

        if print_to_console:
            print("==== 상세 지침 결과(JSON) ====")
            print(json.dumps(result, ensure_ascii=False, indent=2))

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, "detailed_plan.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            if print_to_console:
                print(f"결과 저장: {out_path}")

        return result


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    plan_path = os.path.join(base_dir, "app", "sample", "planner", "example1_plan.json")
    save_dir = os.path.join(base_dir, "app", "sample", "planner", "example1_detailed")

    detailer = StepDetailer()

    # 파일로 실행
    detailer.run(plan_path, print_to_console=True, save_dir=save_dir)

    # JSON 객체로 실행
    plan_data = {
        "goal": "운전면허증을 갱신한다.",
        "total_steps": 3,
        "steps": [
            {"step": 1, "action": "도로교통공단 사이트 접속"},
            {"step": 2, "action": "본인인증 및 로그인"},
            {"step": 3, "action": "갱신 신청 및 결제"}
        ],
        "required_resources": ["도로교통공단", "본인인증 수단", "결제수단"]
    }

    detailer.run(plan_data, print_to_console=True)
