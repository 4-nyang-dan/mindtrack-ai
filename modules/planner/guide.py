import os
import json
from openai import OpenAI

class PlannerGuide:
    def __init__(self, model_name="gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model_name

        base_dir = os.path.dirname(__file__)
        prompt_path = os.path.join(base_dir, "prompts", "guide.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

    def _compose_prompt(self, goal, step_number, caption, total_steps, action, detail):
        return (
            self.prompt_template
            .replace("{{goal}}", goal.strip())
            .replace("{{step_number}}", str(step_number))
            .replace("{{total_steps}}", str(total_steps))
            .replace("{{caption}}", str(caption))  # caption 필드 추가
            .replace("{{action}}", action.strip())
            .replace("{{detail}}", detail.strip())
        )

    def run(self, goal, step_number, caption, total_steps, action, detail):
        prompt = self._compose_prompt(goal, step_number, caption, total_steps, action, detail)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        raw = resp.choices[0].message.content.strip()
        cleaned = raw.replace("```json", "").replace("```", "").strip()

        return json.loads(cleaned)


# 시연용 main — planner/guide.py 파일 하단에 붙여 사용하세요.
if __name__ == "__main__":
    import sys
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    plan_path = os.path.join(base_dir, "app", "sample", "planner", "example1_detailed", "detailed_plan.json")
    out_dir = os.path.join(base_dir, "app", "sample", "planner", "example1_guide")
    out_path = os.path.join(out_dir, "guide_result.json")

    # 안전 체크: plan 파일 존재 여부
    if not os.path.exists(plan_path):
        print(f"Plan 파일을 찾을 수 없습니다: {plan_path}", file=sys.stderr)
        sys.exit(1)

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    # 데모에서는 step 2 사용
    current_step_no = 2
    step = next((s for s in plan.get("steps", []) if s.get("step") == current_step_no), None)
    if step is None:
        print(f"Plan에서 step {current_step_no}을(를) 찾을 수 없습니다.", file=sys.stderr)
        sys.exit(1)

    goal = plan.get("goal", "")
    total_steps = plan.get("total_steps", len(plan.get("steps", [])))
    action = step.get("action", "")
    detail = step.get("detail", "")

    # PlannerGuide 인스턴스 생성 및 실행
    guide_gen = PlannerGuide()
    try:
        result = guide_gen.run(goal=goal, step_number=current_step_no, total_steps=total_steps, action=action, detail=detail)
    except Exception as e:
        print(f"Guide 생성 중 오류 발생: {e}", file=sys.stderr)
        result = {"guide": f"[ERROR] {e}"}

    # 저장 및 출력
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as wf:
        json.dump(result, wf, ensure_ascii=False, indent=2)

    print("=== GUIDE RESULT ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved -> {out_path}")



