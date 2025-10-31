import os
import base64
import json
from openai import OpenAI


class PlannerScreenCaption:
    """
    상세 계획(detailed_plan)과 현재 step, 그리고 화면 이미지를 입력받아
    '현재 화면이 어떤 상태인지' 설명을 생성한다.
    (UI 선택 안내 X, 단순 화면 상황 설명)
    """

    def __init__(self, model_name="gpt-4.1-mini"):
        self.client = OpenAI()
        self.model_name = model_name

        # 프롬프트 로드
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "planner_description.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as img:
            return base64.b64encode(img.read()).decode("utf-8")

    def run(self, detailed_plan: dict, current_step: int, image_path: str):
        step_obj = next((s for s in detailed_plan["steps"] if s["step"] == current_step), None)
        if step_obj is None:
            raise ValueError(f"step {current_step} not found in plan")

        goal = detailed_plan["goal"]
        step_detail = step_obj["detail"]
        image_b64 = self._encode_image(image_path)

        prompt = (
            self.prompt_template
            .replace("{{goal}}", goal)
            .replace("{{step_number}}", str(current_step))
            .replace("{{step_detail}}", step_detail)
        )

        response = self.client.responses.create(
            model=self.model_name,
            temperature=0.2,
            input=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "다음 이미지를 설명해 주세요."},
                        {"type": "input_image", "image_url": f"data:image/png;base64,{image_b64}"}
                    ]
                }
            ]
        )

        raw = response.output_text.strip()
        cleaned = raw.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(cleaned)
        except:
            return {"screen_caption": raw}


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # Paths
    plan_path = os.path.join(base_dir,"../", "app", "sample", "planner", "example1_detailed", "detailed_plan.json")
    image_path = os.path.join(base_dir,"../", "app", "sample", "image", "example5.png")

    # Load plan
    with open(plan_path, "r", encoding="utf-8") as f:
        detailed_plan = json.load(f)

    # Run captioner
    captioner = PlannerScreenCaption()
    result = captioner.run(
        detailed_plan=detailed_plan,
        current_step=2,
        image_path=image_path
    )

    # Print result
    print("\n=== SCREEN CAPTION RESULT ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # Save to /planner/caption
    save_dir = os.path.join(base_dir, "../", "app", "sample", "planner", "caption")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "caption_result.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nSaved → {save_path}")
