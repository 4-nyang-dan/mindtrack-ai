import os
import json
from openai import OpenAI
import re

class UINavigator:
    """
    UI 요소 + 현재 단계 detail + 화면 caption 을 기반으로
    선택해야 할 UI 하나와 guide 문장을 생성한다.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model_name

        base_dir = os.path.dirname(__file__)
        prompt_path = os.path.join(base_dir, "prompts", "navigator.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

    def _compose_prompt(self, goal, step_number, detail, caption, ui_elements, user_message):
        ui_json = json.dumps(ui_elements, ensure_ascii=False)
        return (
            self.prompt_template
            .replace("{{goal}}", goal)
            .replace("{{step_number}}", str(step_number))
            .replace("{{detail}}", detail.strip())
            .replace("{{caption}}", caption.strip() if caption else "없음")
            .replace("{{ui_elements_json}}", ui_json)
            .replace("{{user_message}}", user_message or "없음")
        )

    def _infer_position(self, bbox, img_w, img_h):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1+x2)/2, (y1+y2)/2

        vert = "상단" if cy < img_h*0.33 else ("하단" if cy > img_h*0.66 else "중단")
        horiz = "왼쪽" if cx < img_w*0.33 else ("오른쪽" if cx > img_w*0.66 else "중앙")
        return f"{vert} {horiz}"
    
    def _clean_output_text(self, text: str):
        if not text:
            return text
        # 한글, 영문, 숫자만 남기고 나머지 제거
        text = re.sub(r"[^가-힣A-Za-z0-9]", "", text)
        return text.strip()

    def choose(self, goal, step_number, ui_elements, detail, caption, image_size, user_message=None):
        prompt = self._compose_prompt(goal, step_number, detail, caption, ui_elements, user_message)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        raw = response.choices[0].message.content.strip()
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        out = json.loads(cleaned)

        sel = out["selected_element"]
        bbox = sel["bbox"]
        text = self._clean_output_text(sel.get("text"))

        pos = self._infer_position(bbox, image_size[0], image_size[1])
        base_guide = out.get("guide", "")
        guide = f"**화면의 {pos}에 있는 '{text or '해당 버튼'}'을 눌러주세요.**\n {base_guide}".strip()

        return {
            "selected_element": {"bbox": bbox, "text": text},
            "guide": guide
        }


# ----------------------------
# 시연용 main: example 사용
# ----------------------------
if __name__ == "__main__":
    import json
    from PIL import Image

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    plan_path = os.path.join(base_dir, "app", "sample", "planner", "example1_detailed", "detailed_plan.json")
    image_path = os.path.join(base_dir, "app", "sample", "image", "example5.png")
    caption_path = os.path.join(base_dir, "app", "sample", "planner", "caption", "caption_result.json")

    try:
        from detect_ui import UIDetector
    except Exception:
        print("⚠️ detect_ui 모듈을 찾을 수 없어 목 데이터로 대체합니다.")
        UIDetector = None

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    current_step_no = 2
    current_step = next(s for s in plan["steps"] if s["step"] == current_step_no)
    detail_text = current_step["detail"]
    goal_text = plan["goal"]

    if os.path.exists(caption_path):
        with open(caption_path, "r", encoding="utf-8") as f:
            caption = json.load(f).get("caption", "")
    else:
        caption = ""

    if UIDetector:
        det = UIDetector(gpu=False)
        ui_elements = det.extract(image_path)
    else:
        print("⚠️ UIDetector 미사용 → 샘플 UI 사용")
        ui_elements = [
            {"bbox": [50, 30, 200, 70], "text": "로그인"},
            {"bbox": [210, 30, 360, 70], "text": "회원가입"},
            {"bbox": [400, 30, 520, 70], "text": "검색"}
        ]

    im = Image.open(image_path)
    image_size = (im.width, im.height)

    nav = UINavigator()
    result = nav.choose(goal_text, current_step_no, ui_elements, detail_text, caption, image_size, user_message="회원가입을 하고 싶어요")

    out_dir = os.path.join(base_dir, "app", "sample", "planner", "example1_navigation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "navigation_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n=== NAVIGATION RESULT ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"✅ Saved -> {out_path}")
