import sys
import os
import json
from PIL import Image


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 0) PLAN 생성 + 상세화
from modules.planner.planner import GoalPlanner         # goal → draft plan
from modules.planner.detailer import StepDetailer       # draft → detailed plan
from modules.planner.guide import PlannerGuide          # text 가이드 생성

# 1) 이미지 클러스터링
from modules.image_selector import ImageClusterSelector

# 2) 화면 상황 설명 (실시간 가이드용)
from modules.planner.description.description import ImageDescription

# 3) 화면 UI 상황 캡션 (UI 네비게이션용)
from modules.planner.description.planner_description import PlannerScreenCaption

# 4) UI 감지 + 내비게이터
from modules.planner.detect_ui import UIDetector
from modules.planner.navigator import UINavigator


class PlanService:
    def __init__(self):
        self.plan = None
        self.current_step = 1

        self.cluster_selector = ImageClusterSelector()
        self.image_desc = ImageDescription()
        self.planner_guide = PlannerGuide()

        self.screen_captioner = PlannerScreenCaption()
        self.ui_detector = UIDetector()
        self.navigator = UINavigator()

    # -------------------------
    # PLAN 생성 (planner → detailer)
    # -------------------------
    def create_plan(self, goal: str):
        draft_plan = GoalPlanner().make_plan(goal)     # ✅ 수정: .run() → .make_plan()
        detailed_plan = StepDetailer().run(draft_plan) # ✅ 상세화
        self.plan = detailed_plan
        self.current_step = 1
        return detailed_plan

    # -------------------------
    # STEP 업데이트
    # -------------------------
    def update_step(self, new_step: int):
        if self.plan is None:
            raise ValueError("⚠️ 먼저 create_plan(goal) 을 실행하세요.")
        if not (1 <= new_step <= self.plan["total_steps"]):
            raise ValueError("⚠️ step 번호 범위 초과.")
        self.current_step = new_step

    # -------------------------
    # 1) 실시간 화면 기반 자연어 가이드
    #    upload_dir 내 이미지 → 대표 이미지 선택 → 이미지 설명 → PlannerGuide 적용
    # -------------------------
    def guide_realtime(self, upload_dir: str):
        if self.plan is None:
            raise ValueError("⚠️ Plan 없음. create_plan(goal) 먼저 실행하세요.")

        # 1. 대표 이미지 선택
        rep_image, _ = self.cluster_selector.select(upload_dir)

        # 2. 이미지 상황(description)
        desc_response = self.image_desc.generate_description(rep_image)
        screen_caption = desc_response.output_text.strip()

        # 3. 현재 step 정보
        step_info = next(s for s in self.plan["steps"] if s["step"] == self.current_step)

        # 4. PlannerGuide 호출
        guide_json = self.planner_guide.run(
            goal=self.plan["goal"],
            step_number=self.current_step,
            caption=screen_caption,
            total_steps=self.plan["total_steps"],
            action=step_info["action"],
            detail=step_info["detail"]
        )

        # screen_caption 은 **반환하지 않음** 
        return {
            "step": self.current_step,
            "representative_image": rep_image,
            "guide": guide_json["guide"]
        }

    # -------------------------
    # 2) 어떤 UI 눌러야 하는지 안내 (UI 기반 네비게이션)
    # -------------------------
    def guide_ui(self, image_path: str, step: int, user_message: str = None):
        if self.plan is None:
            raise ValueError("⚠️ Plan 없음. create_plan(goal) 먼저 실행하세요.")

        # ✅ 여기서 step 업데이트 반영
        self.update_step(step)

        step_info = next(s for s in self.plan["steps"] if s["step"] == self.current_step)

        # 화면 상 의미 설명 (UI 맥락)
        caption_dict = self.screen_captioner.run(self.plan, self.current_step, image_path)
        caption = caption_dict.get("screen_caption", "").strip()

        # UI 요소 감지
        ui_elements, _ = self.ui_detector.extract(image_path)

        # 이미지 크기
        im = Image.open(image_path)
        image_size = (im.width, im.height)

        # 네비게이션 모델 호출
        nav = self.navigator.choose(
            goal=self.plan["goal"],
            step_number=self.current_step,
            ui_elements=ui_elements,
            detail=step_info["detail"],
            caption=caption,
            image_size=image_size,
            user_message=user_message
        )

        return {
            "selected_element": nav["selected_element"],
            "guide": nav["guide"]
        }


if __name__ == "__main__":
    import os
    import json

    # 서비스 로드
    service = PlanService()

    # 1) PLAN 생성
    goal = "주민등록등본을 인터넷으로 발급받는다."
    print("\n[1] create_plan 함수 : 계획 생성 중...")
    plan = service.create_plan(goal)
    print(json.dumps(plan, ensure_ascii=False, indent=2))

    # 2) STEP 업데이트 (2단계 진행으로 가정)
    step_to_run = 2
    print(f"\n[2] update_step 함수 : 현재 진행 단계를 {step_to_run} 로 변경...")
    service.update_step(step_to_run)

    # 3) 실시간 화면 기반 가이드
    base_dir = os.path.dirname(__file__)
    upload_dir = os.path.join(base_dir, "sample", "plan_service_img")
    print("\n[3] guide_realtime 함수 : 실시간 화면 가이드 실행...")
    realtime_result = service.guide_realtime(upload_dir)
    print(json.dumps(realtime_result, ensure_ascii=False, indent=2))

    # 4) UI 클릭 안내
    ui_image = os.path.join(upload_dir, "example5.png")
    print("\n[4] guide_ui 함수 : UI 내에서 어디를 눌러야 하는지 안내 실행...")
    ui_result = service.guide_ui(image_path=ui_image, step=step_to_run)
    print(json.dumps(ui_result, ensure_ascii=False, indent=2))

    print("\n✅ 시연 완료.")
