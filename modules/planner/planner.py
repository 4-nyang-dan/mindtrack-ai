import os
import json
import requests
from typing import List, Dict
from openai import OpenAI

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")   # Google Cloud API key
GOOGLE_CSE_ID  = os.getenv("GOOGLE_CSE_ID")    # Programmable Search Engine ID (cx)
GOOGLE_CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"


def google_search(query: str, num: int = 6, hl: str = "ko", gl: str = "kr", timeout: int = 10) -> List[Dict]:
    """Google Programmable Search(JSON API). 반환: [{title, snippet, link}, ...]"""
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        raise RuntimeError("GOOGLE_API_KEY 또는 GOOGLE_CSE_ID가 설정되지 않았습니다.")

    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": max(1, min(num, 10)),
        "hl": hl,
        "gl": gl,
        "safe": "off",
    }
    r = requests.get(GOOGLE_CSE_ENDPOINT, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    items = []
    for it in data.get("items", []):
        items.append({
            "title": it.get("title", "").strip(),
            "snippet": it.get("snippet", "").strip(),
            "url": it.get("link", "").strip(),
        })
    return items


def format_search_context(results: List[Dict]) -> str:
    """LLM 프롬프트용 참고정보 문자열 생성."""
    if not results:
        return "관련 정보 없음"
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"- [{i}] {r['title']}\n  요약: {r['snippet']}\n  링크: {r['url']}")
    return "\n".join(lines)


class GoalPlanner:
    """1) Google 검색 → 2) LLM 플랜 생성"""
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model_name

        base_dir = os.path.dirname(__file__)
        prompt_path = os.path.join(base_dir, "prompts", "plan.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

    def search_info(self, goal: str) -> str:
        """Google Programmable Search로 절차/가이드 검색."""
        print(f"\n[🔍 검색 시작] '{goal}' 관련 정보를 수집 중...\n")
        try:
            query = f"{goal} 절차 방법 과정 안내 공식"
            results = google_search(query, num=6, hl="ko", gl="kr", timeout=10)

            # ✅ 콘솔에 검색 결과 출력
            for i, r in enumerate(results, 1):
                print(f"[{i}] {r['title']}")
                print(f"    요약: {r['snippet']}")
                print(f"    링크: {r['url']}\n")

            print("-" * 60)
            print(f"총 {len(results)}개의 검색 결과를 가져왔습니다.\n")

            return format_search_context(results)
        except Exception as e:
            print(f"[⚠️ 검색 실패] {e}")
            return f"(검색 실패: {e})"

    def make_plan(self, goal: str) -> Dict:
        """검색 → 프롬프트 → LLM → JSON 파싱"""
        info = self.search_info(goal)
        prompt = (
            self.prompt_template
            .replace("{{goal}}", goal.strip())
            .replace("{{info}}", info.strip())
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "너는 목표 달성을 위한 실행 계획을 설계하는 전문가다. 출력은 반드시 JSON만 반환한다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            raw = resp.choices[0].message.content.strip()
        except Exception as e:
            return {"error": f"LLM 호출 실패: {e}"}

        # JSON 파싱
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(cleaned)
        except Exception as e:
            try:
                data = json.loads(cleaned.replace("\n", " ").replace("\r", " "))
            except Exception as e2:
                return {"error": f"JSON 파싱 실패: {e2}", "raw_output": raw}

        steps = data.get("steps") or []
        for i, s in enumerate(steps, 1):
            if "step" not in s:
                s["step"] = i
        rr = data.get("required_resources")
        if isinstance(rr, str):
            data["required_resources"] = [rr]
        elif rr is None:
            data["required_resources"] = []

        if not data.get("goal"):
            data["goal"] = goal

        return data


# --------------------------
# 시연용 main
# --------------------------
if __name__ == "__main__":
    planner = GoalPlanner()

    goals = [
        "주민등록등본을 발급받는다.",
        "운전면허증을 갱신한다.",
        "여권을 재발급받는다.",
        "건강검진을 예약한다.",
        "AI 포렌식 논문 초안을 작성한다."
    ]

    # 샘플 저장 경로
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    sample_dir = os.path.join(base_dir, "app", "sample", "planner")
    os.makedirs(sample_dir, exist_ok=True)

    for i, g in enumerate(goals, 1):
        print(f"\n=== 🎯 목표: {g} ===")
        plan = planner.make_plan(g)
        print("\n[📋 생성된 계획 결과]")
        print(json.dumps(plan, ensure_ascii=False, indent=2))

        # 결과 저장
        out_path = os.path.join(sample_dir, f"example{i}_plan.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        print(f"💾 계획이 {out_path} 에 저장되었습니다.\n")

        print("=" * 80)
