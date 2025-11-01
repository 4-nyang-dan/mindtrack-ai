# modules/image_description/description.py
import os
import base64
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class ImageDescription:
    def __init__(self, model_name="gpt-4.1-mini"):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name
        # Load prompt from file
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "description.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

    def _encode_image(self, image_path: str) -> str:
        """Encode image to Base64 for API"""
        with open(image_path, "rb") as img:
            return base64.b64encode(img.read()).decode("utf-8")

    def generate_description(self, image_path: str):
        """Send image and prompt to GPT model and return the raw response object"""
        image_base64 = self._encode_image(image_path)
        response = self.client.responses.create(
            model=self.model_name,
            temperature=0.3,
            max_output_tokens=300,
            top_p=0.5,
            input=[
                {"role": "system", "content": self.prompt_template},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "이미지를 분석하고 JSON을 반환하세요."},
                        {"type": "input_image", "image_url": f"data:image/png;base64,{image_base64}"}
                    ]
                }
            ]
        )
        return response


if __name__ == "__main__":
    # Image path
    image_path = os.path.join(os.path.dirname(__file__), "../../app/sample/image/example5.png")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Output file path
    output_path = os.path.join(os.path.dirname(__file__), "../../app/sample/description5.txt")

    generator = ImageDescription()
    response = generator.generate_description(image_path)

    # 출력
    print("\n--- Raw Output Text ---")
    print(response.output_text)

    # 파일 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(response.output_text)

    print(f"\nSaved description to {output_path}")