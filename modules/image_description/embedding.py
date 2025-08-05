import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class EmbeddingGenerator:
    def __init__(self, model_name="text-embedding-3-small"):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name

    def generate_embedding(self, text: str):
        """Generate embedding for a given text using OpenAI embeddings API."""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        return response.data[0].embedding


if __name__ == "__main__":
    # Path to the description text generated earlier
    description_path = os.path.join(os.path.dirname(__file__), "../../app/sample/description4.txt")
    if not os.path.exists(description_path):
        raise FileNotFoundError(f"Description file not found at {description_path}")

    # Read text from file
    with open(description_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    print(f"\n--- Loaded Text ---\n{text}\n")

    # Generate embedding
    generator = EmbeddingGenerator()
    embedding = generator.generate_embedding(text)

    # Print result
    print(f"Embedding length: {len(embedding)}")
    print(f"First 10 values: {embedding[:10]}")

    # Save embedding to JSON file
    output_dir = os.path.join(os.path.dirname(__file__), "../../app/sample")
    os.makedirs(output_dir, exist_ok=True)
    embedding_path = os.path.join(output_dir, "embedding4.json")

    with open(embedding_path, "w", encoding="utf-8") as f:
        json.dump({"embedding": embedding}, f, ensure_ascii=False, indent=2)

    print(f"\nEmbedding saved to {embedding_path}")
