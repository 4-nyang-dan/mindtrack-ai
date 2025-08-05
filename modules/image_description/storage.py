import os
import faiss
import numpy as np
import pickle
from datetime import datetime
import json

class VectorDBStorage:
    def __init__(self, db_dir="./vectorstore", index_name="description_index", dim=1536):
        self.dim = dim
        self.db_dir = db_dir
        self.index_path = os.path.join(db_dir, f"{index_name}.faiss")
        self.meta_path = os.path.join(db_dir, f"{index_name}.meta")

        os.makedirs(db_dir, exist_ok=True)

        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []
        self._id_counter = 1

        if os.path.exists(self.index_path):
            self._load()
            self._id_counter = len(self.metadata) + 1

    def add_vector(self, embedding, metadata):
        """Add a vector and its metadata to the index."""
        vector = np.array(embedding).astype("float32").reshape(1, -1)
        self.index.add(vector)

        metadata["id"] = self._id_counter
        metadata["timestamp"] = datetime.now().isoformat()
        self._id_counter += 1
        self.metadata.append(metadata)

    def search_vector(self, query_embedding, top_k=3, exclude_id=None):
        """Search for top_k most similar vectors, excluding exclude_id if provided."""
        if self.index.ntotal == 0:
            return []

        query = np.array(query_embedding).astype("float32").reshape(1, -1)
        distances, indices = self.index.search(query, top_k + 1)  # extra for exclusion

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                if exclude_id and self.metadata[idx]["id"] == exclude_id:
                    continue
                results.append({
                    "metadata": self.metadata[idx],
                    "distance": float(dist)
                })
            if len(results) == top_k:
                break

        return results

    def get_recent(self, k=3):
        """Return the most recent k metadata entries."""
        return self.metadata[-k:]

    def save(self):
        """Save FAISS index and metadata to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def _load(self):
        """Load FAISS index and metadata from disk."""
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)


if __name__ == "__main__":
    # Directories for sample embeddings and descriptions
    embedding_dir = os.path.join(os.path.dirname(__file__), "../../app/sample/embedding")
    description_dir = os.path.join(os.path.dirname(__file__), "../../app/sample/description")

    if not os.path.exists(embedding_dir) or not os.path.exists(description_dir):
        raise FileNotFoundError("Embedding or description directory not found.")

    storage = VectorDBStorage()

    # Add embeddings and related descriptions to the DB
    for file in sorted(os.listdir(embedding_dir)):
        if file.endswith(".json"):
            embedding_path = os.path.join(embedding_dir, file)
            desc_file = file.replace("embedding", "description").replace(".json", ".txt")
            desc_path = os.path.join(description_dir, desc_file)

            if not os.path.exists(desc_path):
                print(f"Warning: No matching description for {file}")
                continue

            # Load embedding
            with open(embedding_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                embedding = data["embedding"]

            # Load description text
            with open(desc_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            # Add to DB
            storage.add_vector(embedding, {"file": desc_file, "text": text})
            print(f"Added {file} to vector DB.")

    # Save DB
    storage.save()
    print("\n== All embeddings added and saved to FAISS DB. ==")

    # Search test
    if len(storage.metadata) > 0:
        # Use first embedding for query
        first_meta = storage.metadata[0]
        first_id = first_meta["id"]
        first_embedding_path = os.path.join(embedding_dir, sorted(os.listdir(embedding_dir))[0])
        with open(first_embedding_path, "r", encoding="utf-8") as f:
            first_embedding = json.load(f)["embedding"]

        results = storage.search_vector(first_embedding, top_k=3, exclude_id=first_id)
        print("\n--- Search Results (excluding self) ---")
        for r in results:
            print(f"ID: {r['metadata']['id']}, Distance: {r['distance']:.4f}, Text: {r['metadata']['text'][:40]}...")

    # Recent entries test
    print("\n--- Recent Entries ---")
    recent_items = storage.get_recent(k=3)
    for item in recent_items:
        print(f"ID: {item['id']} | File: {item['file']} | Timestamp: {item['timestamp']}")
