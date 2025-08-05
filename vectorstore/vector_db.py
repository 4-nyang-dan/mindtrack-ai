import faiss
import numpy as np
import os

class VectorDB:
    def __init__(self, dim: int, index_path="vector_index.faiss"):
        self.dim = dim
        self.index_path = index_path
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = faiss.IndexFlatL2(dim)

    def add_embedding(self, vectors: list[np.ndarray]):
        self.index.add(np.array(vectors, dtype=np.float32))

    def search_embedding(self, query_vector: np.ndarray, top_k: int = 5):
        query_vector = np.array([query_vector], dtype=np.float32)
        distances, indices = self.index.search(query_vector, top_k)
        return indices[0], distances[0]

    def save(self):
        faiss.write_index(self.index, self.index_path)
