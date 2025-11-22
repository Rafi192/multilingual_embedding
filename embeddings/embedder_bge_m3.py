from sentence_transformers import SentenceTransformer
import numpy as np


class BGE_M3_Embedder:
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-m3")


    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return np.array(vectors).tolist()