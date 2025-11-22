from sentence_transformers import SentenceTransformer
import numpy as np


class GTE_Multilingual_Embedder:
    def __init__(self):
        self.model = SentenceTransformer("Alibaba-NLP/gte-multilingual-base")


def embed(self, texts):
    if isinstance(texts, str):
        texts = [texts]
    vectors = self.model.encode(texts, normalize_embeddings=True)
    return np.array(vectors).tolist()