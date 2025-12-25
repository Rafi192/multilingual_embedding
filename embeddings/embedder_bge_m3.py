from sentence_transformers import SentenceTransformer
import numpy as np
from time import perf_counter

class BGE_M3_Embedder:
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-m3")

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, normalize_embeddings=True)

embedder = BGE_M3_Embedder()

# 🔥 Warm-up (DO NOT MEASURE)
embedder.embed("warm up")

def benchmark(text, runs=10):
    times = []
    for _ in range(runs):
        start = perf_counter()
        embedder.embed(text)
        end = perf_counter()
        times.append(end - start)
    return np.mean(times), np.std(times)

langs = {
    "EN": "I love cats!",
    "BN": "আমি বিড়ালকে ভালোবাসি!",
    "ES": "¡Me encantan los gatos!"
}

for lang, text in langs.items():
    avg, std = benchmark(text)
    print(
        f"{lang}: {avg:.6f} sec | {avg*1000:.2f} ms "
        f"(± {std*1000:.2f} ms)"
    )
