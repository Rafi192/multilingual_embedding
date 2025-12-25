from sentence_transformers import SentenceTransformer
import numpy as np
from time import time


class XLMR_Multilingual_Embedder:
    def __init__(self):
        self.model = SentenceTransformer(
            "sentence-transformers/distiluse-base-multilingual-cased-v2"
        )

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, normalize_embeddings=True)


def timed_embedding(embedder, text, lang_label):
    start = time()
    vec = embedder.embed(text)
    end = time()

    elapsed = end - start
    print(
        f"{lang_label:>8} | {elapsed:.6f} sec ({elapsed*1000:.2f} ms)"
    )
    return vec[0]


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ----------------- RUN -----------------

embedder = XLMR_Multilingual_Embedder()

print("=== XLM-R Multilingual Embedding Time ===")

v_en = timed_embedding(embedder, "I love cats!", "EN")
v_bn = timed_embedding(embedder, "আমি বিড়ালকে ভালোবাসি!", "BN")
v_es = timed_embedding(embedder, "¡Me encantan los gatos!", "ES")
v_fr = timed_embedding(embedder, "J'adore les chats !", "FR")
v_hi = timed_embedding(embedder, "मुझे बिल्लियाँ बहुत पसंद हैं!", "HI")
v_ar = timed_embedding(embedder, "أنا أحب القطط!", "AR")
v_ja = timed_embedding(embedder, "私は猫が大好きです！", "JA")
v_zh = timed_embedding(embedder, "我爱猫！", "ZH")
v_de = timed_embedding(embedder, "Ich liebe Katzen!", "DE")
v_ru = timed_embedding(embedder, "Я люблю кошек!", "RU")


# ----------------- SIMILARITY -----------------

pairs = [
    ("EN–BN", v_en, v_bn),
    ("EN–ES", v_en, v_es),
    ("EN–FR", v_en, v_fr),
    ("EN–HI", v_en, v_hi),
    ("EN–AR", v_en, v_ar),
    ("EN–JA", v_en, v_ja),
    ("EN–ZH", v_en, v_zh),
    ("EN–DE", v_en, v_de),
    ("EN–RU", v_en, v_ru),
]

print("\n=== Similarity Scores ===")
for name, a, b in pairs:
    print(f"{name}: {cosine(a, b):.4f}")
