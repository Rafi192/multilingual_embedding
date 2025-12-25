from sentence_transformers import SentenceTransformer
import numpy as np
from time import time


class GTE_Multilingual_Embedder:
    def __init__(self):
        self.model = SentenceTransformer(
            "Alibaba-NLP/gte-multilingual-base",
            trust_remote_code=True
        )

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        vectors = self.model.encode(texts, normalize_embeddings=True)

        print(f"Embedded {len(texts)} texts using GTE-Multilingual model.")
        print(f"First vector (truncated): {vectors[0][:5]}...")
        print("Vector shape:", np.array(vectors).shape)

        return np.array(vectors).tolist()


def timed_embedding(embedder, text, lang_label):
    start = time()
    print(f"start time for {lang_label} embedding: {start}")

    vectors = embedder.embed(text)

    end = time()
    print(f"end time for {lang_label} embedding: {end}")

    elapsed = end - start
    print(
        f"{lang_label} embedding time: "
        f"{elapsed:.6f} sec ({elapsed*1000:.2f} ms)"
    )
    print("-" * 60)

    return vectors


# ----------------- RUN -----------------

obj_gte_multilingual_embedder = GTE_Multilingual_Embedder()

print("Testing GTE-Multilingual Embedder with multilingual sentences:\n")

v_en = timed_embedding(obj_gte_multilingual_embedder, "I love cats!", "english")
v_bn = timed_embedding(obj_gte_multilingual_embedder, "আমি বিড়ালকে ভালোবাসি!", "bengali")
v_es = timed_embedding(obj_gte_multilingual_embedder, "¡Me encantan los gatos!", "spanish")
v_fr = timed_embedding(obj_gte_multilingual_embedder, "J'adore les chats !", "french")
v_hi = timed_embedding(obj_gte_multilingual_embedder, "मुझे बिल्लियाँ बहुत पसंद हैं!", "hindi")
v_ar = timed_embedding(obj_gte_multilingual_embedder, "أنا أحب القطط!", "arabic")
v_ja = timed_embedding(obj_gte_multilingual_embedder, "私は猫が大好きです！", "japanese")
v_zh = timed_embedding(obj_gte_multilingual_embedder, "我爱猫！", "chinese")
v_de = timed_embedding(obj_gte_multilingual_embedder, "Ich liebe Katzen!", "german")
v_ru = timed_embedding(obj_gte_multilingual_embedder, "Я люблю кошек!", "russian")


# ----------------- SIMILARITY -----------------

v_en = np.array(v_en[0])
v_bn = np.array(v_bn[0])
v_es = np.array(v_es[0])
v_fr = np.array(v_fr[0])
v_hi = np.array(v_hi[0])
v_ar = np.array(v_ar[0])
v_ja = np.array(v_ja[0])
v_zh = np.array(v_zh[0])
v_de = np.array(v_de[0])
v_ru = np.array(v_ru[0])


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


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
