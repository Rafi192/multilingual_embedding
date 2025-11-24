from sentence_transformers import SentenceTransformer
import numpy as np


class GTE_Multilingual_Embedder:
    def __init__(self):
        self.model = SentenceTransformer("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
            
        vectors = self.model.encode(texts, normalize_embeddings=True)
        print(f"Embedded {len(texts)} texts using GTE-Multilingual model.")
        print(f"First vector (truncated): {vectors[0][:5]}...")
        print("Vector shape:", np.array(vectors).shape)
        return np.array(vectors).tolist()


obj_gte_multilingual_embedder = GTE_Multilingual_Embedder()

print("Testing GTE-Multilingual Embedder with multilingual sentences:")

v_en = obj_gte_multilingual_embedder.embed("I love cats!")                            # English
v_bn = obj_gte_multilingual_embedder.embed("আমি বিড়ালকে ভালোবাসি!")                  # Bangla
v_es = obj_gte_multilingual_embedder.embed("¡Me encantan los gatos!")                 # Spanish
v_fr = obj_gte_multilingual_embedder.embed("J'adore les chats !")                    # French
v_hi = obj_gte_multilingual_embedder.embed("मुझे बिल्लियाँ बहुत पसंद हैं!")             # Hindi
v_ar = obj_gte_multilingual_embedder.embed("أنا أحب القطط!")                           # Arabic
v_ja = obj_gte_multilingual_embedder.embed("私は猫が大好きです！")                       # Japanese
v_zh = obj_gte_multilingual_embedder.embed("我爱猫！")                                  # Chinese (Simplified)
v_de = obj_gte_multilingual_embedder.embed("Ich liebe Katzen!")  # German
v_ru = obj_gte_multilingual_embedder.embed("Я люблю кошек!")                          # Russian


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

# def cosine(a, b): return np.dot(a, b)

def cosine(a, b): return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# print("EN–BN similarity:", cosine(v_en, v_bn))
# print("EN–ES similarity:", cosine(v_en, v_es))
# print("BN–ES similarity:", cosine(v_bn, v_es))

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
    print(f"{name}: {cosine(np.array(a), np.array(b)):.4f}")
