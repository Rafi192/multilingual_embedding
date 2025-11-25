from sentence_transformers import SentenceTransformer
import numpy as np
# from embedder_bge_m3 import 

class XLMR_Multilingual_Embedder:
    def __init__(self):
        # Multilingual mpnet-style embedding model (banked from XLM-R)
        self.model = SentenceTransformer(
            "sentence-transformers/distiluse-base-multilingual-cased-v2"
        )

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        vectors = self.model.encode(texts, normalize_embeddings=True)
        print(f"Embedded {len(texts)} texts using XLM-R Multilingual model.")
        print(f"First vector (truncated): {vectors[0][:5]}...")
        print("Vector shape:", np.array(vectors).shape)

        return np.array(vectors).tolist()

obj_xlmr_multilingual_embedder = XLMR_Multilingual_Embedder()
print("Testing XLM-R Multilingual Embedder with multilingual sentences:")

v_en = obj_xlmr_multilingual_embedder.embed("I love cats!")                            # English
v_bn = obj_xlmr_multilingual_embedder.embed("আমি বিড়ালকে ভালোবাসি!")                  # Bangla
v_es = obj_xlmr_multilingual_embedder.embed("¡Me encantan los gatos!")                 # Spanish
v_fr = obj_xlmr_multilingual_embedder.embed("J'adore les chats !")                    # French
v_hi = obj_xlmr_multilingual_embedder.embed("मुझे बिल्लियाँ बहुत पसंद हैं!")             # Hindi
v_ar = obj_xlmr_multilingual_embedder.embed("أنا أحب القطط!")                           # Arabic
v_ja = obj_xlmr_multilingual_embedder.embed("私は猫が大好きです！")                       # Japanese
v_zh = obj_xlmr_multilingual_embedder.embed("我爱猫！")

v_de = obj_xlmr_multilingual_embedder.embed("Ich liebe Katzen!")  # German
v_ru = obj_xlmr_multilingual_embedder.embed("Я люблю кошек!")  # Russian

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

def cosine(a, b): return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

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

for lang_pair, vec1, vec2 in pairs:
    sim = cosine(vec1, vec2)
    print(f"{lang_pair} similarity: {sim:.4f}")

print("-------pairs type-----------\n",type(pairs),flush=True)