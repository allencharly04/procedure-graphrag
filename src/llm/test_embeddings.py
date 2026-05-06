"""Smoke test: load embedding model and embed a short text."""
import time

from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    print("[*] Loading", MODEL, "(first run downloads ~80MB)")
    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL)
    print("[*] Model loaded in {:.2f}s".format(time.perf_counter() - t0))
    print("[*] Embedding dim:", model.get_sentence_embedding_dimension())

    texts = [
        "Apply Hi-Lok rivet to wing skin substrate.",
        "Torque value for procedure WS-014 step 7 is 45 Nm.",
        "The cat sat on the mat.",
    ]
    t0 = time.perf_counter()
    emb = model.encode(texts, normalize_embeddings=True)
    print("[*] Embedded {} texts in {:.3f}s".format(len(texts), time.perf_counter() - t0))
    print("[*] Embedding shape:", emb.shape)

    sim_aerospace = float(emb[0] @ emb[1])
    sim_unrelated = float(emb[0] @ emb[2])
    print("[*] sim(aerospace, aerospace) = {:.3f}".format(sim_aerospace))
    print("[*] sim(aerospace, unrelated) = {:.3f}".format(sim_unrelated))
    assert sim_aerospace > sim_unrelated, "Embedding sanity check failed"
    print("[OK] Embedding smoke test passed")


if __name__ == "__main__":
    main()
