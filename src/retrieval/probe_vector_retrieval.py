"""Interactive vector retrieval probe.

Ad-hoc tool for inspecting retrieval quality. Not part of the production
pipeline. Edit the QUERIES list to try different prompts.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()
os.environ["ANONYMIZED_TELEMETRY"] = "False"

ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION = "steps"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Probe queries spanning easy retrieval -> hard retrieval -> things we expect to fail
QUERIES = [
    "how do I replace a pump bearing",                        # easy semantic match
    "what torque should I use for M20 bolts",                 # numeric/fastener query
    "steps requiring electrical certification",                # cert-related
    "how do I align a motor to its driven equipment",         # specific procedure title
    "what are the safety hold points in a heat exchanger",    # multi-concept
    "which procedures need a chain hoist",                    # specific tool reference
    "what defects can occur during compressor service",       # defect-mode query
    "PRC-014 step 3",                                          # graph-internal id reference (probably won't work well in vector)
]


def main():
    print("[*] Loading embedding model ...")
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    coll = client.get_collection(COLLECTION)
    print("[*] Collection has " + str(coll.count()) + " entries")

    for query in QUERIES:
        print()
        print("Q: " + query)
        emb = model.encode([query], normalize_embeddings=True).tolist()
        hits = coll.query(query_embeddings=emb, n_results=3,
                          include=["documents", "metadatas", "distances"])
        for i in range(len(hits["documents"][0])):
            doc = hits["documents"][0][i]
            meta = hits["metadatas"][0][i]
            dist = hits["distances"][0][i]
            short = doc[:110].replace("\n", " ")
            print("  [{:.3f}] {} ({})".format(dist, short, meta["procedure_id"]))


if __name__ == "__main__":
    main()
