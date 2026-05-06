"""Build a Chroma vector index over Step nodes.

Each Step becomes one document in Chroma:
  - id: step uid (e.g., "PRC-001:7")
  - text: enriched form "[procedure_title | equipment_type] Step N: instruction"
  - metadata: procedure_id, procedure_title, step_number, equipment_type,
              hold_point, tool_ids (list), component_ids (list)

Embedded with sentence-transformers/all-MiniLM-L6-v2 (384-dim).
Stored at data/chroma/. Idempotent: rerun replaces the collection.
"""
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION = "steps"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "assemblyrag2026")


def fetch_steps_with_context(driver):
    """Fetch every step with the context needed for retrieval."""
    cypher = """
    MATCH (p:Procedure)-[:HAS_STEP]->(s:Step)
    OPTIONAL MATCH (s)-[:USES_TOOL]->(t:Tool)
    OPTIONAL MATCH (s)-[:USES_COMPONENT]->(c:Component)
    WITH p, s,
         collect(DISTINCT t.id) AS tool_ids,
         collect(DISTINCT c.id) AS comp_ids
    RETURN s.uid AS uid,
           s.step_number AS step_number,
           s.instruction AS instruction,
           s.hold_point AS hold_point,
           p.id AS procedure_id,
           p.title AS procedure_title,
           p.equipment_type AS equipment_type,
           tool_ids,
           comp_ids
    ORDER BY p.id, s.step_number
    """
    with driver.session() as session:
        result = session.run(cypher)
        return [dict(record) for record in result]


def build_document(step):
    """Convert a step record into the enriched text we embed."""
    header = "[" + step["procedure_title"] + " | " + step["equipment_type"] + "]"
    body = "Step " + str(step["step_number"]) + ": " + step["instruction"]
    return header + " " + body


def main():
    print("[*] Connecting to Neo4j ...")
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    driver.verify_connectivity()

    print("[*] Fetching steps with context from Neo4j ...")
    t0 = time.perf_counter()
    steps = fetch_steps_with_context(driver)
    driver.close()
    print("    fetched " + str(len(steps)) + " steps in {:.2f}s".format(time.perf_counter() - t0))

    print("[*] Loading embedding model: " + EMBED_MODEL)
    t0 = time.perf_counter()
    model = SentenceTransformer(EMBED_MODEL)
    print("    loaded in {:.2f}s, dim={}".format(
        time.perf_counter() - t0, model.get_sentence_embedding_dimension()))

    print("[*] Building documents and embedding ...")
    docs = [build_document(s) for s in steps]
    ids = [s["uid"] for s in steps]
    metadatas = []
    for s in steps:
        metadatas.append({
            "procedure_id": s["procedure_id"],
            "procedure_title": s["procedure_title"],
            "step_number": s["step_number"],
            "equipment_type": s["equipment_type"],
            "hold_point": s["hold_point"],
            "tool_ids": ",".join(s["tool_ids"]) if s["tool_ids"] else "",
            "component_ids": ",".join(s["comp_ids"]) if s["comp_ids"] else "",
        })

    t0 = time.perf_counter()
    embeddings = model.encode(docs, normalize_embeddings=True, show_progress_bar=True)
    embed_elapsed = time.perf_counter() - t0
    print("    embedded {} docs in {:.2f}s ({:.1f} docs/sec)".format(
        len(docs), embed_elapsed, len(docs) / embed_elapsed))

    print("[*] Writing to Chroma at " + str(CHROMA_DIR))
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Wipe and recreate the collection (idempotent)
    try:
        client.delete_collection(COLLECTION)
        print("    deleted existing collection")
    except Exception:
        pass

    coll = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    coll.add(
        ids=ids,
        documents=docs,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    print("    added {} entries".format(coll.count()))

    # Sanity check: query for something pump-related, see top 3 hits
    print()
    print("[*] Sanity query: 'how do I replace a pump bearing'")
    test_emb = model.encode(["how do I replace a pump bearing"], normalize_embeddings=True).tolist()
    hits = coll.query(query_embeddings=test_emb, n_results=3, include=["documents", "metadatas", "distances"])
    for i, (doc, meta, dist) in enumerate(zip(hits["documents"][0], hits["metadatas"][0], hits["distances"][0])):
        print("  {}. (dist={:.3f}) {}".format(i + 1, dist, doc[:120]))
        print("      meta: proc={}, equipment={}".format(meta["procedure_id"], meta["equipment_type"]))

    print()
    print("[OK] Vector index built")


if __name__ == "__main__":
    main()
