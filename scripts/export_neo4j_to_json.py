"""Export the local Neo4j graph to a portable JSON file.

Reads from local Docker Neo4j (bolt://localhost:7687) and writes:
  data/synthetic/graph_seed.json

This file is the canonical seed for re-creating the graph in any Neo4j instance
(local, AuraDB, etc.).

Schema captured:
  - All nodes with their labels and properties
  - All relationships with their types, start/end node ids, and properties

Run: python scripts/export_neo4j_to_json.py
"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "assemblyrag2026")

ROOT = Path(__file__).resolve().parent.parent
OUT_FILE = ROOT / "data" / "synthetic" / "graph_seed.json"


def main():
    print("[*] Connecting to " + URI)
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

    with driver.session() as session:
        # Pull all nodes
        print("[*] Exporting nodes ...")
        nodes_result = session.run("MATCH (n) RETURN id(n) AS id, labels(n) AS labels, properties(n) AS props")
        nodes = []
        for rec in nodes_result:
            nodes.append({
                "internal_id": rec["id"],
                "labels": rec["labels"],
                "props": dict(rec["props"]),
            })
        print("    " + str(len(nodes)) + " nodes")

        # Pull all relationships
        print("[*] Exporting relationships ...")
        rels_result = session.run(
            "MATCH (a)-[r]->(b) RETURN id(a) AS start_id, id(b) AS end_id, type(r) AS type, properties(r) AS props"
        )
        rels = []
        for rec in rels_result:
            rels.append({
                "start_internal_id": rec["start_id"],
                "end_internal_id": rec["end_id"],
                "type": rec["type"],
                "props": dict(rec["props"]),
            })
        print("    " + str(len(rels)) + " relationships")

    driver.close()

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({"nodes": nodes, "relationships": rels}, indent=2))
    print("[OK] Wrote " + str(OUT_FILE))
    print("    Size: " + str(OUT_FILE.stat().st_size // 1024) + " KB")


if __name__ == "__main__":
    main()
