"""Auto-fix expected_answer fields by trusting Cypher over Claude.

Re-runs every draft question's ground_truth_cypher against Neo4j and replaces
the expected_answer with the actual database result. This is principled because
the Cypher query IS the formal expression of the question - the LLM-generated
expected_answer was a hand-computation that we have no reason to trust over
the database.

Drops any question whose Cypher errors out.

Outputs:
  data/eval_queries.jsonl - the gold eval set
"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DRAFT_FILE = ROOT / "data" / "eval_queries_draft.jsonl"
OUT_FILE = ROOT / "data" / "eval_queries.jsonl"

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "assemblyrag2026")


def run_query(session, cypher):
    result = session.run(cypher)
    rows = []
    for record in result:
        if len(record.keys()) == 1:
            rows.append(record[record.keys()[0]])
        else:
            rows.append(dict(record))
    return rows


def normalize_answer(value, mode):
    """Convert Neo4j Python types into JSON-serializable answer format."""
    # Unwrap single value
    if isinstance(value, list) and len(value) == 1 and not isinstance(value[0], dict):
        value = value[0]

    if mode == "set":
        if not isinstance(value, list):
            value = [value] if value is not None else []
        return sorted(str(v).strip() for v in value)
    elif mode in ("exact", "numeric_within_tolerance"):
        if isinstance(value, list) and len(value) == 0:
            return None
        # Convert numerics to int when whole, float otherwise
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    return value


def main():
    if not DRAFT_FILE.exists():
        print("[FAIL] Draft file not found")
        sys.exit(1)

    with open(DRAFT_FILE) as f:
        questions = [json.loads(line) for line in f]
    print("[*] Loaded " + str(len(questions)) + " draft questions")

    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    fixed = []
    cypher_errors = 0
    empty_results = 0
    changed = 0

    try:
        with driver.session() as session:
            for q in questions:
                cypher = q.get("ground_truth_cypher", "")
                mode = q.get("expected_match_mode", "exact")
                old_answer = q.get("expected_answer")

                try:
                    rows = run_query(session, cypher)
                except Exception as e:
                    print("  [DROP-CYPHER-ERR] " + q["id"] + ": " + str(e)[:80])
                    cypher_errors += 1
                    continue

                if not rows:
                    print("  [DROP-EMPTY] " + q["id"] + ": cypher returned no rows")
                    empty_results += 1
                    continue

                new_answer = normalize_answer(rows, mode)

                # Track if this is a change
                if str(new_answer).strip() != str(old_answer).strip():
                    changed += 1

                q["expected_answer"] = new_answer
                # Add a small marker showing we trust Cypher here
                q["answer_source"] = "cypher_against_neo4j"
                fixed.append(q)

    finally:
        driver.close()

    print()
    print("[*] Total questions: " + str(len(questions)))
    print("[*] Kept: " + str(len(fixed)))
    print("[*] Dropped (cypher error): " + str(cypher_errors))
    print("[*] Dropped (empty result): " + str(empty_results))
    print("[*] Answer changed: " + str(changed) + " (out of " + str(len(fixed)) + ")")

    by_cat = {}
    for q in fixed:
        by_cat[q["category"]] = by_cat.get(q["category"], 0) + 1
    print("[*] Per-category counts: " + str(by_cat))

    with open(OUT_FILE, "w") as f:
        for q in fixed:
            f.write(json.dumps(q) + "\n")
    print("[OK] Wrote " + str(OUT_FILE))


if __name__ == "__main__":
    main()
