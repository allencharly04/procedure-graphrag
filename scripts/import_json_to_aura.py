"""Import a graph_seed.json file into a target Neo4j instance.

Reads credentials from streamlit_app/.streamlit/secrets.toml.

Run: python scripts/import_json_to_aura.py
"""
import json
import sys
from pathlib import Path

# tomllib for Python 3.11+
try:
    import tomllib
except ImportError:
    import tomli as tomllib

from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent.parent
SECRETS = ROOT / "streamlit_app" / ".streamlit" / "secrets.toml"
SEED_FILE = ROOT / "data" / "synthetic" / "graph_seed.json"


def main():
    if not SECRETS.exists():
        print("ERROR: secrets file not found at " + str(SECRETS))
        sys.exit(1)
    if not SEED_FILE.exists():
        print("ERROR: seed file not found at " + str(SEED_FILE))
        print("       Run scripts/export_neo4j_to_json.py first.")
        sys.exit(1)

    with open(SECRETS, "rb") as f:
        s = tomllib.load(f)
    uri = s["NEO4J_URI"]
    user = s["NEO4J_USERNAME"]
    pw = s["NEO4J_PASSWORD"]
    db = s.get("NEO4J_DATABASE", "neo4j")

    print("[*] Loading seed file: " + str(SEED_FILE))
    seed = json.loads(SEED_FILE.read_text())
    nodes = seed["nodes"]
    rels = seed["relationships"]
    print("    " + str(len(nodes)) + " nodes, " + str(len(rels)) + " relationships")

    print("[*] Connecting to " + uri + " (db=" + db + ")")
    driver = GraphDatabase.driver(uri, auth=(user, pw))

    with driver.session(database=db) as session:
        print("[*] Wiping target database ...")
        session.run("MATCH (n) DETACH DELETE n")
        # Drop indexes/constraints from any prior load
        try:
            for rec in session.run("SHOW CONSTRAINTS"):
                name = rec.get("name")
                if name:
                    session.run("DROP CONSTRAINT " + name)
        except Exception:
            pass
        try:
            for rec in session.run("SHOW INDEXES"):
                name = rec.get("name")
                if name and rec.get("type") != "LOOKUP":
                    session.run("DROP INDEX " + name)
        except Exception:
            pass
        print("    cleared")

        # Insert nodes with __seed_id mapped to internal_id
        print("[*] Loading " + str(len(nodes)) + " nodes ...")
        # Group nodes by label for cleaner Cypher (single-label assumption per row)
        for node in nodes:
            label_str = ":".join(node["labels"])
            props = dict(node["props"])
            props["__seed_id"] = node["internal_id"]
            cypher = "CREATE (n:" + label_str + " $props)"
            session.run(cypher, props=props)
        print("    nodes loaded")

        # Insert relationships using __seed_id lookup
        print("[*] Loading " + str(len(rels)) + " relationships ...")
        for rel in rels:
            cypher = (
                "MATCH (a {__seed_id: $start_id}), (b {__seed_id: $end_id}) "
                "CREATE (a)-[r:" + rel["type"] + "]->(b) "
                "SET r = $props"
            )
            session.run(
                cypher,
                start_id=rel["start_internal_id"],
                end_id=rel["end_internal_id"],
                props=rel["props"],
            )
        print("    relationships loaded")

        # Drop the temp __seed_id property
        print("[*] Removing temporary __seed_id property ...")
        session.run("MATCH (n) REMOVE n.__seed_id")

        # Recreate the procedure_id index/constraint that the local DB had
        print("[*] Recreating constraint on Procedure.id ...")
        try:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Procedure) REQUIRE p.id IS UNIQUE")
        except Exception as e:
            print("    (could not create constraint: " + str(e)[:80] + ")")

        # Verify
        result = session.run("MATCH (n) RETURN count(n) AS c").single()
        print("[*] AuraDB now has " + str(result["c"]) + " nodes")
        result = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()
        print("[*] AuraDB now has " + str(result["c"]) + " relationships")

    driver.close()
    print("[OK] Migration complete")


if __name__ == "__main__":
    main()
