"""Load synthetic catalog + procedures into Neo4j.

Idempotent: safe to rerun. Uses MERGE everywhere so re-running won't create
duplicates. Recreates constraints/indexes on every run (cheap).

Order:
  1. Constraints + indexes
  2. Nodes (catalog entities, then procedures, steps, torque specs)
  3. Relationships
"""
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic"
CATALOG_FILE = DATA_DIR / "catalog.json"
PROCEDURES_FILE = DATA_DIR / "procedures.json"
TORQUE_SPECS_FILE = DATA_DIR / "torque_specs.json"

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "assemblyrag2026")


CONSTRAINTS = [
    "CREATE CONSTRAINT proc_id IF NOT EXISTS FOR (p:Procedure) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT tool_id IF NOT EXISTS FOR (t:Tool) REQUIRE t.id IS UNIQUE",
    "CREATE CONSTRAINT comp_id IF NOT EXISTS FOR (c:Component) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT cert_id IF NOT EXISTS FOR (c:Certification) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT defect_id IF NOT EXISTS FOR (d:Defect) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT spec_id IF NOT EXISTS FOR (s:TorqueSpec) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT step_uid IF NOT EXISTS FOR (s:Step) REQUIRE s.uid IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX proc_equipment IF NOT EXISTS FOR (p:Procedure) ON (p.equipment_type)",
    "CREATE INDEX proc_criticality IF NOT EXISTS FOR (p:Procedure) ON (p.criticality)",
    "CREATE INDEX comp_category IF NOT EXISTS FOR (c:Component) ON (c.category)",
    "CREATE INDEX cert_level IF NOT EXISTS FOR (c:Certification) ON (c.level)",
    "CREATE INDEX defect_severity IF NOT EXISTS FOR (d:Defect) ON (d.severity)",
]


def setup_schema(session):
    print("[*] Setting up constraints + indexes ...")
    for stmt in CONSTRAINTS + INDEXES:
        session.run(stmt)
    print("    " + str(len(CONSTRAINTS)) + " constraints, " + str(len(INDEXES)) + " indexes")


def wipe_data(session):
    print("[*] Wiping existing graph data and constraints ...")
    session.run("MATCH (n) DETACH DELETE n")
    # Drop all existing constraints so setup_schema can recreate them cleanly
    result = session.run("SHOW CONSTRAINTS YIELD name")
    names = [record["name"] for record in result]
    for name in names:
        session.run("DROP CONSTRAINT " + name + " IF EXISTS")
    if names:
        print("    dropped " + str(len(names)) + " existing constraints")


def load_catalog(session, catalog):
    print("[*] Loading catalog ...")
    session.run(
        """
        UNWIND $rows AS row
        MERGE (t:Tool {id: row.id})
        SET t.name = row.name, t.type = row.type,
            t.calibration_required = row.calibration_required
        """,
        rows=catalog["tools"],
    )
    print("    " + str(len(catalog["tools"])) + " tools")

    session.run(
        """
        UNWIND $rows AS row
        MERGE (c:Component {id: row.id})
        SET c.name = row.name, c.category = row.category, c.material = row.material
        """,
        rows=catalog["components"],
    )
    print("    " + str(len(catalog["components"])) + " components")

    session.run(
        """
        UNWIND $rows AS row
        MERGE (c:Certification {id: row.id})
        SET c.name = row.name, c.level = row.level, c.scope = row.scope
        """,
        rows=catalog["certifications"],
    )
    print("    " + str(len(catalog["certifications"])) + " certifications")

    session.run(
        """
        UNWIND $rows AS row
        MERGE (d:Defect {id: row.id})
        SET d.name = row.name, d.severity = row.severity, d.escalation = row.escalation
        """,
        rows=catalog["defects"],
    )
    print("    " + str(len(catalog["defects"])) + " defects")


def load_torque_specs(session, specs):
    print("[*] Loading torque specs ...")
    session.run(
        """
        UNWIND $rows AS row
        MERGE (s:TorqueSpec {id: row.id})
        SET s.target_nm = row.target_nm, s.tolerance_nm = row.tolerance_nm,
            s.pattern = row.pattern, s.conditions = row.conditions
        """,
        rows=specs,
    )
    # Link torque specs to the fastener components they apply to
    session.run(
        """
        UNWIND $rows AS row
        MATCH (s:TorqueSpec {id: row.id})
        MATCH (c:Component {id: row.applies_to_component})
        MERGE (s)-[:APPLIES_TO]->(c)
        """,
        rows=specs,
    )
    print("    " + str(len(specs)) + " torque specs (linked to fasteners)")


def load_procedures(session, procs):
    print("[*] Loading procedures + steps ...")

    # 1. Procedure nodes
    proc_rows = [
        {
            "id": p["id"],
            "title": p["title"],
            "equipment_type": p["equipment_type"],
            "criticality": p["criticality"],
            "est_duration_min": p["est_duration_min"],
        }
        for p in procs
    ]
    session.run(
        """
        UNWIND $rows AS row
        MERGE (p:Procedure {id: row.id})
        SET p.title = row.title, p.equipment_type = row.equipment_type,
            p.criticality = row.criticality, p.est_duration_min = row.est_duration_min
        """,
        rows=proc_rows,
    )
    print("    " + str(len(proc_rows)) + " procedures")

    # 2. Step nodes (uid = procedure_id + ':' + step_number for uniqueness)
    step_rows = []
    for p in procs:
        for s in p["steps"]:
            step_rows.append({
                "uid": p["id"] + ":" + str(s["step_number"]),
                "step_number": s["step_number"],
                "instruction": s["instruction"],
                "est_duration_sec": s["est_duration_sec"],
                "hold_point": s["hold_point"],
                "proc_id": p["id"],
            })
    session.run(
        """
        UNWIND $rows AS row
        MERGE (s:Step {uid: row.uid})
        SET s.step_number = row.step_number, s.instruction = row.instruction,
            s.est_duration_sec = row.est_duration_sec, s.hold_point = row.hold_point
        """,
        rows=step_rows,
    )
    print("    " + str(len(step_rows)) + " steps")

    # 3. HAS_STEP relationships (with order property)
    session.run(
        """
        UNWIND $rows AS row
        MATCH (p:Procedure {id: row.proc_id})
        MATCH (s:Step {uid: row.uid})
        MERGE (p)-[r:HAS_STEP]->(s)
        SET r.order = row.step_number
        """,
        rows=step_rows,
    )

    # 4. PRECEDED_BY relationships
    prereq_rows = []
    for p in procs:
        for prereq_id in p.get("preceded_by", []):
            prereq_rows.append({"proc_id": p["id"], "prereq_id": prereq_id})
    if prereq_rows:
        session.run(
            """
            UNWIND $rows AS row
            MATCH (a:Procedure {id: row.proc_id})
            MATCH (b:Procedure {id: row.prereq_id})
            MERGE (a)-[:PRECEDED_BY]->(b)
            """,
            rows=prereq_rows,
        )
        print("    " + str(len(prereq_rows)) + " procedure prerequisites")

    # 5. REQUIRES_CERT relationships
    cert_rows = []
    for p in procs:
        for cert_id in p.get("required_certifications", []):
            cert_rows.append({"proc_id": p["id"], "cert_id": cert_id})
    if cert_rows:
        session.run(
            """
            UNWIND $rows AS row
            MATCH (p:Procedure {id: row.proc_id})
            MATCH (c:Certification {id: row.cert_id})
            MERGE (p)-[:REQUIRES_CERT]->(c)
            """,
            rows=cert_rows,
        )
        print("    " + str(len(cert_rows)) + " procedure-certification edges")

    # 6. Step relationships (USES_TOOL, USES_COMPONENT, HAS_TORQUE_SPEC, CAN_PRODUCE_DEFECT)
    tool_rows, comp_rows, spec_rows, defect_rows = [], [], [], []
    for p in procs:
        for s in p["steps"]:
            uid = p["id"] + ":" + str(s["step_number"])
            for tid in s.get("tools_used", []):
                tool_rows.append({"uid": uid, "tid": tid})
            for cid in s.get("components_used", []):
                comp_rows.append({"uid": uid, "cid": cid})
            ts_id = s.get("torque_spec_id")
            if ts_id:
                spec_rows.append({"uid": uid, "ts_id": ts_id})
            for did in s.get("potential_defects", []):
                defect_rows.append({"uid": uid, "did": did})

    session.run(
        """
        UNWIND $rows AS row
        MATCH (s:Step {uid: row.uid})
        MATCH (t:Tool {id: row.tid})
        MERGE (s)-[:USES_TOOL]->(t)
        """,
        rows=tool_rows,
    )
    print("    " + str(len(tool_rows)) + " step-tool edges")

    session.run(
        """
        UNWIND $rows AS row
        MATCH (s:Step {uid: row.uid})
        MATCH (c:Component {id: row.cid})
        MERGE (s)-[:USES_COMPONENT]->(c)
        """,
        rows=comp_rows,
    )
    print("    " + str(len(comp_rows)) + " step-component edges")

    if spec_rows:
        session.run(
            """
            UNWIND $rows AS row
            MATCH (s:Step {uid: row.uid})
            MATCH (ts:TorqueSpec {id: row.ts_id})
            MERGE (s)-[:HAS_TORQUE_SPEC]->(ts)
            """,
            rows=spec_rows,
        )
        print("    " + str(len(spec_rows)) + " step-torque-spec edges")

    if defect_rows:
        session.run(
            """
            UNWIND $rows AS row
            MATCH (s:Step {uid: row.uid})
            MATCH (d:Defect {id: row.did})
            MERGE (s)-[:CAN_PRODUCE_DEFECT]->(d)
            """,
            rows=defect_rows,
        )
        print("    " + str(len(defect_rows)) + " step-defect edges")


def report_stats(session):
    print()
    print("[*] Final graph stats:")
    result = session.run(
        """
        MATCH (n) RETURN labels(n)[0] AS label, count(*) AS c ORDER BY c DESC
        """
    )
    for record in result:
        print("    " + str(record["c"]).rjust(5) + " nodes labeled :" + record["label"])

    result = session.run(
        """
        MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS c ORDER BY c DESC
        """
    )
    for record in result:
        print("    " + str(record["c"]).rjust(5) + " :" + record["rel"] + " edges")


def main():
    catalog = json.loads(CATALOG_FILE.read_text())
    procs = json.loads(PROCEDURES_FILE.read_text())["procedures"]
    specs = json.loads(TORQUE_SPECS_FILE.read_text())["torque_specs"]

    print("[*] Connecting to " + URI + " ...")
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    try:
        driver.verify_connectivity()
        with driver.session() as session:
            t0 = time.perf_counter()
            wipe_data(session)
            setup_schema(session)
            load_catalog(session, catalog)
            load_torque_specs(session, specs)
            load_procedures(session, procs)
            report_stats(session)
            print()
            print("[OK] Loaded in {:.2f}s".format(time.perf_counter() - t0))
    finally:
        driver.close()


if __name__ == "__main__":
    main()
