"""Graph retriever: templated Cypher queries against the procedure knowledge graph.

The router is intentionally simple (regex + keywords). It returns a list of
(template_name, parameters) pairs. Each template is a parameterized Cypher
query that returns context relevant to the user question.

Design notes:
- No LLM in the router. Keeps baseline latency floor low and deterministic.
- Templates are conservative: they retrieve relevant context but don't try to
  compute the final answer. Synthesis happens in the LLM stage.
- Multiple templates can fire for one question. Their results are concatenated.
- Falls back to empty list if no template matches; vector retrieval still runs.
"""
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase

load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "assemblyrag2026")


# -----------------------
# Routing patterns (regex)
# -----------------------

# These patterns extract structured arguments from natural-language queries.
# Order matters: more specific patterns first.

RE_PROC_STEP = re.compile(r"\bPRC-(\d{3})\s*(?:step|s\.?)?\s*(\d{1,2})\b", re.IGNORECASE)
RE_PROC_ID   = re.compile(r"\bPRC-(\d{3})\b", re.IGNORECASE)
RE_TOOL_ID   = re.compile(r"\b(T-\d{3})\b", re.IGNORECASE)
RE_COMP_ID   = re.compile(r"\b(C-\d{3})\b", re.IGNORECASE)
RE_CERT_ID   = re.compile(r"\b(CERT-\d{1,2})\b", re.IGNORECASE)
RE_DEFECT_ID = re.compile(r"\b(D-\d{3})\b", re.IGNORECASE)
RE_TS_ID     = re.compile(r"\b(TS-\d{3})\b", re.IGNORECASE)

RE_CERT_LEVEL = re.compile(r"\b(?:level|lvl|l)\s*([1-3])\b", re.IGNORECASE)
RE_TORQUE_GT  = re.compile(r"torque\b[\w\s]{0,30}?\b(?:above|over|greater than|>)\s+(\d+)\s*nm", re.IGNORECASE)
RE_TORQUE_LT  = re.compile(r"torque\b[\w\s]{0,30}?\b(?:below|under|less than|<)\s+(\d+)\s*nm", re.IGNORECASE)
RE_FASTENER_M = re.compile(r"\bM(\d{1,2})\b")

EQUIPMENT_KEYWORDS = {
    "centrifugal_pump":         ["centrifugal pump", "pump"],
    "induction_motor":          ["induction motor", "electric motor", "motor"],
    "globe_valve":              ["globe valve"],
    "gate_valve":               ["gate valve"],
    "shell_tube_heat_exchanger":["heat exchanger", "shell-tube", "tube bundle"],
    "reciprocating_compressor": ["reciprocating compressor", "compressor"],
    "instrumentation":          ["transmitter", "instrumentation", "calibration"],
}

CERT_KEYWORDS = {
    "electrical": ["CERT-4", "CERT-5"],
    "pressure":   ["CERT-6", "CERT-7"],
    "instrumentation_cert": ["CERT-8"],
    "confined":   ["CERT-9"],
    "lifting":    ["CERT-10"],
    "mechanical": ["CERT-1", "CERT-2", "CERT-3"],
}


# -----------------------
# Templates (Cypher)
# -----------------------

TEMPLATES = {
    "procedure_by_id": """
        MATCH (p:Procedure {id: $proc_id})
        OPTIONAL MATCH (p)-[:REQUIRES_CERT]->(cert:Certification)
        OPTIONAL MATCH (p)-[:HAS_STEP]->(s:Step)
        OPTIONAL MATCH (p)-[:PRECEDED_BY]->(prereq:Procedure)
        WITH p,
             collect(DISTINCT cert.name) AS certs,
             collect(DISTINCT prereq.id) AS prereqs,
             collect(DISTINCT {n: s.step_number, txt: s.instruction, hp: s.hold_point})
                AS steps
        RETURN p.id AS id, p.title AS title, p.equipment_type AS equipment,
               p.criticality AS criticality, p.est_duration_min AS duration_min,
               certs, prereqs, steps
    """,

    "step_by_uid": """
        MATCH (s:Step {uid: $uid})
        MATCH (p:Procedure)-[:HAS_STEP]->(s)
        OPTIONAL MATCH (s)-[:USES_TOOL]->(t:Tool)
        OPTIONAL MATCH (s)-[:USES_COMPONENT]->(c:Component)
        OPTIONAL MATCH (s)-[:HAS_TORQUE_SPEC]->(ts:TorqueSpec)
        OPTIONAL MATCH (s)-[:CAN_PRODUCE_DEFECT]->(d:Defect)
        OPTIONAL MATCH (ts)-[:APPLIES_TO]->(fc:Component)
        WITH s, p,
             collect(DISTINCT {id: t.id, name: t.name, type: t.type}) AS tools,
             collect(DISTINCT {id: c.id, name: c.name}) AS comps,
             collect(DISTINCT {
                 id: ts.id, target_nm: ts.target_nm, tolerance_nm: ts.tolerance_nm,
                 pattern: ts.pattern, applies_to: fc.name
             }) AS torque,
             collect(DISTINCT {id: d.id, name: d.name, severity: d.severity}) AS defects
        RETURN p.id AS proc_id, p.title AS proc_title,
               s.step_number AS step_number, s.instruction AS instruction,
               s.hold_point AS hold_point,
               tools, comps, torque, defects
    """,

    "tool_lookup": """
        MATCH (t:Tool)
        WHERE t.id = $tool_id OR toLower(t.name) CONTAINS toLower($name_substr)
        OPTIONAL MATCH (s:Step)-[:USES_TOOL]->(t)
        OPTIONAL MATCH (p:Procedure)-[:HAS_STEP]->(s)
        WITH t,
             collect(DISTINCT {proc: p.id, step: s.step_number}) AS used_in
        RETURN t.id AS id, t.name AS name, t.type AS type,
               t.calibration_required AS cal_required, used_in
    """,

    "procedures_by_equipment": """
        MATCH (p:Procedure {equipment_type: $equipment})
        OPTIONAL MATCH (p)-[:REQUIRES_CERT]->(cert:Certification)
        WITH p, collect(DISTINCT cert.name) AS certs
        RETURN p.id AS id, p.title AS title, p.criticality AS criticality,
               p.est_duration_min AS duration_min, certs
        ORDER BY p.id
    """,

    "procedures_by_cert": """
        MATCH (p:Procedure)-[:REQUIRES_CERT]->(c:Certification)
        WHERE c.id IN $cert_ids
        RETURN DISTINCT p.id AS id, p.title AS title, p.equipment_type AS equipment,
               c.id AS matched_cert, c.name AS cert_name
        ORDER BY p.id
    """,

    "procedures_by_cert_level": """
        MATCH (p:Procedure)-[:REQUIRES_CERT]->(c:Certification)
        WHERE c.level >= $min_level
        RETURN DISTINCT p.id AS id, p.title AS title, p.equipment_type AS equipment,
               collect(DISTINCT c.name) AS certs
        ORDER BY p.id
    """,

    "torque_filter_gt": """
        MATCH (ts:TorqueSpec)-[:APPLIES_TO]->(c:Component)
        WHERE ts.target_nm > $threshold
        OPTIONAL MATCH (s:Step)-[:HAS_TORQUE_SPEC]->(ts)
        OPTIONAL MATCH (p:Procedure)-[:HAS_STEP]->(s)
        RETURN ts.id AS ts_id, ts.target_nm AS target_nm, ts.pattern AS pattern,
               c.name AS fastener,
               collect(DISTINCT {proc: p.id, step: s.step_number,
                                 instruction: s.instruction}) AS used_in
        ORDER BY ts.target_nm DESC
    """,

    "torque_filter_lt": """
        MATCH (ts:TorqueSpec)-[:APPLIES_TO]->(c:Component)
        WHERE ts.target_nm < $threshold
        OPTIONAL MATCH (s:Step)-[:HAS_TORQUE_SPEC]->(ts)
        OPTIONAL MATCH (p:Procedure)-[:HAS_STEP]->(s)
        RETURN ts.id AS ts_id, ts.target_nm AS target_nm, ts.pattern AS pattern,
               c.name AS fastener,
               collect(DISTINCT {proc: p.id, step: s.step_number,
                                 instruction: s.instruction}) AS used_in
        ORDER BY ts.target_nm ASC
    """,

    "torque_for_metric": """
        MATCH (ts:TorqueSpec)-[:APPLIES_TO]->(c:Component)
        WHERE c.name CONTAINS $size_pattern
        RETURN ts.id AS ts_id, ts.target_nm AS target_nm,
               ts.tolerance_nm AS tolerance_nm, ts.pattern AS pattern,
               c.name AS fastener
        ORDER BY ts.target_nm DESC
    """,
}


@dataclass
class TemplateInvocation:
    name: str
    params: Dict[str, Any]


def route(question: str) -> List[TemplateInvocation]:
    """Decide which template(s) to fire based on question text. Returns a list
    of (name, params) pairs. Empty list = no graph retrieval, vector only."""
    q = question
    invocations: List[TemplateInvocation] = []

    # Most specific: PRC-XXX step N
    m = RE_PROC_STEP.search(q)
    if m:
        proc_num, step_num = m.group(1), m.group(2)
        invocations.append(TemplateInvocation(
            "step_by_uid", {"uid": "PRC-" + proc_num + ":" + step_num}))
        return invocations  # specific step query, no need for broader matches

    # Procedure ID alone
    m = RE_PROC_ID.search(q)
    if m:
        invocations.append(TemplateInvocation(
            "procedure_by_id", {"proc_id": "PRC-" + m.group(1)}))

    # Tool ID
    m = RE_TOOL_ID.search(q)
    if m:
        invocations.append(TemplateInvocation(
            "tool_lookup", {"tool_id": m.group(1).upper(), "name_substr": "__no_match__"}))

    # Cert ID -> get procedures that need it
    cert_ids = [match.upper() for match in RE_CERT_ID.findall(q)]
    if cert_ids:
        invocations.append(TemplateInvocation(
            "procedures_by_cert", {"cert_ids": cert_ids}))

    # Cert family (mechanical/electrical/etc.)
    q_lower = q.lower()
    cert_family_ids = []
    for family, ids in CERT_KEYWORDS.items():
        if family.replace("_cert", "") in q_lower:
            cert_family_ids.extend(ids)
    if cert_family_ids and not cert_ids:
        invocations.append(TemplateInvocation(
            "procedures_by_cert", {"cert_ids": list(set(cert_family_ids))}))

    # Cert level
    m = RE_CERT_LEVEL.search(q)
    if m:
        invocations.append(TemplateInvocation(
            "procedures_by_cert_level", {"min_level": int(m.group(1))}))

    # Torque thresholds
    m = RE_TORQUE_GT.search(q)
    if m:
        invocations.append(TemplateInvocation(
            "torque_filter_gt", {"threshold": float(m.group(1))}))
    m = RE_TORQUE_LT.search(q)
    if m:
        invocations.append(TemplateInvocation(
            "torque_filter_lt", {"threshold": float(m.group(1))}))

    # Metric fastener size (M10, M16, etc.)
    m = RE_FASTENER_M.search(q)
    if m:
        invocations.append(TemplateInvocation(
            "torque_for_metric", {"size_pattern": "M" + m.group(1)}))

    # Equipment type keywords
    matched_equipment = []
    for eq, kws in EQUIPMENT_KEYWORDS.items():
        if any(kw in q_lower for kw in kws):
            matched_equipment.append(eq)
    # If multiple match (e.g. "centrifugal pump" matches both "centrifugal_pump"
    # and "pump"), prefer the more specific one
    if matched_equipment:
        # Heuristic: longest equipment-type string wins
        best = sorted(matched_equipment, key=lambda x: -len(x))[0]
        invocations.append(TemplateInvocation(
            "procedures_by_equipment", {"equipment": best}))

    return invocations


def execute(driver: Driver, invocations: List[TemplateInvocation]) -> List[Dict[str, Any]]:
    """Run each invocation against Neo4j, return list of result blocks."""
    results = []
    with driver.session() as session:
        for inv in invocations:
            cypher = TEMPLATES[inv.name]
            try:
                rows = [dict(r) for r in session.run(cypher, **inv.params)]
            except Exception as e:
                rows = [{"error": str(e)[:200]}]
            results.append({
                "template": inv.name,
                "params": inv.params,
                "rows": rows,
            })
    return results


def retrieve(driver: Driver, question: str) -> Tuple[List[TemplateInvocation],
                                                       List[Dict[str, Any]]]:
    """Convenience wrapper: route + execute in one call."""
    invocations = route(question)
    results = execute(driver, invocations)
    return invocations, results


# Quick CLI for ad-hoc testing
if __name__ == "__main__":
    import sys
    import json

    PROBES = [
        "Tell me about PRC-014",
        "What torque should I use for M20 bolts?",
        "Which procedures require electrical certification?",
        "Show me PRC-002 step 3",
        "List torque specs above 200 Nm",
        "What does T-005 do?",
        "Which procedures involve a centrifugal pump?",
        "what defects can occur during compressor service",
    ]

    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    for q in PROBES:
        print()
        print("Q: " + q)
        invs, res = retrieve(driver, q)
        if not invs:
            print("  [no template fired - would fall back to vector only]")
            continue
        for inv, block in zip(invs, res):
            print("  -> " + inv.name + " params=" + json.dumps(inv.params))
            print("     " + str(len(block["rows"])) + " rows returned")
            if block["rows"]:
                first = block["rows"][0]
                print("     first row keys: " + str(list(first.keys())))
    driver.close()
