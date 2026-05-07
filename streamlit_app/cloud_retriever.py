"""Cloud-aware hybrid retriever for the live tab.

Schema (verified against AuraDB):
- TorqueSpec props: id, target_nm, tolerance_nm, pattern, conditions
- Component props: id, name, material, category
- Step -[HAS_TORQUE_SPEC]-> TorqueSpec
- TorqueSpec -[APPLIES_TO]-> Component (M20 etc. lives in Component.name)
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer


APP_ROOT = Path(__file__).resolve().parent
EMBEDDINGS_FILE = APP_ROOT / "data" / "embeddings.npz"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_VECTOR_TOP_K = 5
DEFAULT_TOKEN_BUDGET = 2000
CHARS_PER_TOKEN = 4


def _id_pattern(prefix: str) -> str:
    """Match exactly an ID like 'PRC-014' without picking up 'PRCI-014' or 'PRC-0145'."""
    return r"(?<![A-Za-z])(" + prefix + r"-[0-9]{1,3})(?![A-Za-z0-9])"


TEMPLATES = {
    "step_by_uid": {
        "patterns": [
            r"(?<![A-Za-z])(PRC-[0-9]{1,3})(?![A-Za-z0-9])[\s\S]*?step\s*([0-9]{1,3})",
        ],
        "cypher": (
            "MATCH (p:Procedure {id: $proc_id})-[:HAS_STEP]->(s:Step {step_number: $step_num}) "
            "OPTIONAL MATCH (s)-[:USES_TOOL]->(t:Tool) "
            "OPTIONAL MATCH (s)-[:USES_COMPONENT]->(c:Component) "
            "OPTIONAL MATCH (s)-[:HAS_TORQUE_SPEC]->(ts:TorqueSpec) "
            "RETURN p.title AS title, s, "
            "  collect(DISTINCT {id: t.id, name: t.name}) AS tools, "
            "  collect(DISTINCT {id: c.id, name: c.name}) AS components, "
            "  collect(DISTINCT ts) AS torque_specs"
        ),
    },
    "procedure_by_id": {
        "patterns": [_id_pattern("PRC")],
        "cypher": (
            "MATCH (p:Procedure {id: $proc_id}) "
            "OPTIONAL MATCH (p)-[:HAS_STEP]->(s:Step) "
            "OPTIONAL MATCH (p)-[:REQUIRES_CERT]->(c:Certification) "
            "RETURN p AS procedure, "
            "  collect(DISTINCT s) AS steps, "
            "  collect(DISTINCT c.name) AS cert_names"
        ),
    },
    "torquespec_by_id": {
        "patterns": [_id_pattern("TS")],
        "cypher": (
            "MATCH (ts:TorqueSpec {id: $ts_id}) "
            "OPTIONAL MATCH (ts)-[:APPLIES_TO]->(c:Component) "
            "RETURN ts, c.id AS component_id, c.name AS component_name, "
            "       c.material AS component_material"
        ),
    },
    "tool_lookup": {
        "patterns": [_id_pattern("T")],
        "cypher": (
            "MATCH (t:Tool {id: $tool_id}) "
            "OPTIONAL MATCH (s:Step)-[:USES_TOOL]->(t) "
            "OPTIONAL MATCH (p:Procedure)-[:HAS_STEP]->(s) "
            "RETURN t, "
            "  collect(DISTINCT {procedure_id: p.id, step_number: s.step_number}) AS uses"
        ),
    },
    "torque_for_metric": {
        "patterns": [
            r"(?<![A-Za-z])M([0-9]{1,3})(?![A-Za-z0-9])[\s\S]*?(?:torque|nm|bolt|nut)",
            r"(?:torque|bolt|nut)[\s\S]*?(?<![A-Za-z])M([0-9]{1,3})(?![A-Za-z0-9])",
        ],
        "cypher": (
            "MATCH (ts:TorqueSpec)-[:APPLIES_TO]->(c:Component) "
            "WHERE c.name CONTAINS ('M' + $metric_size + ' ') "
            "   OR c.name CONTAINS ('M' + $metric_size + 'x') "
            "RETURN ts, c.name AS component_name, c.id AS component_id "
            "ORDER BY ts.target_nm DESC LIMIT 10"
        ),
    },
    "torque_filter_gt": {
        "patterns": [
            r"torque[\s\w]{0,30}?(?:above|over|greater than|more than|higher than|>)\s*([0-9]+)",
        ],
        "cypher": (
            "MATCH (ts:TorqueSpec)-[:APPLIES_TO]->(c:Component) "
            "WHERE ts.target_nm > $threshold "
            "RETURN ts, c.name AS component_name, c.id AS component_id "
            "ORDER BY ts.target_nm DESC LIMIT 10"
        ),
    },
    "torque_filter_lt": {
        "patterns": [
            r"torque[\s\w]{0,30}?(?:below|under|less than|fewer than|lower than|<)\s*([0-9]+)",
        ],
        "cypher": (
            "MATCH (ts:TorqueSpec)-[:APPLIES_TO]->(c:Component) "
            "WHERE ts.target_nm < $threshold "
            "RETURN ts, c.name AS component_name, c.id AS component_id "
            "ORDER BY ts.target_nm ASC LIMIT 10"
        ),
    },
    "procedures_by_equipment": {
        "patterns": [
            r"(?<![A-Za-z])(centrifugal[_\s]+pump|induction[_\s]+motor|"
            r"reciprocating[_\s]+compressor|globe[_\s]+valve|gate[_\s]+valve|"
            r"shell[_\s]+tube[_\s]+heat[_\s]+exchanger|"
            r"shell[_\s]+and[_\s]+tube[_\s]+heat[_\s]+exchanger|instrumentation|"
            r"pump|motor|compressor|valve|heat\s+exchanger)(?![A-Za-z])",
        ],
        "cypher": (
            "MATCH (p:Procedure {equipment_type: $equipment}) "
            "OPTIONAL MATCH (p)-[:REQUIRES_CERT]->(c:Certification) "
            "RETURN p, collect(DISTINCT c.name) AS cert_names ORDER BY p.id"
        ),
    },
    "procedures_by_cert": {
        "patterns": [
            r"(?<![A-Za-z])(electrical|mechanical|pressure|lifting|height|chemical|confined|hot)(?![A-Za-z])",
        ],
        "cypher": (
            "MATCH (p:Procedure)-[:REQUIRES_CERT]->(c:Certification) "
            "WHERE toLower(c.name) CONTAINS toLower($cert_keyword) "
            "RETURN DISTINCT p, collect(DISTINCT c.id) AS cert_ids ORDER BY p.id"
        ),
    },
}


EQUIPMENT_CANONICAL = {
    "centrifugal pump": "centrifugal_pump",
    "centrifugal_pump": "centrifugal_pump",
    "induction motor": "induction_motor",
    "induction_motor": "induction_motor",
    "reciprocating compressor": "reciprocating_compressor",
    "reciprocating_compressor": "reciprocating_compressor",
    "globe valve": "globe_valve",
    "globe_valve": "globe_valve",
    "gate valve": "gate_valve",
    "gate_valve": "gate_valve",
    "shell tube heat exchanger": "shell_tube_heat_exchanger",
    "shell_tube_heat_exchanger": "shell_tube_heat_exchanger",
    "shell and tube heat exchanger": "shell_tube_heat_exchanger",
    "shell_and_tube_heat_exchanger": "shell_tube_heat_exchanger",
    "instrumentation": "instrumentation",
    "pump": "centrifugal_pump",
    "motor": "induction_motor",
    "compressor": "reciprocating_compressor",
    "valve": "globe_valve",
    "heat exchanger": "shell_tube_heat_exchanger",
}


@dataclass
class CloudHybridContext:
    question: str
    formatted_context: str
    graph_invocations: List[str] = field(default_factory=list)
    vector_uids: List[str] = field(default_factory=list)
    char_count: int = 0
    est_tokens: int = 0


class CloudHybridRetriever:
    def __init__(self, secrets, vector_top_k: int = DEFAULT_VECTOR_TOP_K):
        self.secrets = secrets
        self.vector_top_k = vector_top_k
        self.driver = GraphDatabase.driver(
            secrets["NEO4J_URI"],
            auth=(secrets["NEO4J_USERNAME"], secrets["NEO4J_PASSWORD"]),
        )
        self.database = secrets.get("NEO4J_DATABASE", "neo4j")

        if not EMBEDDINGS_FILE.exists():
            raise FileNotFoundError("Embeddings file not found: " + str(EMBEDDINGS_FILE))
        npz = np.load(EMBEDDINGS_FILE, allow_pickle=False)
        self.embeddings = npz["embeddings"].astype(np.float32)
        self.documents = npz["documents"].tolist()
        self.ids = npz["ids"].tolist()
        self.metadatas = [json.loads(m) for m in npz["metadatas"].tolist()]

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-9
        self.embeddings_norm = self.embeddings / norms

        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    def close(self):
        try:
            self.driver.close()
        except Exception:
            pass

    def _run_template(self, template_key, params):
        cypher = TEMPLATES[template_key]["cypher"]
        with self.driver.session(database=self.database) as session:
            return list(session.run(cypher, **params))

    def _try_template(self, question, template_key, lower_q):
        tmpl = TEMPLATES[template_key]
        for pattern in tmpl["patterns"]:
            m = re.search(pattern, lower_q, re.IGNORECASE)
            if not m:
                continue

            if template_key == "procedure_by_id":
                return self._run_template(template_key, {"proc_id": m.group(1).upper()})
            elif template_key == "step_by_uid":
                return self._run_template(template_key, {
                    "proc_id": m.group(1).upper(),
                    "step_num": int(m.group(2)),
                })
            elif template_key == "tool_lookup":
                return self._run_template(template_key, {"tool_id": m.group(1).upper()})
            elif template_key == "torquespec_by_id":
                return self._run_template(template_key, {"ts_id": m.group(1).upper()})
            elif template_key == "torque_for_metric":
                return self._run_template(template_key, {"metric_size": m.group(1)})
            elif template_key == "torque_filter_gt":
                return self._run_template(template_key, {"threshold": float(m.group(1))})
            elif template_key == "torque_filter_lt":
                return self._run_template(template_key, {"threshold": float(m.group(1))})
            elif template_key == "procedures_by_equipment":
                raw = m.group(1).lower().replace("_", " ").strip()
                canonical = EQUIPMENT_CANONICAL.get(raw)
                if not canonical:
                    continue
                return self._run_template(template_key, {"equipment": canonical})
            elif template_key == "procedures_by_cert":
                return self._run_template(template_key, {"cert_keyword": m.group(1)})
        return None

    def _format_template_results(self, template_key, records):
        if template_key == "procedure_by_id":
            rec = records[0]
            p = rec["procedure"]
            steps = rec["steps"]
            certs = rec["cert_names"]
            lines = [
                "## Procedure " + p["id"] + ": " + p["title"],
                "- Equipment: " + p["equipment_type"],
                "- Criticality: " + p["criticality"],
                "- Duration: " + str(p.get("duration_minutes", "?")) + " min",
                "- Required certifications: " + (", ".join(c for c in certs if c) or "(none)"),
                "- Steps:",
            ]
            for s in sorted(steps, key=lambda x: x.get("step_number", 0)):
                hp = " [HOLD POINT]" if s.get("hold_point") else ""
                lines.append("  " + str(s["step_number"]) + ". " + s["instruction"] + hp)
            return "\n".join(lines)

        elif template_key == "step_by_uid":
            rec = records[0]
            s = rec["s"]
            tools = [t for t in rec["tools"] if t.get("id")]
            comps = [c for c in rec["components"] if c.get("id")]
            torques = [t for t in rec["torque_specs"] if t]
            lines = [
                "## Step " + str(s["step_number"]) + " of " + rec["title"],
                "**Instruction:** " + s["instruction"],
                "- Hold point: " + str(s.get("hold_point", False)),
            ]
            if tools:
                lines.append("- Tools: " + ", ".join(t["id"] + " (" + t["name"] + ")" for t in tools))
            if comps:
                lines.append("- Components: " + ", ".join(c["id"] + " (" + c["name"] + ")" for c in comps))
            if torques:
                tline = ", ".join(t["id"] + ": " + str(t["target_nm"]) + " Nm" for t in torques)
                lines.append("- Torque specs: " + tline)
            return "\n".join(lines)

        elif template_key == "torquespec_by_id":
            rec = records[0]
            ts = rec["ts"]
            lines = [
                "## TorqueSpec " + ts["id"],
                "- Target: " + str(ts["target_nm"]) + " Nm",
                "- Tolerance: +/- " + str(ts.get("tolerance_nm", "?")) + " Nm",
                "- Pattern: " + ts.get("pattern", "?"),
                "- Conditions: " + ts.get("conditions", "?"),
            ]
            if rec.get("component_id"):
                lines.append("- Applies to: " + rec["component_id"] + " (" +
                             (rec.get("component_name") or "?") + ")")
            return "\n".join(lines)

        elif template_key == "tool_lookup":
            rec = records[0]
            t = rec["t"]
            uses = [u for u in rec["uses"] if u.get("procedure_id")]
            lines = [
                "## Tool " + t["id"] + ": " + t["name"],
                "- Type: " + t.get("tool_type", "?"),
                "- Calibration required: " + str(t.get("calibration_required", False)),
            ]
            if uses:
                use_lines = ["  - " + u["procedure_id"] + " step " + str(u["step_number"]) for u in uses[:10]]
                lines.append("- Used in (up to 10):")
                lines.extend(use_lines)
            return "\n".join(lines)

        elif template_key in ("torque_for_metric", "torque_filter_gt", "torque_filter_lt"):
            label = {
                "torque_for_metric": "fastener size",
                "torque_filter_gt": "torque > threshold",
                "torque_filter_lt": "torque < threshold",
            }[template_key]
            lines = ["## Torque specs for " + label + " (" + str(len(records)) + " specs)"]
            for rec in records:
                ts = rec["ts"]
                pat = ts.get("pattern", "?")
                tol = ts.get("tolerance_nm", "?")
                comp_name = rec.get("component_name") or "?"
                comp_id = rec.get("component_id") or "?"
                lines.append(
                    "- " + ts["id"] + ": " + str(ts["target_nm"]) + " Nm +/- " + str(tol) +
                    " Nm (" + pat + " pattern, applies to " + comp_id + " " + comp_name + ")"
                )
            return "\n".join(lines)

        elif template_key == "procedures_by_equipment":
            lines = ["## Procedures matching equipment filter (" + str(len(records)) + " procedures)"]
            for rec in records:
                p = rec["p"]
                certs = ", ".join(c for c in rec["cert_names"] if c)
                lines.append(
                    "- " + p["id"] + ": " + p["title"] + " [" + p["criticality"] + "] - certs: " + certs
                )
            return "\n".join(lines)

        elif template_key == "procedures_by_cert":
            lines = ["## Procedures matching certification filter (" + str(len(records)) + " procedures)"]
            for rec in records:
                p = rec["p"]
                cert_ids = ", ".join(rec["cert_ids"])
                lines.append(
                    "- " + p["id"] + ": " + p["title"] + " (" + p["equipment_type"] + ", matches " + cert_ids + ")"
                )
            return "\n".join(lines)
        return ""

    def graph_retrieve(self, question):
        lower_q = question.lower()
        priority = [
            "step_by_uid",
            "procedure_by_id",
            "torquespec_by_id",
            "tool_lookup",
            "torque_filter_gt",
            "torque_filter_lt",
            "torque_for_metric",
            "procedures_by_equipment",
            "procedures_by_cert",
        ]
        out = []
        for key in priority:
            try:
                records = self._try_template(question, key, lower_q)
            except Exception:
                continue
            if records:
                block = self._format_template_results(key, records)
                if block:
                    out.append((key, block))
        return out

    def vector_retrieve(self, question, top_k=None):
        if top_k is None:
            top_k = self.vector_top_k
        q_emb = self.embed_model.encode([question], normalize_embeddings=True)[0].astype(np.float32)
        sims = self.embeddings_norm @ q_emb
        top_idx = np.argsort(-sims)[:top_k]
        results = []
        for idx in top_idx:
            results.append({
                "id": self.ids[idx],
                "document": self.documents[idx],
                "metadata": self.metadatas[idx],
                "score": float(sims[idx]),
            })
        return results

    def retrieve(self, question):
        graph_blocks = self.graph_retrieve(question)
        vector_hits = self.vector_retrieve(question)

        all_blocks_text = " ".join(b for _, b in graph_blocks)
        graph_proc_ids = set(re.findall(r"PRC-[0-9]{1,3}", all_blocks_text))

        kept_vec = []
        for hit in vector_hits:
            proc_id = hit["metadata"].get("procedure_id", "")
            if proc_id in graph_proc_ids:
                continue
            kept_vec.append(hit)

        sections = []
        invocations = []
        for key, block in graph_blocks:
            sections.append(block)
            invocations.append(key)
        if kept_vec:
            vec_lines = ["## Additional semantically-similar steps"]
            for hit in kept_vec:
                vec_lines.append("- " + hit["document"])
            sections.append("\n".join(vec_lines))

        formatted = "\n\n".join(sections)
        if len(formatted) > DEFAULT_TOKEN_BUDGET * CHARS_PER_TOKEN:
            formatted = formatted[: DEFAULT_TOKEN_BUDGET * CHARS_PER_TOKEN]

        return CloudHybridContext(
            question=question,
            formatted_context=formatted,
            graph_invocations=invocations,
            vector_uids=[h["id"] for h in kept_vec],
            char_count=len(formatted),
            est_tokens=len(formatted) // CHARS_PER_TOKEN,
        )
