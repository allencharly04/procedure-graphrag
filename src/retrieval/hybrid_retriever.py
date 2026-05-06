"""Hybrid retriever: combine vector + graph retrieval into one context block.

Both retrievers always run. Graph results are formatted as structured headers
followed by relevant facts. Vector results are appended as additional candidate
steps. Duplicates (steps already covered by a graph result) are removed.

Context is capped at ~2000 tokens to keep prompts compact for small LLMs.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import chromadb
from neo4j import Driver, GraphDatabase
from sentence_transformers import SentenceTransformer

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_retriever import retrieve as graph_retrieve

load_dotenv()
os.environ["ANONYMIZED_TELEMETRY"] = "False"

ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION = "steps"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "assemblyrag2026")

# Approx tokens per char for English text + structured data (rough heuristic)
CHARS_PER_TOKEN = 4
DEFAULT_TOKEN_BUDGET = 2000


@dataclass
class HybridContext:
    """Result of hybrid retrieval, packaged for the LLM."""
    question: str
    formatted_context: str
    graph_invocations: List[str] = field(default_factory=list)
    vector_uids: List[str] = field(default_factory=list)
    char_count: int = 0
    est_tokens: int = 0


# -----------------------
# Formatters: turn raw retrieval results into Markdown blocks
# -----------------------

def fmt_procedure_by_id(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    p = rows[0]
    if "error" in p:
        return ""
    out = ["## Procedure " + p["id"] + ": " + p["title"]]
    out.append("- Equipment: " + str(p["equipment"]))
    out.append("- Criticality: " + str(p["criticality"]))
    out.append("- Duration: " + str(p["duration_min"]) + " min")
    if p.get("certs"):
        out.append("- Required certifications: " + ", ".join(p["certs"]))
    if p.get("prereqs"):
        out.append("- Prerequisites: " + ", ".join(p["prereqs"]))
    out.append("- Steps:")
    for step in sorted(p.get("steps", []), key=lambda s: s.get("n", 0)):
        if step.get("n") is None:
            continue
        marker = " [HOLD POINT]" if step.get("hp") else ""
        out.append("  " + str(step["n"]) + ". " + step["txt"] + marker)
    return "\n".join(out)


def fmt_step_by_uid(rows: List[Dict[str, Any]]) -> str:
    if not rows or "error" in rows[0]:
        return ""
    s = rows[0]
    out = ["## Step " + str(s["step_number"]) + " of " + s["proc_id"]
           + " (" + s["proc_title"] + ")"]
    out.append("**Instruction:** " + s["instruction"])
    out.append("- Hold point: " + str(s["hold_point"]))
    tools = [t for t in s.get("tools", []) if t.get("id")]
    if tools:
        out.append("- Tools: " + ", ".join(t["id"] + " (" + t["name"] + ")" for t in tools))
    comps = [c for c in s.get("comps", []) if c.get("id")]
    if comps:
        out.append("- Components: " + ", ".join(c["id"] + " (" + c["name"] + ")" for c in comps))
    torque = [t for t in s.get("torque", []) if t.get("id")]
    if torque:
        for t in torque:
            out.append("- Torque " + t["id"] + ": "
                       + str(t["target_nm"]) + " Nm +/- " + str(t["tolerance_nm"])
                       + " (" + t["pattern"] + " pattern, applies to " + str(t["applies_to"]) + ")")
    defects = [d for d in s.get("defects", []) if d.get("id")]
    if defects:
        out.append("- Potential defects: "
                   + ", ".join(d["id"] + " (" + d["name"] + ", " + d["severity"] + ")" for d in defects))
    return "\n".join(out)


def fmt_tool_lookup(rows: List[Dict[str, Any]]) -> str:
    if not rows or "error" in rows[0]:
        return ""
    blocks = []
    for t in rows:
        out = ["## Tool " + t["id"] + ": " + t["name"]]
        out.append("- Type: " + str(t["type"]))
        out.append("- Calibration required: " + str(t["cal_required"]))
        used = [u for u in t.get("used_in", []) if u.get("proc")]
        if used:
            sample = used[:8]
            out.append("- Used in (sample): "
                       + ", ".join(u["proc"] + " step " + str(u["step"]) for u in sample))
            if len(used) > 8:
                out.append("  (and " + str(len(used) - 8) + " more)")
        blocks.append("\n".join(out))
    return "\n\n".join(blocks)


def fmt_procedures_by_equipment(rows: List[Dict[str, Any]]) -> str:
    if not rows or "error" in rows[0]:
        return ""
    eq = rows[0].get("equipment", "?")  # Note: equipment is in metadata not row directly
    out = ["## Procedures matching equipment filter (" + str(len(rows)) + " procedures)"]
    for p in rows:
        line = "- " + p["id"] + ": " + p["title"] + " [" + str(p["criticality"]) + "]"
        if p.get("certs"):
            line += " - certs: " + ", ".join(p["certs"])
        out.append(line)
    return "\n".join(out)


def fmt_procedures_by_cert(rows: List[Dict[str, Any]]) -> str:
    if not rows or "error" in rows[0]:
        return ""
    out = ["## Procedures matching certification filter (" + str(len(rows)) + " procedures)"]
    for p in rows:
        out.append("- " + p["id"] + ": " + p["title"]
                   + " (" + str(p.get("equipment", "?"))
                   + ", matches " + str(p.get("matched_cert", "?")) + ")")
    return "\n".join(out)


def fmt_procedures_by_cert_level(rows: List[Dict[str, Any]]) -> str:
    if not rows or "error" in rows[0]:
        return ""
    out = ["## Procedures matching cert-level filter (" + str(len(rows)) + " procedures)"]
    for p in rows:
        certs = ", ".join(p.get("certs", []))
        out.append("- " + p["id"] + ": " + p["title"] + " - certs: " + certs)
    return "\n".join(out)


def fmt_torque_filter(rows: List[Dict[str, Any]], label: str) -> str:
    if not rows or "error" in rows[0]:
        return ""
    out = ["## Torque specs " + label + " (" + str(len(rows)) + " specs)"]
    for ts in rows:
        line = ("- " + ts["ts_id"] + ": " + str(ts["target_nm"]) + " Nm, "
                + str(ts["pattern"]) + " pattern, " + str(ts["fastener"]))
        used = [u for u in ts.get("used_in", []) if u.get("proc")]
        if used:
            line += " (used in " + ", ".join(u["proc"] + ":step " + str(u["step"])
                                              for u in used[:3]) + ")"
        out.append(line)
    return "\n".join(out)


def fmt_torque_for_metric(rows: List[Dict[str, Any]]) -> str:
    if not rows or "error" in rows[0]:
        return ""
    out = ["## Torque specs for fastener size (" + str(len(rows)) + " specs)"]
    for ts in rows:
        out.append("- " + ts["ts_id"] + ": " + str(ts["target_nm"])
                   + " Nm +/- " + str(ts["tolerance_nm"])
                   + " (" + str(ts["pattern"]) + " pattern, " + str(ts["fastener"]) + ")")
    return "\n".join(out)


FORMATTERS = {
    "procedure_by_id":           fmt_procedure_by_id,
    "step_by_uid":               fmt_step_by_uid,
    "tool_lookup":               fmt_tool_lookup,
    "procedures_by_equipment":   fmt_procedures_by_equipment,
    "procedures_by_cert":        fmt_procedures_by_cert,
    "procedures_by_cert_level":  fmt_procedures_by_cert_level,
    "torque_filter_gt":          lambda r: fmt_torque_filter(r, "(threshold filter)"),
    "torque_filter_lt":          lambda r: fmt_torque_filter(r, "(threshold filter)"),
    "torque_for_metric":         fmt_torque_for_metric,
}


# -----------------------
# Hybrid Retriever
# -----------------------

class HybridRetriever:
    def __init__(self, vector_top_k: int = 5,
                 token_budget: int = DEFAULT_TOKEN_BUDGET):
        self.vector_top_k = vector_top_k
        self.token_budget = token_budget

        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        self.chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.chroma.get_collection(COLLECTION)
        self.driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

    def close(self):
        self.driver.close()

    def _vector_search(self, question: str) -> Tuple[List[str], List[str]]:
        emb = self.embed_model.encode([question], normalize_embeddings=True).tolist()
        hits = self.collection.query(
            query_embeddings=emb,
            n_results=self.vector_top_k,
            include=["documents", "metadatas"],
        )
        return hits["documents"][0], [m for m in hits["metadatas"][0]]

    def retrieve(self, question: str) -> HybridContext:
        # 1. Graph retrieval
        invocations, graph_results = graph_retrieve(self.driver, question)

        graph_blocks = []
        graph_covered_uids = set()
        for inv, block in zip(invocations, graph_results):
            formatter = FORMATTERS.get(inv.name)
            if not formatter:
                continue
            text = formatter(block["rows"])
            if text:
                graph_blocks.append(text)
            # Track which step uids this graph result already covers,
            # so vector results don't duplicate them.
            for row in block["rows"]:
                if row.get("proc_id") and row.get("step_number") is not None:
                    graph_covered_uids.add(row["proc_id"] + ":" + str(row["step_number"]))
                if row.get("id") and "PRC-" in str(row.get("id", "")):
                    # This is a procedure result; mark all its steps as covered
                    for s in row.get("steps", []):
                        n = s.get("n")
                        if n is not None:
                            graph_covered_uids.add(row["id"] + ":" + str(n))

        # 2. Vector retrieval
        docs, metas = self._vector_search(question)
        vector_block_lines = []
        included_uids = []
        for doc, meta in zip(docs, metas):
            uid = meta["procedure_id"] + ":" + str(meta["step_number"])
            if uid in graph_covered_uids:
                continue
            vector_block_lines.append("- " + doc)
            included_uids.append(uid)

        vector_block = ""
        if vector_block_lines:
            vector_block = ("## Additional semantically-similar steps\n"
                            + "\n".join(vector_block_lines))

        # 3. Combine, respect token budget
        all_blocks = graph_blocks + ([vector_block] if vector_block else [])
        formatted = "\n\n".join(all_blocks)

        # Soft truncation if over budget
        budget_chars = self.token_budget * CHARS_PER_TOKEN
        if len(formatted) > budget_chars:
            formatted = formatted[:budget_chars] + "\n\n[... context truncated to fit token budget]"

        return HybridContext(
            question=question,
            formatted_context=formatted,
            graph_invocations=[inv.name for inv in invocations],
            vector_uids=included_uids,
            char_count=len(formatted),
            est_tokens=len(formatted) // CHARS_PER_TOKEN,
        )


# Quick CLI: pretty-print hybrid retrieval for a few questions
if __name__ == "__main__":
    PROBES = [
        "Tell me about PRC-014",
        "What torque should I use for M20 bolts?",
        "Which procedures require electrical certification?",
        "Show me PRC-002 step 3",
        "what defects can occur during compressor service",
        "how do I replace a pump bearing",
    ]

    retriever = HybridRetriever()
    try:
        for q in PROBES:
            print()
            print("=" * 70)
            print("Q: " + q)
            ctx = retriever.retrieve(q)
            print("Templates fired: " + str(ctx.graph_invocations))
            print("Vector uids included: " + str(ctx.vector_uids))
            print("Context size: " + str(ctx.char_count) + " chars / "
                  + "~" + str(ctx.est_tokens) + " tokens")
            print()
            print(ctx.formatted_context[:1200])
            if len(ctx.formatted_context) > 1200:
                print("... [truncated for display, full length above]")
    finally:
        retriever.close()
