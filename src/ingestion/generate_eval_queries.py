"""Generate evaluation Q/A pairs from the live Neo4j graph.

Three batches: 20 single-hop, 20 multi-hop, 10 arithmetic.
Each question has the natural-language form, the ground-truth Cypher, the
expected answer, and a match mode for scoring.

Cost: roughly $0.20 with Claude Sonnet 4.5.
"""
import json
import os
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic"
EVAL_OUT = Path(__file__).resolve().parents[2] / "data" / "eval_queries_draft.jsonl"

CATALOG_FILE = DATA_DIR / "catalog.json"
PROCEDURES_FILE = DATA_DIR / "procedures.json"
TORQUE_SPECS_FILE = DATA_DIR / "torque_specs.json"

MODEL = "claude-sonnet-4-5-20250929"

BATCHES = [
    {
        "category": "single_hop",
        "count": 20,
        "id_start": 1,
        "description": (
            "Single-hop questions answerable by traversing 1-2 relationships. "
            "Examples: \"What tool is used in step 5 of PRC-007?\", "
            "\"Which certifications does the centrifugal pump impeller inspection require?\", "
            "\"What torque pattern applies to TS-003?\". "
            "Mostly graph-internal phrasing using procedure IDs or full procedure titles. "
            "About 30 percent should reference procedures by their title rather than ID."
        ),
    },
    {
        "category": "multi_hop",
        "count": 20,
        "id_start": 21,
        "description": (
            "Multi-hop questions requiring 3+ traversal steps OR set intersections. "
            "Examples: \"Which procedures have a prerequisite that itself requires CERT-6?\", "
            "\"List procedures where any step uses both a torque wrench AND a M20 fastener.\", "
            "\"Which defects appear in steps that have a torque spec above 200 Nm?\". "
            "These should be questions a vector-only RAG would struggle with - they "
            "require structural reasoning over multiple node types."
        ),
    },
    {
        "category": "arithmetic",
        "count": 10,
        "id_start": 41,
        "description": (
            "Aggregation questions requiring count/sum/average/max/min over filtered "
            "subgraphs. Examples: \"How many total steps across all induction motor procedures?\", "
            "\"What is the average estimated duration of procedures requiring Pressure Systems "
            "Inspector certification?\", \"What is the highest torque value used in any "
            "centrifugal pump procedure?\". All answers should be numeric or a small ordered list."
        ),
    },
]


def build_prompt(category, count, id_start, id_end, description, data_summary):
    """Build the prompt with f-string concatenation. No .format() so curly braces are safe."""
    schema_block = """GRAPH SCHEMA:

Nodes:
  Procedure (id, title, equipment_type, criticality, est_duration_min)
  Step (uid, step_number, instruction, est_duration_sec, hold_point)
  Tool (id, name, type, calibration_required)
  Component (id, name, category, material)
  Certification (id, name, level, scope)
  Defect (id, name, severity, escalation)
  TorqueSpec (id, target_nm, tolerance_nm, pattern, conditions)

Relationships:
  (Procedure) -[:HAS_STEP order]-> (Step)
  (Procedure) -[:REQUIRES_CERT]-> (Certification)
  (Procedure) -[:PRECEDED_BY]-> (Procedure)
  (Step) -[:USES_TOOL]-> (Tool)
  (Step) -[:USES_COMPONENT]-> (Component)
  (Step) -[:HAS_TORQUE_SPEC]-> (TorqueSpec)
  (Step) -[:CAN_PRODUCE_DEFECT]-> (Defect)
  (TorqueSpec) -[:APPLIES_TO]-> (Component)

Note: in your ground_truth_cypher, use real Cypher node syntax with curly braces,
e.g. (p:Procedure {id: \"PRC-001\"}). The schema above is shown without braces only
so this prompt parses cleanly.
"""

    task_block = (
        "YOUR TASK:\n\n"
        f"Generate exactly {count} questions of category \"{category}\". {description}\n\n"
        "OUTPUT: a JSON object with one key \"questions\" containing exactly "
        f"{count} entries.\n"
        "Each entry must have these fields:\n\n"
        f"  id: \"Q-{id_start:03d}\" through \"Q-{id_end:03d}\"\n"
        f"  category: \"{category}\"\n"
        "  question: natural language (one sentence)\n"
        "  ground_truth_cypher: the Cypher query that computes the answer. Must return\n"
        "    a single column with the answer rows. Must be valid Cypher 5.x.\n"
        "  expected_answer: the actual answer the Cypher should return, derived from\n"
        "    the domain data summary. Be precise.\n"
        "  expected_match_mode: one of:\n"
        "    - \"exact\" for single string/number answers\n"
        "    - \"set\" for unordered collections (lists of names, IDs)\n"
        "    - \"numeric_within_tolerance\" for floats where +/- 1 percent is acceptable\n"
        "  reasoning: 1 sentence explaining what this question tests.\n\n"
        "CRITICAL:\n"
        "- Only reference real IDs from the data summary (PRC-XXX, T-XXX, C-XXX,\n"
        "  CERT-X, D-XXX, TS-XXX). Do not invent IDs.\n"
        "- Verify your expected_answer against the data summary before including it.\n"
        "  If you cannot determine the answer with certainty, do not include the question.\n"
        "- Vary the surface form: some questions reference procedures by ID, others by\n"
        "  title or equipment type. Some reference tools by ID, others by name.\n"
        "- For \"set\" mode, expected_answer must be a JSON array.\n"
        "- Numerical answers must be exact integers or floats, not ranges.\n\n"
        "Return only the JSON object. No markdown fences, no commentary."
    )

    return (
        "You are generating evaluation Q/A pairs for a GraphRAG benchmark. "
        "The graph is loaded into Neo4j. Write questions whose answers can "
        "be computed deterministically by running Cypher against the graph.\n\n"
        + schema_block
        + "\nDOMAIN DATA SUMMARY:\n\n"
        + data_summary
        + "\n\n"
        + task_block
    )


def make_data_summary(catalog, procs, specs):
    lines = []
    lines.append("PROCEDURES (id | title | equipment | criticality | duration_min "
                 "| certs | prereqs | num_steps):")
    for p in procs:
        lines.append(
            "  " + p["id"] + " | " + p["title"] + " | " + p["equipment_type"]
            + " | " + p["criticality"] + " | " + str(p["est_duration_min"])
            + " min | certs=" + str(p["required_certifications"])
            + " | prereq=" + str(p["preceded_by"])
            + " | " + str(len(p["steps"])) + " steps"
        )

    lines.append("")
    lines.append("STEPS (procedure step_no: hp=hold_point tools=... comps=... ts=... defects=...):")
    for p in procs:
        for s in p["steps"]:
            lines.append(
                "  " + p["id"] + " step " + str(s["step_number"])
                + ": hp=" + str(s["hold_point"])
                + " tools=" + str(s["tools_used"])
                + " comps=" + str(s["components_used"])
                + " ts=" + str(s.get("torque_spec_id"))
                + " defects=" + str(s["potential_defects"])
            )

    lines.append("")
    lines.append("TORQUE_SPECS (id | target_nm +/- tolerance | pattern | applies_to):")
    for s in specs:
        lines.append(
            "  " + s["id"] + " | " + str(s["target_nm"])
            + " Nm +/- " + str(s["tolerance_nm"])
            + " | " + s["pattern"] + " | " + s["applies_to_component"]
        )

    lines.append("")
    lines.append("TOOLS:")
    for t in catalog["tools"]:
        lines.append(
            "  " + t["id"] + " | " + t["name"]
            + " [" + t["type"] + "] cal_req=" + str(t["calibration_required"])
        )

    lines.append("")
    lines.append("COMPONENTS:")
    for c in catalog["components"]:
        lines.append(
            "  " + c["id"] + " | " + c["name"]
            + " [" + c["category"] + "] " + c["material"]
        )

    lines.append("")
    lines.append("CERTIFICATIONS:")
    for c in catalog["certifications"]:
        lines.append(
            "  " + c["id"] + " | " + c["name"] + " (level " + str(c["level"]) + ")"
        )

    lines.append("")
    lines.append("DEFECTS:")
    for d in catalog["defects"]:
        lines.append(
            "  " + d["id"] + " | " + d["name"] + " (" + d["severity"] + ")"
        )

    return "\n".join(lines)


def main():
    catalog = json.loads(CATALOG_FILE.read_text())
    procs = json.loads(PROCEDURES_FILE.read_text())["procedures"]
    specs = json.loads(TORQUE_SPECS_FILE.read_text())["torque_specs"]

    data_summary = make_data_summary(catalog, procs, specs)
    print("[*] Data summary length: " + str(len(data_summary)) + " chars")

    if EVAL_OUT.exists():
        print("[!] " + str(EVAL_OUT) + " already exists.")
        if input("Overwrite? (y/N): ").strip().lower() != "y":
            sys.exit(0)

    client = anthropic.Anthropic()
    all_questions = []
    total_input = 0
    total_output = 0

    for batch in BATCHES:
        category = batch["category"]
        count = batch["count"]
        id_start = batch["id_start"]
        id_end = id_start + count - 1
        description = batch["description"]

        prompt = build_prompt(category, count, id_start, id_end, description, data_summary)

        print("\n[*] Generating " + str(count) + " " + category
              + " questions (Q-" + str(id_start).zfill(3)
              + "..Q-" + str(id_end).zfill(3) + ") ...")
        t0 = time.perf_counter()
        response = client.messages.create(
            model=MODEL,
            max_tokens=8000,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed = time.perf_counter() - t0
        total_input += response.usage.input_tokens
        total_output += response.usage.output_tokens
        print("    {:.1f}s, tokens {} in / {} out".format(
            elapsed, response.usage.input_tokens, response.usage.output_tokens))

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0]
            raw = raw.strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            print("[FAIL] " + category + " JSON parse error: " + str(e))
            (DATA_DIR / ("eval_" + category + "_raw.txt")).write_text(raw)
            sys.exit(1)

        questions = data.get("questions", [])
        print("    Got " + str(len(questions)) + " questions")
        all_questions.extend(questions)

    print("\n[*] TOTAL: " + str(len(all_questions)) + " questions")
    print("[*] Total tokens: " + str(total_input) + " in / " + str(total_output) + " out")
    cost = (total_input * 3 + total_output * 15) / 1_000_000
    print("[*] Approx cost: ${:.3f}".format(cost))

    EVAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_OUT, "w") as f:
        for q in all_questions:
            f.write(json.dumps(q) + "\n")
    print("[OK] Wrote " + str(EVAL_OUT))


if __name__ == "__main__":
    main()
