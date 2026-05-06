"""Generate procedures referencing the catalog from Pass 1.

Calls Claude Sonnet six times, batching 5 procedures per call (30 total).
Each batch is told what equipment types to cover, what the catalog contains,
and which procedure IDs to use. Procedures reference catalog IDs (T-XXX, C-XXX,
CERT-X, D-XXX), and a torque_specs section is generated alongside.

Cost: roughly $0.30 with Claude Sonnet 4.5.
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
CATALOG_FILE = DATA_DIR / "catalog.json"
OUT_PROCEDURES = DATA_DIR / "procedures.json"
OUT_TORQUE_SPECS = DATA_DIR / "torque_specs.json"

MODEL = "claude-sonnet-4-5-20250929"

BATCHES = [
    {"equipment": "centrifugal_pump", "count": 5, "id_start": 1},
    {"equipment": "centrifugal_pump", "count": 3, "id_start": 6,
     "extra": "Mix in 2 globe/gate valve procedures (positions 4 and 5 of this batch)."},
    {"equipment": "globe_or_gate_valve", "count": 5, "id_start": 9,
     "extra": "Cover both globe and gate valve types. Include packing replacement and trim overhaul."},
    {"equipment": "induction_motor", "count": 6, "id_start": 14,
     "extra": "Cover bearing replacement, terminal box re-wiring, alignment to driven equipment, "
              "insulation resistance testing, vibration analysis, and decoupling for shop overhaul."},
    {"equipment": "shell_tube_heat_exchanger", "count": 4, "id_start": 20,
     "extra": "Include tube bundle removal, tube cleaning, gasket replacement, hydrostatic testing."},
    {"equipment": "reciprocating_compressor_or_instrumentation", "count": 7, "id_start": 24,
     "extra": "Generate 3 reciprocating compressor procedures (valve replacement, piston ring change, "
              "lubrication system service) and 4 instrumentation procedures (pressure transmitter "
              "calibration, temperature transmitter loop check, control valve stroke test, "
              "differential pressure transmitter zero/span)."},
]

PROMPT_TEMPLATE = """You are generating 5 industrial maintenance procedures for a synthetic
knowledge graph. Reference the provided catalog by ID; do not invent new tool, component,
certification, or defect IDs.

CATALOG (use these IDs only):
{catalog_summary}

EQUIPMENT FOCUS for this batch: {equipment}
PROCEDURE IDs to use: PRC-{id_start:03d} through PRC-{id_end:03d}
{extra}

OUTPUT: a single JSON object with two top-level keys: "procedures" and "torque_specs".

procedures (exactly {count} entries):
  id: "PRC-XXX" using the IDs assigned above
  title: short imperative title (e.g., "Centrifugal Pump Bearing Replacement")
  equipment_type: one of [centrifugal_pump, globe_valve, gate_valve, induction_motor,
                  shell_tube_heat_exchanger, reciprocating_compressor, instrumentation]
  criticality: one of [low, medium, high, critical]
  est_duration_min: integer, total wall-clock time
  required_certifications: list of CERT-X IDs (1-3 entries; mechanical work needs CERT-1/2/3,
                           electrical needs CERT-4/5, pressure needs CERT-6/7, etc.)
  preceded_by: list of PRC-XXX IDs that must be completed first.
               IMPORTANT: about 30% of procedures should have one prerequisite. Use only
               IDs that have already been generated in earlier batches (PRC-001 through
               PRC-{prev_max:03d}); leave empty list if no prior procedure fits.
  steps: list of 4-12 step objects, with the typical procedure averaging 7 steps:

    step_number: integer starting at 1
    instruction: one-sentence imperative instruction (e.g., "Torque flange bolts to spec
                 in star pattern.")
    est_duration_sec: integer
    hold_point: boolean (true for safety-critical or QA-witness points)
    tools_used: list of T-XXX IDs (can be empty for purely procedural steps like waiting)
    components_used: list of C-XXX IDs (parts consumed/handled in this step)
    torque_spec_id: a TS-XXX ID if this step requires torquing fasteners, else null
    potential_defects: list of D-XXX IDs (defects that can occur if step done wrong; 0-2 per step)

torque_specs (one entry per unique torque application referenced by steps above):
  id: "TS-XXX" with a unique number across this batch and all prior batches.
      For this batch, use TS-{ts_start:03d} onwards.
  applies_to_component: a C-XXX fastener ID
  target_nm: target torque in Newton-meters (realistic for the bolt size)
  tolerance_nm: tolerance band (e.g., 5)
  pattern: one of [star, sequential, cross, single]
  conditions: short string describing when this spec applies (e.g.,
              "lubricated threads, room temperature")

  IMPORTANT: only generate torque specs for fastener components (those with
  category=fastener in the catalog). Realistic torque ranges:
    M10 -> 40-55 Nm
    M12 -> 70-95 Nm
    M16 -> 170-220 Nm
    M20 -> 340-420 Nm
    M24 -> 580-700 Nm

Return only valid JSON. No markdown fences, no commentary."""


def make_catalog_summary(catalog):
    lines = []
    lines.append("TOOLS (id, name, type):")
    for t in catalog["tools"]:
        lines.append("  " + t["id"] + ": " + t["name"] + " [" + t["type"] + "]")
    lines.append("\nCOMPONENTS (id, name, category):")
    for c in catalog["components"]:
        lines.append("  " + c["id"] + ": " + c["name"] + " [" + c["category"] + "]")
    lines.append("\nCERTIFICATIONS (id, name, level):")
    for c in catalog["certifications"]:
        lines.append("  " + c["id"] + ": " + c["name"] + " (L" + str(c["level"]) + ")")
    lines.append("\nDEFECTS (id, name, severity):")
    for d in catalog["defects"]:
        lines.append("  " + d["id"] + ": " + d["name"] + " [" + d["severity"] + "]")
    return "\n".join(lines)


def main():
    if not CATALOG_FILE.exists():
        print("[FAIL] Catalog not found at " + str(CATALOG_FILE))
        sys.exit(1)

    catalog = json.loads(CATALOG_FILE.read_text())
    catalog_summary = make_catalog_summary(catalog)
    print("[*] Catalog loaded: " + str(len(catalog["tools"])) + " tools, "
          + str(len(catalog["components"])) + " components, "
          + str(len(catalog["certifications"])) + " certifications, "
          + str(len(catalog["defects"])) + " defects")

    if OUT_PROCEDURES.exists():
        print("[!] " + str(OUT_PROCEDURES) + " already exists.")
        if input("Overwrite? (y/N): ").strip().lower() != "y":
            sys.exit(0)

    client = anthropic.Anthropic()

    all_procedures = []
    all_torque_specs = []
    ts_counter = 1
    total_input = 0
    total_output = 0

    for i, batch in enumerate(BATCHES, start=1):
        id_start = batch["id_start"]
        count = batch["count"]
        id_end = id_start + count - 1
        prev_max = id_start - 1

        prompt = PROMPT_TEMPLATE.format(
            catalog_summary=catalog_summary,
            equipment=batch["equipment"],
            id_start=id_start,
            id_end=id_end,
            count=count,
            prev_max=prev_max if prev_max > 0 else 0,
            ts_start=ts_counter,
            extra=batch.get("extra", ""),
        )

        print("\n[*] Batch " + str(i) + "/6: " + batch["equipment"]
              + ", PRC-{:03d}..PRC-{:03d}".format(id_start, id_end))
        t0 = time.perf_counter()
        response = client.messages.create(
            model=MODEL,
            max_tokens=8000,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed = time.perf_counter() - t0
        total_input += response.usage.input_tokens
        total_output += response.usage.output_tokens
        print("    Generated in {:.1f}s, tokens: {} in / {} out".format(
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
            print("[FAIL] Batch " + str(i) + " JSON parse error: " + str(e))
            (DATA_DIR / ("batch_" + str(i) + "_raw.txt")).write_text(raw)
            print("[*] Raw saved to batch_" + str(i) + "_raw.txt")
            sys.exit(1)

        procs = data.get("procedures", [])
        specs = data.get("torque_specs", [])
        print("    Got " + str(len(procs)) + " procedures, " + str(len(specs)) + " torque specs")
        all_procedures.extend(procs)
        all_torque_specs.extend(specs)
        ts_counter += len(specs)

    print("\n[*] TOTAL: " + str(len(all_procedures)) + " procedures, "
          + str(len(all_torque_specs)) + " torque specs")
    print("[*] Total tokens: " + str(total_input) + " in / " + str(total_output) + " out")
    cost = (total_input * 3 + total_output * 15) / 1_000_000
    print("[*] Approx cost: ${:.3f}".format(cost))

    OUT_PROCEDURES.write_text(json.dumps({"procedures": all_procedures}, indent=2))
    OUT_TORQUE_SPECS.write_text(json.dumps({"torque_specs": all_torque_specs}, indent=2))
    print("[OK] Wrote " + str(OUT_PROCEDURES))
    print("[OK] Wrote " + str(OUT_TORQUE_SPECS))


if __name__ == "__main__":
    main()
