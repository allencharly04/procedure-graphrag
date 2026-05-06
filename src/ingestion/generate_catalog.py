"""Generate the catalog: tools, components, certifications, defects.

This is Pass 1 of synthetic data generation. The output of this script is
read by generate_procedures.py (Pass 2), which references these catalog IDs
when generating procedures.

Cost: roughly $0.05 with Claude Sonnet 4.5.
"""
import json
import os
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "catalog.json"

MODEL = "claude-sonnet-4-5-20250929"

PROMPT = """You are generating a synthetic catalog of entities for an industrial
maintenance knowledge graph. The graph will support a GraphRAG benchmark.

Generate exactly the counts specified, with realistic, internally consistent entries.
Do not number tools sequentially by category - mix them up so IDs are diverse.

Required output: a single JSON object with the keys "tools", "components",
"certifications", "defects". No markdown, no commentary, just valid JSON.

SCHEMA AND COUNTS:

tools (30 entries):
  id: "T-001" through "T-030"
  name: realistic full name including model or specification where appropriate
        (e.g., "Calibrated Torque Wrench, 10-50 Nm", "Digital Multimeter, Fluke 87V")
  type: one of [torque_wrench, diagnostic, hand_tool, power_tool, lifting,
        measurement, lubrication, ppe, lockout]
  calibration_required: boolean (true for torque wrenches, multimeters, pressure
        gauges; false for hand tools and PPE)

components (60 entries):
  id: "C-001" through "C-060"
  name: realistic part name with size or grade (e.g., "Mechanical Seal Type-A 50mm",
        "Stainless Bolt M12x80 Grade 8.8", "Pump Bearing 6205-2RS")
  category: one of [pump_part, valve_part, motor_part, heat_exchanger_part,
        compressor_part, instrumentation_part, fastener, gasket, lubricant,
        consumable]
  material: one of [stainless_steel, carbon_steel, chrome_steel, brass, bronze,
        ptfe, viton, epdm, carbon_ceramic, copper, aluminum, n/a]

  IMPORTANT: include at least 10 fasteners (bolts, studs, nuts) with explicit
  metric sizing in the name, since torque specs will reference these.

certifications (10 entries):
  id: "CERT-1" through "CERT-10"
  name: realistic title (e.g., "Mechanical Maintenance Level 2",
        "Electrical Safety LV Authorization", "Pressure Systems Inspector")
  level: 1, 2, or 3 (1 = supervised, 3 = expert/inspector)
  scope: one-sentence description of what this cert authorizes

  Cover: mechanical (3 levels), electrical (2 levels), pressure systems (2 levels),
  instrumentation (1 level), confined space (1), lifting operations (1).

defects (15 entries):
  id: "D-001" through "D-015"
  name: realistic defect mode (e.g., "Improper Bolt Torque Sequence",
        "Seal Face Misalignment", "Cable Insulation Damage")
  severity: one of [low, medium, high, critical]
  escalation: who handles it (e.g., "supervisor sign-off",
        "QA inspector + engineering review", "stop work, notify safety officer")

Return only valid JSON matching this structure. Validate IDs are sequential and unique."""


def main() -> None:
    if OUTPUT_FILE.exists():
        print(f"[!] {OUTPUT_FILE} already exists.")
        response = input("Overwrite? (y/N): ").strip().lower()
        if response != "y":
            print("Aborted.")
            sys.exit(0)

    client = anthropic.Anthropic()
    print(f"[*] Generating catalog with {MODEL} ...")
    t0 = time.perf_counter()

    response = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        messages=[{"role": "user", "content": PROMPT}],
    )
    elapsed = time.perf_counter() - t0

    raw = response.content[0].text.strip()
    print(f"[*] Generated in {elapsed:.1f}s")
    print(f"[*] Tokens: {response.usage.input_tokens} in / {response.usage.output_tokens} out")

    # Strip optional markdown fences
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

    try:
        catalog = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[FAIL] JSON parse error: {e}")
        debug_path = OUTPUT_DIR / "catalog_raw.txt"
        debug_path.write_text(raw)
        print(f"[*] Raw output saved to {debug_path} for debugging")
        sys.exit(1)

    # Validate counts
    expected = {"tools": 30, "components": 60, "certifications": 10, "defects": 15}
    actual = {k: len(catalog.get(k, [])) for k in expected}
    print("[*] Counts:", actual)
    for key, count in expected.items():
        if actual.get(key, 0) != count:
            print(f"[!] WARNING: {key} expected {count}, got {actual.get(key, 0)}")

    # Validate ID uniqueness within each category
    for key in expected:
        ids = [item.get("id", "") for item in catalog.get(key, [])]
        if len(ids) != len(set(ids)):
            print(f"[!] WARNING: {key} has duplicate IDs")

    OUTPUT_FILE.write_text(json.dumps(catalog, indent=2))
    print(f"[OK] Wrote {OUTPUT_FILE}")
    print(f"[OK] File size: {OUTPUT_FILE.stat().st_size:,} bytes")

    # Show a few samples for quick eyeball check
    print("\n[*] Sample entries:")
    for cat in ["tools", "components", "certifications", "defects"]:
        items = catalog.get(cat, [])
        if items:
            print(f"  {cat}[0]: {items[0]}")


if __name__ == "__main__":
    main()
