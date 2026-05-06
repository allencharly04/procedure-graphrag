"""Run the full v0 baseline: 48 eval questions through the GraphRAG pipeline.

For each question:
  - Run the pipeline
  - Score the answer against the ground-truth expected answer
  - Record per-stage latency

Outputs:
  benchmarks/results/baseline.json - per-question results + summary stats

This is THE baseline. Every Phase 4 optimization gets compared to these numbers.
"""
import json
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pipeline import Pipeline, PipelineResult


ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = ROOT / "data" / "eval_queries.jsonl"
OUT_DIR = ROOT / "benchmarks" / "results"
OUT_FILE = OUT_DIR / "baseline.json"


# -----------------------
# Scoring
# -----------------------

ID_PATTERN = re.compile(r"\b(?:PRC|TS|CERT|D|T|C)-\d{1,3}\b")
NUM_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def extract_ids(text: str) -> List[str]:
    """Extract any IDs from a free-form LLM response."""
    return list(set(m.upper() for m in ID_PATTERN.findall(text)))


def extract_first_number(text: str) -> Any:
    """Pull the first number from text, integer if whole."""
    m = NUM_PATTERN.search(text)
    if not m:
        return None
    val = float(m.group(0))
    if val.is_integer():
        return int(val)
    return val


def score(predicted: str, expected: Any, mode: str) -> Dict[str, Any]:
    """Score a single answer. Returns dict with: correct (bool), kind (str), notes (str)."""
    pred_text = predicted.strip()

    if mode == "set":
        pred_ids = sorted(set(extract_ids(pred_text)))
        if isinstance(expected, list):
            expected_norm = sorted(set(str(x).strip().upper() for x in expected))
        else:
            expected_norm = [str(expected).strip().upper()]

        if pred_ids == expected_norm:
            return {"correct": True, "kind": "set_exact", "notes": ""}

        # Lenient: subset match (we got all expected IDs, even with extra noise)
        missing = [x for x in expected_norm if x not in pred_ids]
        extra = [x for x in pred_ids if x not in expected_norm]
        if not missing:
            return {"correct": True, "kind": "set_superset",
                    "notes": "extra_ids=" + str(extra)}
        return {"correct": False, "kind": "set_mismatch",
                "notes": "missing=" + str(missing[:3]) + " extra=" + str(extra[:3])}

    elif mode == "numeric_within_tolerance":
        pred_num = extract_first_number(pred_text)
        try:
            exp_num = float(expected)
        except (TypeError, ValueError):
            return {"correct": False, "kind": "expected_not_numeric", "notes": ""}
        if pred_num is None:
            return {"correct": False, "kind": "no_number_in_response", "notes": ""}
        if exp_num == 0:
            ok = pred_num == 0
        else:
            ok = abs(pred_num - exp_num) / abs(exp_num) <= 0.01
        return {"correct": ok, "kind": "numeric",
                "notes": "pred=" + str(pred_num) + " exp=" + str(exp_num)}

    else:  # exact
        # Try numeric exact first
        pred_num = extract_first_number(pred_text)
        try:
            exp_num = float(expected)
            if pred_num is not None and pred_num == exp_num:
                return {"correct": True, "kind": "exact_numeric", "notes": ""}
        except (TypeError, ValueError):
            pass

        # Then string match (case-insensitive, trimmed)
        pred_lower = pred_text.lower().strip()
        exp_lower = str(expected).lower().strip()
        if pred_lower == exp_lower:
            return {"correct": True, "kind": "exact_string", "notes": ""}

        # Substring containment for verbose answers
        if exp_lower in pred_lower:
            return {"correct": True, "kind": "exact_substring",
                    "notes": "expected appeared as substring"}

        return {"correct": False, "kind": "exact_mismatch",
                "notes": "pred=" + repr(pred_text[:80]) + " exp=" + repr(str(expected)[:80])}


# -----------------------
# Main
# -----------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load eval set
    with open(EVAL_FILE) as f:
        questions = [json.loads(line) for line in f]
    print("[*] Loaded " + str(len(questions)) + " eval questions")

    # Initialize pipeline
    print("[*] Initializing pipeline ...")
    pipeline = Pipeline()

    # Warmup so timing is fair
    print("[*] Warming up model ...")
    _ = pipeline.answer("Reply with: ready")
    print("    warmup done")

    # Run each question
    results = []
    t_start = time.perf_counter()
    for i, q in enumerate(questions, start=1):
        try:
            r = pipeline.answer(q["question"])
            sc = score(r.answer, q["expected_answer"], q["expected_match_mode"])
            verdict = "OK  " if sc["correct"] else "FAIL"
            print("  [{}/{}] {} {} {} ({:.0f}ms)".format(
                str(i).rjust(2), len(questions), verdict, q["id"], q["category"],
                r.latency.total_ms))
            if not sc["correct"]:
                print("       {}".format(sc["notes"][:100]))

            results.append({
                "id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "expected": q["expected_answer"],
                "expected_mode": q["expected_match_mode"],
                "predicted": r.answer,
                "correct": sc["correct"],
                "score_kind": sc["kind"],
                "score_notes": sc["notes"],
                "context_chars": r.context_chars,
                "context_tokens": r.context_tokens,
                "prompt_chars": r.prompt_chars,
                "graph_invocations": r.graph_invocations,
                "vector_uids_kept": r.vector_uids,
                "model": r.model,
                "latency": asdict(r.latency),
                "prompt_eval_count": r.prompt_eval_count,
                "eval_count": r.eval_count,
            })
        except Exception as e:
            print("  [{}/{}] ERROR {}: {}".format(i, len(questions), q["id"], str(e)[:120]))
            results.append({
                "id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "expected": q["expected_answer"],
                "error": str(e)[:200],
            })
    total_elapsed = time.perf_counter() - t_start

    pipeline.close()

    # Aggregate stats
    valid = [r for r in results if "error" not in r]
    correct = [r for r in valid if r["correct"]]

    by_cat = {}
    for r in valid:
        cat = r["category"]
        by_cat.setdefault(cat, {"total": 0, "correct": 0, "total_ms": 0.0})
        by_cat[cat]["total"] += 1
        if r["correct"]:
            by_cat[cat]["correct"] += 1
        by_cat[cat]["total_ms"] += r["latency"]["total_ms"] or 0

    total_ms_list = sorted([r["latency"]["total_ms"] for r in valid if r["latency"]["total_ms"]])
    n = len(total_ms_list)
    p50 = total_ms_list[n // 2] if n else 0
    p95 = total_ms_list[int(n * 0.95)] if n else 0
    mean = sum(total_ms_list) / n if n else 0

    decode_ms_list = [r["latency"]["llm_decode_ms"] for r in valid if r["latency"]["llm_decode_ms"]]
    decode_mean = sum(decode_ms_list) / len(decode_ms_list) if decode_ms_list else 0

    prefill_ms_list = [r["latency"]["llm_prompt_eval_ms"] for r in valid if r["latency"]["llm_prompt_eval_ms"]]
    prefill_mean = sum(prefill_ms_list) / len(prefill_ms_list) if prefill_ms_list else 0

    retrieve_ms_list = [r["latency"]["retrieve_graph_ms"] for r in valid if r["latency"]["retrieve_graph_ms"]]
    retrieve_mean = sum(retrieve_ms_list) / len(retrieve_ms_list) if retrieve_ms_list else 0

    summary = {
        "model": "llama3.1:latest",
        "total_questions": len(questions),
        "valid_runs": len(valid),
        "errors": len(results) - len(valid),
        "correct": len(correct),
        "accuracy": len(correct) / len(valid) if valid else 0,
        "wall_clock_seconds": total_elapsed,
        "latency_ms": {
            "p50_total": p50,
            "p95_total": p95,
            "mean_total": mean,
            "mean_retrieve": retrieve_mean,
            "mean_prefill": prefill_mean,
            "mean_decode": decode_mean,
        },
        "by_category": {
            cat: {
                "total": v["total"],
                "correct": v["correct"],
                "accuracy": v["correct"] / v["total"] if v["total"] else 0,
                "mean_total_ms": v["total_ms"] / v["total"] if v["total"] else 0,
            } for cat, v in by_cat.items()
        },
    }

    # Persist
    OUT_FILE.write_text(json.dumps({"summary": summary, "results": results}, indent=2))

    print()
    print("=" * 70)
    print("BASELINE SUMMARY")
    print("=" * 70)
    print("Total questions:      " + str(summary["total_questions"]))
    print("Valid runs:           " + str(summary["valid_runs"]))
    print("Correct:              " + str(summary["correct"]))
    print("Accuracy:             {:.1%}".format(summary["accuracy"]))
    print()
    print("Wall-clock:           {:.1f}s".format(summary["wall_clock_seconds"]))
    print()
    print("Latency (ms):")
    print("  p50 total:          {:.0f}".format(p50))
    print("  p95 total:          {:.0f}".format(p95))
    print("  mean total:         {:.0f}".format(mean))
    print("  mean retrieve:      {:.0f}".format(retrieve_mean))
    print("  mean prefill:       {:.0f}".format(prefill_mean))
    print("  mean decode:        {:.0f}".format(decode_mean))
    print()
    print("By category:")
    for cat, v in summary["by_category"].items():
        print("  {:<14s} {}/{} ({:.0%})  mean={:.0f}ms".format(
            cat, v["correct"], v["total"], v["accuracy"], v["mean_total_ms"]))
    print()
    print("[OK] Written to " + str(OUT_FILE))


if __name__ == "__main__":
    main()
