"""Run a chained-pipeline sweep configuration.

Usage:
    python benchmarks/run_sweep_chained.py <model_name> [config_label]

Example:
    python benchmarks/run_sweep_chained.py llama3.1:latest chained-8b

Writes:
    benchmarks/results/sweep_<safe_label>.json

Identical scoring to run_sweep.py, but uses ChainedPipeline (3 LLM calls per question).
"""
import json
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pipeline_chained import ChainedPipeline


ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = ROOT / "data" / "eval_queries.jsonl"
OUT_DIR = ROOT / "benchmarks" / "results"


ID_PATTERN = re.compile(r"\b(?:PRC|TS|CERT|D|T|C)-\d{1,3}\b")
NUM_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def extract_ids(text):
    return list(set(m.upper() for m in ID_PATTERN.findall(text)))


def extract_first_number(text):
    m = NUM_PATTERN.search(text)
    if not m:
        return None
    val = float(m.group(0))
    if val.is_integer():
        return int(val)
    return val


def score(predicted, expected, mode):
    pred_text = predicted.strip()

    if mode == "set":
        pred_ids = sorted(set(extract_ids(pred_text)))
        if isinstance(expected, list):
            expected_norm = sorted(set(str(x).strip().upper() for x in expected))
        else:
            expected_norm = [str(expected).strip().upper()]

        if pred_ids == expected_norm:
            return {"correct": True, "kind": "set_exact", "notes": ""}
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

    else:
        pred_num = extract_first_number(pred_text)
        try:
            exp_num = float(expected)
            if pred_num is not None and pred_num == exp_num:
                return {"correct": True, "kind": "exact_numeric", "notes": ""}
        except (TypeError, ValueError):
            pass

        pred_lower = pred_text.lower().strip()
        exp_lower = str(expected).lower().strip()
        if pred_lower == exp_lower:
            return {"correct": True, "kind": "exact_string", "notes": ""}
        if exp_lower in pred_lower:
            return {"correct": True, "kind": "exact_substring",
                    "notes": "expected appeared as substring"}
        return {"correct": False, "kind": "exact_mismatch",
                "notes": "pred=" + repr(pred_text[:80]) + " exp=" + repr(str(expected)[:80])}


def safe_filename(label):
    return label.replace(":", "_").replace("/", "_")


def main():
    if len(sys.argv) < 2:
        print("Usage: python benchmarks/run_sweep_chained.py <model_name> [config_label]")
        sys.exit(1)

    model_name = sys.argv[1]
    config_label = sys.argv[2] if len(sys.argv) > 2 else ("chained_" + model_name)
    out_file = OUT_DIR / ("sweep_" + safe_filename(config_label) + ".json")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(EVAL_FILE) as f:
        questions = [json.loads(line) for line in f]
    print("[*] Config: " + config_label)
    print("[*] Model: " + model_name)
    print("[*] Strategy: chained (decompose -> filter -> answer)")
    print("[*] Loaded " + str(len(questions)) + " eval questions")

    pipeline = ChainedPipeline(model=model_name)

    print("[*] Warming up model ...")
    _ = pipeline.answer("Reply with: ready")
    print("    warmup done")

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
            results.append({
                "id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "expected": q["expected_answer"],
                "expected_mode": q["expected_match_mode"],
                "predicted": r.answer,
                "sub_questions": r.sub_questions,
                "filtered": r.filtered,
                "correct": sc["correct"],
                "score_kind": sc["kind"],
                "score_notes": sc["notes"],
                "context_chars": r.context_chars,
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
                "error": str(e)[:200],
            })
    total_elapsed = time.perf_counter() - t_start
    pipeline.close()

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

    summary = {
        "config_label": config_label,
        "model": model_name,
        "backend": "ollama",
        "prompt_strategy": "chained",
        "total_questions": len(questions),
        "valid_runs": len(valid),
        "correct": len(correct),
        "accuracy": len(correct) / len(valid) if valid else 0,
        "wall_clock_seconds": total_elapsed,
        "latency_ms": {
            "p50_total": p50,
            "p95_total": p95,
            "mean_total": mean,
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

    out_file.write_text(json.dumps({"summary": summary, "results": results}, indent=2))

    print()
    print("=" * 70)
    print("SWEEP CONFIG: " + config_label)
    print("=" * 70)
    print("Accuracy:    {:.1%}  ({}/{})".format(summary["accuracy"], summary["correct"], summary["valid_runs"]))
    print("Wall-clock:  {:.1f}s".format(summary["wall_clock_seconds"]))
    print("Latency p50/p95/mean: {:.0f} / {:.0f} / {:.0f} ms".format(p50, p95, mean))
    print()
    print("By category:")
    for cat, v in summary["by_category"].items():
        print("  {:<14s} {}/{} ({:.0%})  mean={:.0f}ms".format(
            cat, v["correct"], v["total"], v["accuracy"], v["mean_total_ms"]))
    print()
    print("[OK] Written to " + str(out_file))


if __name__ == "__main__":
    main()
