# Phase 4 — Optimization Sweep Report

## Goal

Find configurations of model size, quantization, and prompt strategy that
beat the Phase 3 baseline (llama3.1:8b monolithic, 18.8% accuracy at 2053 ms
mean latency on a 6 GB RTX 2060) on at least one Pareto axis.

## Sweep design

| Dimension | Values | Notes |
|---|---|---|
| Model size | 0.5b, 1b, 3b, 8b | qwen2.5 and llama3.2 families |
| Quantization | Q4_K_M | 6 GB VRAM forces Q4 for 8B; Q8/fp16 not testable |
| Backend | Ollama | llama-cpp dropped — Ollama is already <500ms on 1B; speedup not user-visible |
| Prompt strategy | monolithic, chained (3-call), monolithic + few-shot | three strategies tested at both ends of the size scale |

7 configurations measured against the 48-question gold eval set.

## Results

| Config | Accuracy | Mean ms | p50 | p95 | Mean retrieve | Mean prefill | Mean decode |
|---|---|---|---|---|---|---|---|
| **llama3.1:8b monolithic** (baseline) | **18.8%** | 2053 | 1695 | 5526 | 115 | 684 | 1190 |
| llama3.1:8b chained | 8.3% | 11478 | 11527 | 18630 | — | — | — |
| llama3.1:8b few-shot | 12.5% | 2189 | 1484 | 6239 | — | — | — |
| llama3.2:3b monolithic | 12.5% | 491 | 456 | 729 | 28 | 158 | 127 |
| **llama3.2:1b monolithic** | **12.5%** | **484** | 438 | 857 | 31 | 58 | 195 |
| llama3.2:1b few-shot | 8.3% | 842 | 435 | 2505 | — | — | — |
| qwen2.5:0.5b monolithic | 12.5% | 670 | 399 | 2403 | 170 | 39 | 264 |

By category for the two Pareto-frontier configs:

| | llama3.1:8b monolithic | llama3.2:1b monolithic |
|---|---|---|
| single_hop | 5/20 (25%) | 2/20 (10%) |
| multi_hop | 3/18 (17%) | 3/18 (17%) |
| arithmetic | 1/10 (10%) | 1/10 (10%) |

## Pareto frontier

Two configurations dominate the others:

- **Latency-optimal:** llama3.2:1b monolithic at 484 ms mean / 12.5% accuracy
- **Quality-optimal:** llama3.1:8b monolithic at 2053 ms mean / 18.8% accuracy

Every other measured config is dominated by one of these two. Neither chained
prompts nor few-shot examples produced a Pareto-improving point.

## Negative results

These are the most informative findings of the sweep.

### NR-1: Naive chained prompts halve accuracy on the 8B model

`llama3.1:8b` with a 3-call decompose -> filter -> answer chain dropped from
18.8% to **8.3%** while latency rose 5.6x to **11478 ms** mean. Multi-hop
collapsed entirely (0/18) and arithmetic likewise (0/10).

**Root cause:** the filter stage compresses retrieved context to a list of IDs
matching each sub-question. The answer stage then sees only the filtered list,
not the full structured context. When the decompose stage breaks a multi-hop
question into the wrong sub-questions, the answer stage cannot recover because
the original context has already been discarded. Information loss between
chain stages is the killer.

### NR-2: Few-shot examples hurt the 1B model

Adding three in-context examples to `llama3.2:1b` dropped accuracy from
**12.5% to 8.3%** with no latency improvement (842 ms vs 484 ms — actually
worse due to longer prompts).

**Root cause:** the 1B model treats few-shot examples as content to repeat
rather than format to imitate. Probe questions showed the model literally
echoing example markers like `### EXAMPLE 1` in its output.

### NR-3: Few-shot examples hurt the 8B model too

Adding three in-context examples to `llama3.1:8b` dropped accuracy from
**18.8% to 12.5%** at similar latency (2189 ms vs 2053 ms).

**Root cause hypothesis:** examples consume prefill tokens (~400 added per
prompt) that reduce attention budget for the retrieved context. Examples may
also bias outputs toward shapes that match the example categories but not all
48 question types.

### NR-4: Smaller models converge to the same accuracy floor

llama3.2:3b, llama3.2:1b, and qwen2.5:0.5b all hit exactly **12.5%** accuracy.
Different architectures, different sizes (4-6x), identical scores. This
suggests the bottleneck for sub-3B models is **retrieval quality and
single-hop generation**, not parameter count. Adding parameters within this
range is wasted on this task.

## What this means for production

For RAG systems on commodity GPUs (~6 GB VRAM):

- Size selection is the dominant lever. Monolithic prompts at multiple sizes
  give a clean Pareto curve.
- Chained prompts require a more sophisticated chain design than naive
  decompose -> filter -> answer. The filter stage must not discard information
  the answer stage will need.
- Few-shot examples are not a free win. In retrieval-grounded tasks they can
  hurt by displacing context tokens.
- The accuracy ceiling here (~19% for 8B) is dominated by retrieval gaps and
  the model losing track of long ID lists, not by prompt strategy.

## Reproducibility

Each config result is a JSON file under `benchmarks/results/`:
- `baseline.json` — Phase 3 baseline
- `sweep_qwen2.5_0.5b.json`
- `sweep_llama3.2_1b.json`
- `sweep_llama3.2_3b.json`
- `sweep_chained_llama3.1_8b.json`
- `sweep_fewshot_llama3.2_1b.json`
- `sweep_fewshot_llama3.1_8b.json`

Re-run any config with:
python benchmarks/run_sweep.py <model_name>
python benchmarks/run_sweep_chained.py <model_name>
python benchmarks/run_sweep_fewshot.py <model_name>

## What was not tested

- **llama-cpp-python:** dropped because Ollama already runs 1B at <500 ms; further speedup is not user-visible at this scale.
- **HuggingFace Transformers + bitsandbytes:** dropped to keep the sweep focused.
- **Q8/fp16 quantization:** 6 GB VRAM does not fit 8B at higher precision.
- **Re-engineered chain designs:** the negative result on naive chaining is what we report. A chain that preserved the full context across stages might do better, but that is future work.
