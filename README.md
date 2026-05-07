# procedure-graphrag

**Live demo:** [procedure-graphrag.streamlit.app](https://procedure-graphrag.streamlit.app) — interactive benchmark report + try the live cloud pipeline (Neo4j AuraDB + Groq).



> Latency-optimized GraphRAG pipeline with a multi-backend LLM benchmark suite.

A reference implementation for **hybrid graph + vector retrieval over procedural knowledge**, paired with a benchmark harness that measures end-to-end query latency across model sizes, quantization levels, inference backends, and prompt strategies. Built to explore how far GraphRAG response times can be pushed for low-latency, real-time use cases — voice assistants, on-device copilots, embedded support tools.

## Why this exists

GraphRAG produces high-quality answers on multi-hop, structured-knowledge questions by combining a knowledge graph (relational reasoning) with a vector store (semantic recall). It outperforms vector-only RAG on questions like *"which procedures require Level 2 certification and use a torque value above 40 Nm?"* — the kind of relational query that breaks naive cosine similarity.

The cost: latency. A naive GraphRAG round-trip — embedding lookup, Cypher traversal, context formatting, 7B-class LLM at fp16 — can easily exceed 4–5 seconds. Too slow for voice interfaces, too slow for hands-free interaction, too slow for fluid conversation.

This project quantifies the latency cost of each pipeline stage and maps the Pareto frontier of latency vs. answer quality across:

- **Model size:** 0.5B → 8B parameters
- **Quantization:** fp16 → int8 → int4 (Q4_K_M)
- **Inference backend:** Ollama, llama.cpp (CUDA), HuggingFace Transformers (bitsandbytes)
- **Prompt strategy:** monolithic vs. chained (decompose → retrieve → answer)

## Reference domain

The benchmark uses a synthetic **industrial equipment maintenance** knowledge graph (~30 procedures covering pumps, valves, motors, and heat exchangers; ~200 steps; ~60 components; ~40 torque specs; ~10 certifications; ~15 defects). The schema is intentionally generic — the same structure transfers to assembly procedures, lab protocols, surgical workflows, IT runbooks, or any domain where structured procedures are queried multi-hop.

The dataset is fully reproducible from a seed JSON committed to this repo.

## Architecture
        Streamlit UI (text + optional voice)
                  |
        Optimization Router
        (model / backend / quant)
          /        |        \\
      Ollama   llama.cpp   HF Transformers
                  |
      GraphRAG Retriever (graph + vector hybrid)
          /                         \\
      Neo4j 5                    Chroma
      (Cypher templates)        (sentence-transformers)
                  |
      Profiler -> benchmark dashboard

## Benchmark methodology

48 ground-truth Q/A pairs spanning single-hop (20), multi-hop (18), and arithmetic (10) queries are evaluated end-to-end. The profiler tags time per stage:

- Embedding lookup (Chroma)
- Cypher traversal (Neo4j)
- Context formatting
- LLM time-to-first-token
- LLM decode

Results are reported as the latency Pareto frontier vs. answer F1 (string match) and LLM-as-judge quality scores. Reproducibility: all numbers are reported against a single reference setup (RTX 2060, CUDA 12.1, WSL2 Ubuntu 24.04, Python 3.11).

## Current state (Phase 2 complete)

| Metric | Value |
|---|---|
| Procedures | 30 across 7 equipment types |
| Steps | 257 (avg 8.6 per procedure, range 4-14) |
| Tools / Components / Certifications / Defects | 30 / 60 / 10 / 15 |
| Torque specs | 18 (range 22-390 Nm) |
| Procedure prerequisites | 10 (33% have one prereq) |
| Graph nodes / edges | 420 / 1034 |
| Eval questions | 48 (20 single-hop / 18 multi-hop / 10 arithmetic) |

The eval ground-truth answers are computed by executing the gold-standard Cypher
against Neo4j, not by LLM hand-computation. Initial validation showed 60% of
LLM-generated answers were incorrect when compared against database results,
which motivated the "trust the database, not the LLM" methodology now used
throughout the eval pipeline.

## Phase 4 sweep summary

Seven configurations were tested against the Phase 3 baseline. Two remain on
the Pareto frontier; five are dominated. Three negative results are documented
in `REPORT.md`.

| Config | Accuracy | Mean latency |
|---|---|---|
| **llama3.1:8b monolithic** (quality-optimal) | **18.8%** | 2053 ms |
| **llama3.2:1b monolithic** (latency-optimal) | **12.5%** | **484 ms** |
| llama3.2:3b monolithic | 12.5% | 491 ms |
| qwen2.5:0.5b monolithic | 12.5% | 670 ms |
| llama3.1:8b few-shot | 12.5% | 2189 ms |
| llama3.2:1b few-shot | 8.3% | 842 ms |
| llama3.1:8b chained | 8.3% | 11478 ms |

Negative results:
- Naive chained prompts halve accuracy on the 8B model (filter stage discards information)
- Few-shot examples hurt both 1B and 8B models in retrieval-grounded settings
- Small models (0.5B-3B) converge to the same 12.5% floor — retrieval-bound, not generation-bound

See `REPORT.md` for full analysis and root causes.

## Baseline (Phase 3 complete)

The v0 GraphRAG pipeline (vector + graph hybrid retrieval, llama3.1:8b Q4_K_M
on RTX 2060 via Ollama, monolithic prompt) was evaluated on the 48-question
gold set:

| Metric | Value |
|---|---|
| **Accuracy** | 18.8% (9 of 48) |
| Mean total latency | 2053 ms |
| p50 / p95 latency | 1695 / 5526 ms |
| Mean retrieve stage | 115 ms |
| Mean prefill stage | 684 ms |
| Mean decode stage | 1190 ms |

By category: single-hop 25%, multi-hop 17%, arithmetic 10%.

Five failure modes identified from the per-question logs:

1. **ID hallucination** - the model invents plausible-looking IDs when uncertain (e.g. T-123, PRC-456)
2. **Set incompleteness** - missing items in long ID lists, especially in multi-hop questions
3. **Arithmetic weakness** - small models struggle with counting and aggregation over context
4. **Retrieval gaps** - some attribute queries (material, tolerance, hold-point counts) fall outside template coverage
5. **Format mismatches** - some answers are semantically correct but lose strict scoring (e.g. "shell_and_tube" vs "shell_tube")

Phase 4 targets each mode independently: quantization sweep for latency,
prompt chaining for arithmetic, schema-aware prompting for hallucination,
expanded templates for retrieval coverage. Improvements are measured against
this baseline.

Per-question detail: `benchmarks/results/baseline.json`.

## Project status

This is an active build. Tracking progress:

- [x] **Phase 1 — Foundation:** WSL2 environment, Neo4j 5.26 + APOC via Docker, Ollama integration, embeddings smoke tests
- [x] **Phase 2 — Synthetic domain:** maintenance procedure knowledge graph generation, 48 Cypher-validated ground-truth Q/A pairs
- [x] **Phase 3 — GraphRAG v0 baseline:** vector + graph retrieval + llama3.1:8b Q4_K_M monolithic prompt
- [x] **Phase 4 — Optimization sweep:** 7 configurations across model size, prompt strategy, and chain design
- [x] **Phase 5 — Live demo:** Streamlit app deployed at procedure-graphrag.streamlit.app (Neo4j AuraDB + Groq)
- [ ] **Phase 6 — Voice layer (optional):** faster-whisper + Piper for hands-free use

See [REPORT.md](./REPORT.md) for benchmark results once Phase 4 is complete.

## Stack

| Layer | Tool |
|---|---|
| Graph database | Neo4j 5.26 Community + APOC |
| Vector database | Chroma |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM backends | Ollama, llama-cpp-python (CUDA), HuggingFace Transformers + bitsandbytes |
| Models under test | qwen2.5:0.5b, llama3.2:1b, llama3.2:3b, llama3.1:8b |
| Application | Streamlit |
| Voice (optional) | faster-whisper (STT) + Piper (TTS) |
| Orchestration | Docker Compose |

## Why this generalizes

The same retrieval and inference pipeline applies anywhere structured procedures are queried multi-hop:

- **Manufacturing** — assembly procedures, quality inspection workflows
- **Field service** — equipment maintenance, repair procedures
- **Healthcare** — clinical pathways, surgical protocols
- **IT operations** — incident response runbooks, deployment procedures
- **Compliance** — audit checklists, regulatory procedures

Schema rename and seed regeneration are the only domain-specific work; the retrieval, optimization, and benchmarking code is domain-agnostic.

## License

MIT.

## Author

Allen Charly — M.Sc. Digital Engineering and Management, RWTH Aachen.
