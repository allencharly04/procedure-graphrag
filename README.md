# assembly-rag

> Latency-optimized GraphRAG pipeline for shop-floor assembly assistance.

A reference implementation that pairs a Neo4j knowledge graph of assembly procedures with a tunable LLM inference layer (Ollama, llama.cpp, HuggingFace Transformers) and a benchmark harness for measuring end-to-end query latency. Built to explore how far GraphRAG response times can be pushed for real-time, hands-free use on the manufacturing floor.

## Why this exists

GraphRAG produces strong answers on multi-hop, structured-knowledge questions — exactly the kind of queries an assembly worker asks ("for procedure WS-014 step 7, what torque applies if the rivet is 5/32 inch Hi-Lok and the substrate is CFRP?"). But the round-trip latency of naive GraphRAG pipelines (vector search + Cypher traversal + 7B-class LLM at fp16) is often too slow for hands-free voice interaction.

This project quantifies the latency cost of each pipeline stage and explores the Pareto frontier of latency vs. answer quality across:

- **Model size:** 0.5B → 8B parameters
- **Quantization:** fp16 → int8 → int4 (Q4_K_M)
- **Inference backend:** Ollama, llama.cpp (CUDA), HuggingFace Transformers (bitsandbytes)
- **Prompt strategy:** monolithic vs. chained (decompose → retrieve → answer)

## Architecture
            Streamlit UI (text + optional voice)
                      |
            Optimization Router
            (model / backend / quant)
              /        |        \
          Ollama   llama.cpp   HF Transformers
                      |
          GraphRAG Retriever (graph + vector hybrid)
              /                         \
          Neo4j 5                    Chroma
          (Cypher templates)        (sentence-transformers)
                      |
          Profiler -> benchmark dashboard

## Domain

A synthetic aircraft wing-skin assembly knowledge graph (~30 procedures, ~200 steps, ~100 tools, ~80 part numbers, torque specs, cross-references). The dataset is fully reproducible from a seed JSON committed to this repo.

## Benchmark methodology

50 ground-truth Q/A pairs spanning single-hop, multi-hop, and arithmetic queries are evaluated end-to-end. The profiler tags time per stage:

- Embedding lookup (Chroma)
- Cypher traversal (Neo4j)
- Context formatting
- LLM time-to-first-token
- LLM decode

Final results are reported as the latency Pareto frontier vs. answer F1 (string match) and LLM-as-judge quality scores.

## Project status

This is an active build (Phase 1 of 6 complete). Tracking progress:

- [x] **Phase 1 — Foundation:** WSL2 environment, Neo4j 5.26 + APOC via Docker, Ollama smoke tests, embeddings smoke tests
- [ ] **Phase 2 — Synthetic domain:** wing-skin knowledge graph generation, 50 ground-truth Q/A pairs
- [ ] **Phase 3 — GraphRAG v0 baseline:** vector-only retrieval + llama3.1:8b fp16 monolithic prompt
- [ ] **Phase 4 — Optimization sweep:** model x quant x backend x chain grid (~50 configurations)
- [ ] **Phase 5 — Final pipeline + tests:** Pareto-optimal config wired into Streamlit, pytest latency regression
- [ ] **Phase 6 — Voice layer (optional):** faster-whisper + Piper for hands-free assistance

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

## Reproducibility

All hardware-relevant numbers are reported against a single reference setup:

- **GPU:** NVIDIA RTX 2060 (6 GB VRAM), CUDA 12.1
- **CPU:** Intel i7 (Lenovo Legion 5)
- **OS:** Windows 11 + WSL2 Ubuntu 24.04
- **Python:** 3.11

Environment is captured in `requirements.txt` (Phase 2). Synthetic data is regenerated from a single seed JSON.

## License

MIT.

## Author

Allen Charly — M.Sc. Digital Engineering and Management, RWTH Aachen.
