"""GraphRAG Benchmark Demo — main entry point.

Two tabs:
  1. Benchmark Report: replays results from benchmarks/results/*.json
  2. Try It Live: hybrid retrieval + Groq inference (requires cloud secrets)
"""
import streamlit as st
from pathlib import Path
import sys

# Make local module imports work
APP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_ROOT))

st.set_page_config(
    page_title="GraphRAG Benchmark Demo",
    page_icon="charts",
    layout="wide",
)

# Header
st.title("Procedure GraphRAG: Benchmark Demo")
st.caption(
    "Vector + graph hybrid retrieval over a synthetic industrial maintenance corpus, "
    "evaluated across 7 model and prompt configurations on a 48-question gold set."
)

# Top-level metadata
col1, col2, col3, col4 = st.columns(4)
col1.metric("Procedures", "30")
col2.metric("Steps", "257")
col3.metric("Eval questions", "48")
col4.metric("Configs tested", "7")

st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["Benchmark Report", "Try It Live", "About"])

with tab1:
    from replay import render_replay_tab
    render_replay_tab()

with tab2:
    from live import render_live_tab
    render_live_tab()

with tab3:
    st.markdown("""
### What is this?

A reproducible benchmark suite for graph-augmented retrieval (GraphRAG) over a
synthetic industrial maintenance domain. The system uses a Neo4j knowledge
graph plus vector embeddings, with deterministic Cypher templates for known
query shapes and semantic similarity for everything else.

### Why does it matter?

Most "RAG demos" you see online ship a pipeline and call it done. This project
**measured** a pipeline — across 7 configurations, 48 evaluation questions,
and per-stage latency for retrieval, prefill, and decode. Three negative
results were documented (chained prompts halve accuracy on the 8B model;
few-shot examples hurt small and large models alike on this task; small
models converge to an accuracy floor independent of size).

### Architecture

- **Data:** 30 procedures, 257 steps, 18 torque specs, 75 components,
  generated via Anthropic Claude API.
- **Graph:** Neo4j (locally Docker, in cloud Neo4j AuraDB free tier) with
  9 Cypher templates routed by regex.
- **Vector:** sentence-transformers `all-MiniLM-L6-v2` (384-dim), 257 step
  embeddings in Chroma.
- **LLM:** Locally Ollama on RTX 2060 (Q4_K_M); in cloud Groq API.
- **Eval:** 48 questions with Cypher-validated ground truth.

### Where to find more

[GitHub repo](https://github.com/allencharly04/procedure-graphrag) ·
Full report in `REPORT.md` · Per-question results in `benchmarks/results/`
""")

# Footer
st.markdown("---")
st.caption(
    "Built by Allen M C as a portfolio piece for ML Engineer / Werkstudent roles. "
    "All code, data, and benchmark results are open-source on GitHub."
)
