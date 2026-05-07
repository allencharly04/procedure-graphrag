"""Tab 2: Try It Live — runs new questions through the cloud GraphRAG pipeline."""
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from cloud_retriever import CloudHybridRetriever
from groq_backend import generate as groq_generate


SAMPLE_QUESTIONS = [
    "What certifications does PRC-001 require?",
    "What torque should I use for M20 bolts?",
    "What is the torque pattern in TS-009?",
    "Which procedures involve a centrifugal pump?",
    "How many steps are in PRC-014?",
    "Which procedures require pressure systems certification?",
    "What torque specs are above 200 Nm?",
    "Show me PRC-002 step 3",
]


SYSTEM_PROMPT = (
    "You are an industrial maintenance assistant. Answer using ONLY the structured CONTEXT below. "
    "If CONTEXT does not contain the answer, reply with exactly: I don't have that information.\n"
    "Output rules: terse, no Markdown, no explanation. "
    "Use IDs (PRC-XXX, TS-XXX, T-XXX, C-XXX, CERT-X, D-XXX) where applicable."
)


@st.cache_resource(show_spinner="Connecting to AuraDB and loading embeddings...")
def get_retriever():
    secrets = {
        "NEO4J_URI": st.secrets["NEO4J_URI"],
        "NEO4J_USERNAME": st.secrets["NEO4J_USERNAME"],
        "NEO4J_PASSWORD": st.secrets["NEO4J_PASSWORD"],
        "NEO4J_DATABASE": st.secrets.get("NEO4J_DATABASE", "neo4j"),
    }
    return CloudHybridRetriever(secrets)


def _has_required_secrets():
    required = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "GROQ_API_KEY"]
    for k in required:
        try:
            v = st.secrets[k]
        except (KeyError, FileNotFoundError):
            return False
        if not v or "your-" in str(v):
            return False
    return True


def render_live_tab():
    st.markdown("### Try a question live")
    st.caption(
        "This runs the full GraphRAG pipeline on cloud infrastructure: Neo4j AuraDB, "
        "in-memory vector index, and Groq for LLM inference."
    )

    if not _has_required_secrets():
        st.error(
            "Cloud secrets not configured. Add NEO4J_URI, NEO4J_USERNAME, "
            "NEO4J_PASSWORD, and GROQ_API_KEY to .streamlit/secrets.toml."
        )
        return

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        sample = st.selectbox(
            "Pick a sample question or type your own below:",
            options=[""] + SAMPLE_QUESTIONS,
            index=0,
            key="live_sample",
        )
        question = st.text_area(
            "Your question:",
            value=sample if sample else "",
            height=80,
            placeholder="e.g. What torque should I use for M20 bolts?",
            key="live_question",
        )
    with col_btn:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.markdown("&nbsp;", unsafe_allow_html=True)
        run_clicked = st.button("Ask", type="primary", use_container_width=True)

    with st.expander("Configuration"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Retrieval**")
            st.text("Graph: AuraDB")
            st.text("Vector: 384-dim MiniLM, 257 docs in-memory")
            st.text("Templates: 9 Cypher templates with regex routing")
        with c2:
            st.markdown("**Inference**")
            st.text("Model: " + str(st.secrets.get("GROQ_MODEL", "llama-3.1-8b-instant")))
            st.text("Provider: Groq (cloud)")
            st.text("Strategy: monolithic prompt (Pareto-optimal per Phase 4)")

    if not run_clicked:
        return

    if not question.strip():
        st.warning("Please enter a question.")
        return

    retriever = get_retriever()

    t0 = time.perf_counter()
    try:
        ctx = retriever.retrieve(question)
    except Exception as e:
        st.error("Retrieval failed: " + str(e))
        return
    retrieve_ms = (time.perf_counter() - t0) * 1000

    prompt_context = ctx.formatted_context if ctx.formatted_context else "(no context retrieved)"
    prompt = (
        SYSTEM_PROMPT + "\n\nCONTEXT:\n" + prompt_context +
        "\n\nQUESTION: " + question + "\n\nANSWER:"
    )

    t0 = time.perf_counter()
    try:
        gen = groq_generate(
            prompt,
            api_key=st.secrets["GROQ_API_KEY"],
            model=st.secrets.get("GROQ_MODEL", "llama-3.1-8b-instant"),
        )
    except Exception as e:
        st.error("Groq inference failed: " + str(e))
        return
    groq_ms = (time.perf_counter() - t0) * 1000
    total_ms = retrieve_ms + groq_ms

    st.markdown("---")

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Total", str(int(total_ms)) + " ms")
    col_b.metric("Retrieve", str(int(retrieve_ms)) + " ms")
    col_c.metric("LLM prefill", str(int(gen.prompt_eval_ms or 0)) + " ms")
    col_d.metric("LLM decode", str(int(gen.eval_ms or 0)) + " ms")

    st.markdown("**Answer:**")
    st.code(gen.response or "(empty response)")

    stages = [
        {"Stage": "Retrieve (Aura + vector)", "ms": retrieve_ms},
        {"Stage": "LLM prefill (Groq)", "ms": gen.prompt_eval_ms or 0},
        {"Stage": "LLM decode (Groq)", "ms": gen.eval_ms or 0},
    ]
    df_stages = pd.DataFrame(stages)
    fig = px.bar(df_stages, x="Stage", y="ms", height=220)
    fig.update_layout(margin=dict(l=20, r=20, t=10, b=20), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    templates_str = ", ".join(ctx.graph_invocations) if ctx.graph_invocations else "none (vector-only)"
    st.caption(
        "Prompt tokens: " + str(gen.prompt_eval_count or "?") + " | " +
        "Generated tokens: " + str(gen.eval_count or "?") + " | " +
        "Templates fired: " + templates_str + " | " +
        "Vector hits kept: " + str(len(ctx.vector_uids))
    )

    with st.expander("Retrieved context (what the LLM saw)"):
        if not ctx.formatted_context:
            st.warning(
                "No context retrieved. The LLM was told to say "
                "'I don't have that information' rather than guess."
            )
        else:
            st.text(ctx.formatted_context)

    st.caption(
        "Live latencies are typically faster than the RTX 2060 baseline shown on the "
        "Benchmark Report tab because Groq's hardware (specialized LPUs) delivers higher "
        "tokens/sec than a consumer GPU running Q4_K_M weights."
    )
