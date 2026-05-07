"""Tab 1: Benchmark Report — replays results from JSON files."""
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

APP_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = APP_ROOT.parent / "benchmarks" / "results"


CONFIG_LABELS = {
    "baseline.json": ("llama3.1:8b monolithic (baseline)", "Quality leader"),
    "sweep_qwen2.5_0.5b.json": ("qwen2.5:0.5b monolithic", "Smallest"),
    "sweep_llama3.2_1b.json": ("llama3.2:1b monolithic", "Latency leader"),
    "sweep_llama3.2_3b.json": ("llama3.2:3b monolithic", "Pareto-dominated"),
    "sweep_chained_llama3.1_8b.json": ("llama3.1:8b chained (3-call)", "Negative result"),
    "sweep_fewshot_llama3.2_1b.json": ("llama3.2:1b few-shot", "Negative result"),
    "sweep_fewshot_llama3.1_8b.json": ("llama3.1:8b few-shot", "Negative result"),
}


@st.cache_data
def load_all_results():
    """Load every result JSON into a list of dicts: {file, label, note, summary, results}."""
    out = []
    for filename, (label, note) in CONFIG_LABELS.items():
        p = RESULTS_DIR / filename
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        out.append({
            "file": filename,
            "label": label,
            "note": note,
            "summary": data["summary"],
            "results": data["results"],
        })
    return out


def render_pareto_chart(all_results):
    """Scatter plot of latency vs accuracy. Each config is one point."""
    rows = []
    for r in all_results:
        s = r["summary"]
        rows.append({
            "Config": r["label"],
            "Note": r["note"],
            "Mean latency (ms)": s["latency_ms"]["mean_total"],
            "Accuracy (%)": s["accuracy"] * 100,
            "p95 (ms)": s["latency_ms"].get("p95_total", 0),
        })
    df = pd.DataFrame(rows)

    color_map = {
        "Quality leader": "#2E7D32",
        "Latency leader": "#1565C0",
        "Smallest": "#90A4AE",
        "Pareto-dominated": "#90A4AE",
        "Negative result": "#C62828",
    }
    df["Color"] = df["Note"].map(color_map)

    fig = go.Figure()
    for note in df["Note"].unique():
        sub = df[df["Note"] == note]
        fig.add_trace(go.Scatter(
            x=sub["Mean latency (ms)"],
            y=sub["Accuracy (%)"],
            mode="markers+text",
            text=sub["Config"],
            textposition="top center",
            marker=dict(size=14, color=color_map.get(note, "#999")),
            name=note,
            hovertemplate="<b>%{text}</b><br>" +
                          "Mean latency: %{x:.0f} ms<br>" +
                          "Accuracy: %{y:.1f}%<br>" +
                          "<extra></extra>",
        ))

    fig.update_layout(
        xaxis_title="Mean total latency (ms, lower is better)",
        yaxis_title="Accuracy (%, higher is better)",
        xaxis=dict(type="log"),
        height=480,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98),
    )
    return fig, df


def render_results_table(results, summary):
    """Per-question table for one config."""
    rows = []
    for r in results:
        if "error" in r:
            rows.append({
                "ID": r["id"],
                "Category": r["category"],
                "Question": r["question"][:60] + ("..." if len(r["question"]) > 60 else ""),
                "Correct": "ERR",
                "Latency (ms)": None,
                "Predicted": r.get("error", "")[:60],
                "Expected": "",
            })
            continue
        rows.append({
            "ID": r["id"],
            "Category": r["category"],
            "Question": r["question"][:60] + ("..." if len(r["question"]) > 60 else ""),
            "Correct": "yes" if r["correct"] else "no",
            "Latency (ms)": int(r["latency"]["total_ms"]) if r["latency"].get("total_ms") else 0,
            "Predicted": (r.get("predicted") or "")[:80],
            "Expected": str(r.get("expected", ""))[:60],
        })
    df = pd.DataFrame(rows)
    return df


def render_question_detail(question_data):
    """Detail view for one question result."""
    r = question_data
    if "error" in r:
        st.error("This question errored: " + r.get("error", ""))
        return

    cols = st.columns(3)
    cols[0].metric("Result", "Correct" if r["correct"] else "Incorrect")
    cols[1].metric("Total latency", "{:.0f} ms".format(r["latency"]["total_ms"] or 0))
    cols[2].metric("Category", r["category"])

    st.markdown("**Question:**")
    st.code(r["question"])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Expected:**")
        st.code(str(r.get("expected", "")))
    with c2:
        st.markdown("**Predicted:**")
        st.code(r.get("predicted", "(none)"))

    if r.get("score_notes"):
        st.caption("Scoring notes: " + r["score_notes"])

    # Latency stages
    lat = r["latency"]
    stage_data = []
    if lat.get("retrieve_graph_ms"):
        stage_data.append({"Stage": "Retrieve", "ms": lat["retrieve_graph_ms"]})
    if lat.get("llm_prompt_eval_ms"):
        stage_data.append({"Stage": "LLM prefill", "ms": lat["llm_prompt_eval_ms"]})
    if lat.get("llm_decode_ms"):
        stage_data.append({"Stage": "LLM decode", "ms": lat["llm_decode_ms"]})
    if lat.get("llm_decompose_ms"):
        stage_data.append({"Stage": "LLM decompose", "ms": lat["llm_decompose_ms"]})
    if lat.get("llm_filter_ms"):
        stage_data.append({"Stage": "LLM filter", "ms": lat["llm_filter_ms"]})
    if lat.get("llm_answer_ms"):
        stage_data.append({"Stage": "LLM answer", "ms": lat["llm_answer_ms"]})
    if stage_data:
        st.markdown("**Per-stage latency:**")
        df_lat = pd.DataFrame(stage_data)
        fig = px.bar(df_lat, x="Stage", y="ms", height=240)
        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Retrieved context (graph templates fired, vector hits)
    with st.expander("Retrieval detail"):
        if r.get("graph_invocations"):
            st.markdown("**Graph templates fired:** " + ", ".join(r["graph_invocations"]))
        else:
            st.markdown("_No graph templates fired (fell back to vector-only)._")
        if r.get("vector_uids_kept"):
            st.markdown("**Vector hits retained:**")
            for uid in r["vector_uids_kept"]:
                st.code(uid)
        if r.get("context_chars"):
            st.caption("Context size: " + str(r["context_chars"]) + " chars / "
                       + str(r.get("context_tokens", "?")) + " est tokens")


def render_replay_tab():
    """Top-level orchestrator for the replay tab."""
    all_results = load_all_results()
    if not all_results:
        st.error("No benchmark results found in benchmarks/results/. "
                 "Run benchmarks/run_baseline.py and the run_sweep_*.py scripts first.")
        return

    st.markdown("### Pareto frontier: latency vs accuracy")
    st.caption("Each point is one configuration. Lower-right = better. "
               "Note: x-axis is log-scaled.")
    fig, df = render_pareto_chart(all_results)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Configurations summary table"):
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Drill into a configuration")

    config_options = [r["label"] for r in all_results]
    selected_label = st.selectbox(
        "Pick a configuration to inspect:",
        config_options,
        index=0,
    )
    selected = next(r for r in all_results if r["label"] == selected_label)

    s = selected["summary"]
    cols = st.columns(4)
    cols[0].metric("Accuracy", "{:.1%}".format(s["accuracy"]))
    cols[1].metric("Mean latency", "{:.0f} ms".format(s["latency_ms"]["mean_total"]))
    cols[2].metric("p50 latency", "{:.0f} ms".format(s["latency_ms"]["p50_total"]))
    cols[3].metric("p95 latency", "{:.0f} ms".format(s["latency_ms"]["p95_total"]))

    # Per-category breakdown
    cat_rows = []
    for cat, v in s.get("by_category", {}).items():
        cat_rows.append({
            "Category": cat,
            "Correct": "{}/{}".format(v["correct"], v["total"]),
            "Accuracy": "{:.0%}".format(v["accuracy"]),
            "Mean latency (ms)": "{:.0f}".format(v["mean_total_ms"]),
        })
    if cat_rows:
        st.markdown("**By category:**")
        st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)

    # Question table
    st.markdown("**Per-question results** (click a row to see the detail below)")
    df_q = render_results_table(selected["results"], s)

    selected_idx = st.dataframe(
        df_q,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    if selected_idx and selected_idx.selection.rows:
        idx = selected_idx.selection.rows[0]
        question_data = selected["results"][idx]
        st.markdown("---")
        st.markdown("### Question detail: " + question_data["id"])
        render_question_detail(question_data)
