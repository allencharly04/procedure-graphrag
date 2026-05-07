"""Chained-prompt pipeline: decompose -> filter -> answer.

Three sequential LLM calls instead of one monolithic call. Each step has a
simpler job, which helps small models avoid hallucination and miscounting.

The retrieve and format stages are unchanged from monolithic Pipeline.
"""
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent / "retrieval"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "llm"))

from hybrid_retriever import HybridRetriever, HybridContext
import ollama_backend


# -----------------------
# Three-stage prompts
# -----------------------

DECOMPOSE_PROMPT = """You are rewriting a question into 1-3 retrieval steps for an industrial maintenance knowledge graph.

The graph contains: Procedures (PRC-XXX), Steps, Tools (T-XXX), Components (C-XXX), Certifications (CERT-X), Defects (D-XXX), TorqueSpecs (TS-XXX).

STRICT RULES:
- Only ask about entities that actually exist in this graph (procedures, steps, tools, components, certs, defects, torque specs).
- DO NOT ask about industries, sectors, applications, history, types, or anything outside the graph.
- DO NOT introduce new topics not in the original question.
- If the question references a specific ID (PRC-014, T-005, etc.), keep that ID in the sub-questions.
- Use 1 sub-question for simple lookups, 2-3 only for multi-hop or counting questions.

Reply with hyphen bullets only, no header, no commentary.

Question: {question}

Sub-questions:"""


FILTER_PROMPT = """You are extracting relevant items from a knowledge graph context.
Given the sub-questions and the structured CONTEXT below, list the IDs that
answer each sub-question. Use only IDs that appear in the CONTEXT.
Format: one line per sub-question, "sub-question: ID, ID, ID" or "sub-question: none".
No commentary.

Sub-questions:
{sub_questions}

CONTEXT:
{context}

Filtered items:"""


ANSWER_PROMPT = """You are giving the final answer to a question using filtered items.

OUTPUT FORMAT IS STRICT:
- For ID lookups, output ONLY the IDs, one per line. Use IDs (PRC-XXX, T-XXX, CERT-X, D-XXX, C-XXX, TS-XXX) never names.
- For numeric answers (counts, torques, durations), output ONLY the number. No units. No sentence.
- For specific value lookups (pattern, criticality), output ONLY the value.
- Never echo the context. Never use Markdown. Never explain.

Question: {question}

Filtered items from context:
{filtered}

Final answer:"""


# -----------------------
# Result types
# -----------------------

@dataclass
class StageLatency:
    embed_ms: Optional[float] = None
    retrieve_vector_ms: Optional[float] = None
    retrieve_graph_ms: Optional[float] = None
    format_ms: Optional[float] = None
    # Three LLM stages instead of one
    llm_decompose_ms: Optional[float] = None
    llm_filter_ms: Optional[float] = None
    llm_answer_ms: Optional[float] = None
    # Aggregates across all three calls
    llm_request_ms: Optional[float] = None
    llm_prompt_eval_ms: Optional[float] = None
    llm_decode_ms: Optional[float] = None
    total_ms: Optional[float] = None


@dataclass
class ChainedResult:
    question: str
    answer: str
    sub_questions: str
    filtered: str
    context: str
    graph_invocations: List[str]
    vector_uids: List[str]
    context_chars: int
    context_tokens: int
    model: str
    latency: StageLatency
    eval_count: Optional[int] = None
    prompt_eval_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------
# Chained Pipeline
# -----------------------

class ChainedPipeline:
    """Three-call pipeline: decompose -> filter -> answer."""

    def __init__(self, model: str = "llama3.2:1b",
                 num_ctx: int = 4096, vector_top_k: int = 5):
        self.model = model
        self.num_ctx = num_ctx
        self.retriever = HybridRetriever(vector_top_k=vector_top_k)

    def close(self):
        self.retriever.close()

    def _llm(self, prompt: str, max_tokens: int):
        """Single LLM call with deterministic settings."""
        return ollama_backend.generate(
            prompt,
            model=self.model,
            num_ctx=self.num_ctx,
            max_tokens=max_tokens,
            temperature=0.0,
        )

    def answer(self, question: str) -> ChainedResult:
        latency = StageLatency()
        t_total = time.perf_counter()

        # 1. Retrieval
        t = time.perf_counter()
        ctx = self.retriever.retrieve(question)
        latency.retrieve_graph_ms = (time.perf_counter() - t) * 1000.0

        # 2. Decompose call (small, just sub-questions)
        decompose_prompt = DECOMPOSE_PROMPT.format(question=question)
        t = time.perf_counter()
        d = self._llm(decompose_prompt, max_tokens=120)
        latency.llm_decompose_ms = (time.perf_counter() - t) * 1000.0
        sub_questions = d.response.strip()

        # 3. Filter call (medium, IDs from context)
        filter_prompt = FILTER_PROMPT.format(
            sub_questions=sub_questions,
            context=ctx.formatted_context if ctx.formatted_context else "(no context retrieved)",
        )
        t = time.perf_counter()
        fres = self._llm(filter_prompt, max_tokens=400)
        latency.llm_filter_ms = (time.perf_counter() - t) * 1000.0
        filtered = fres.response.strip()

        # 4. Answer call (strict format, terse)
        answer_prompt = ANSWER_PROMPT.format(
            question=question,
            filtered=filtered,
        )
        t = time.perf_counter()
        ares = self._llm(answer_prompt, max_tokens=200)
        latency.llm_answer_ms = (time.perf_counter() - t) * 1000.0

        # Aggregate LLM stats
        latency.llm_request_ms = (
            (d.request_ms or 0) + (fres.request_ms or 0) + (ares.request_ms or 0)
        )
        latency.llm_prompt_eval_ms = (
            (d.prompt_eval_ms or 0) + (fres.prompt_eval_ms or 0) + (ares.prompt_eval_ms or 0)
        )
        latency.llm_decode_ms = (
            (d.eval_ms or 0) + (fres.eval_ms or 0) + (ares.eval_ms or 0)
        )

        latency.total_ms = (time.perf_counter() - t_total) * 1000.0

        eval_count = (d.eval_count or 0) + (fres.eval_count or 0) + (ares.eval_count or 0)
        prompt_eval_count = (
            (d.prompt_eval_count or 0) + (fres.prompt_eval_count or 0) + (ares.prompt_eval_count or 0)
        )

        return ChainedResult(
            question=question,
            answer=ares.response.strip(),
            sub_questions=sub_questions,
            filtered=filtered,
            context=ctx.formatted_context,
            graph_invocations=ctx.graph_invocations,
            vector_uids=ctx.vector_uids,
            context_chars=ctx.char_count,
            context_tokens=ctx.est_tokens,
            model=self.model,
            latency=latency,
            eval_count=eval_count,
            prompt_eval_count=prompt_eval_count,
        )


# -----------------------
# CLI smoke test
# -----------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.2:1b")
    args = parser.parse_args()

    PROBES = [
        "What certifications does PRC-001 require?",
        "How many steps are in the induction motor bearing replacement procedure?",
        "What torque should I use for M20 bolts?",
        "How many total steps are there across all induction motor procedures?",
    ]

    pipeline = ChainedPipeline(model=args.model)
    try:
        print("[*] Warming up model ...")
        _ = pipeline.answer("Reply with: ready")
        print("    warmup done")
        print()

        for q in PROBES:
            print("=" * 72)
            print("Q: " + q)
            r = pipeline.answer(q)
            print()
            print("Sub-questions:")
            for line in r.sub_questions.split("\n")[:6]:
                print("  " + line)
            print()
            print("Filtered:")
            for line in r.filtered.split("\n")[:6]:
                print("  " + line)
            print()
            print("FINAL ANSWER: " + r.answer[:200])
            print()
            print("Stages (ms):")
            print("  retrieve:           {:7.1f}".format(r.latency.retrieve_graph_ms or 0))
            print("  llm decompose:      {:7.1f}".format(r.latency.llm_decompose_ms or 0))
            print("  llm filter:         {:7.1f}".format(r.latency.llm_filter_ms or 0))
            print("  llm answer:         {:7.1f}".format(r.latency.llm_answer_ms or 0))
            print("  llm decode total:   {:7.1f}  ({} tokens)".format(
                r.latency.llm_decode_ms or 0, r.eval_count or 0))
            print("  TOTAL:              {:7.1f}".format(r.latency.total_ms or 0))
            print()
    finally:
        pipeline.close()
