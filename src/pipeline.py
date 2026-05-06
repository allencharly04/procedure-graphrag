"""GraphRAG pipeline: hybrid retrieve -> prompt -> Ollama generate.

The pipeline is the unit benchmarked in Phase 4. Every public method tracks
per-stage latency in milliseconds. Latencies are reported alongside the answer
so downstream evaluation can build the Pareto frontier of latency vs. quality.
"""
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make sibling modules importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "retrieval"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "llm"))

from hybrid_retriever import HybridRetriever, HybridContext
import ollama_backend


# -----------------------
# Prompt template
# -----------------------

# Monolithic prompt: one shot, retrieved context inline. This is the v0 baseline.
# Phase 4 will compare against chained prompts (decompose -> retrieve -> answer).
SYSTEM_INSTRUCTION = """You are an industrial maintenance assistant. Answer using ONLY the structured context.
If the context doesn't contain the answer, say "I don't have that information."

OUTPUT FORMAT IS STRICT:
- For ID lookups (which procedures, which tools, which certifications, which defects), output ONLY the IDs, one per line. Use the IDs (PRC-XXX, T-XXX, CERT-X, D-XXX, C-XXX, TS-XXX), never the full names.
- For numeric answers (counts, torques, durations), output ONLY the number. No units unless they were in the question. No sentence.
- For specific value lookups (torque pattern, criticality), output ONLY the value.
- Never echo the context back. Never use Markdown headers or bullets in your answer.
- Never explain your reasoning. Be terse."""

PROMPT_TEMPLATE = """{system}

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


# -----------------------
# Result types
# -----------------------

@dataclass
class StageLatency:
    """Per-stage timing in milliseconds. All optional - some stages may be skipped."""
    embed_ms: Optional[float] = None
    retrieve_vector_ms: Optional[float] = None
    retrieve_graph_ms: Optional[float] = None
    format_ms: Optional[float] = None
    llm_request_ms: Optional[float] = None
    llm_prompt_eval_ms: Optional[float] = None
    llm_decode_ms: Optional[float] = None
    total_ms: Optional[float] = None


@dataclass
class PipelineResult:
    """One full pipeline run."""
    question: str
    answer: str
    context: str
    graph_invocations: List[str]
    vector_uids: List[str]
    context_chars: int
    context_tokens: int
    prompt_chars: int
    model: str
    latency: StageLatency
    eval_count: Optional[int] = None
    prompt_eval_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# -----------------------
# Pipeline
# -----------------------

class Pipeline:
    """End-to-end GraphRAG pipeline. Reusable across questions."""

    def __init__(self, model: str = "llama3.1:latest",
                 num_ctx: int = 4096, max_tokens: int = 512,
                 vector_top_k: int = 5):
        self.model = model
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens
        self.retriever = HybridRetriever(vector_top_k=vector_top_k)

    def close(self):
        self.retriever.close()

    def answer(self, question: str) -> PipelineResult:
        latency = StageLatency()
        t_total = time.perf_counter()

        # 1. Retrieval (hybrid: graph + vector)
        # We measure these as one block since HybridRetriever interleaves them.
        # Phase 3D.3 will split these into separate timings.
        t = time.perf_counter()
        ctx = self.retriever.retrieve(question)
        retrieve_ms = (time.perf_counter() - t) * 1000.0
        # For now, attribute combined retrieval to the graph slot since that's
        # the dominant work for queries that fire templates. The hybrid module
        # will be split in 3D.3.
        latency.retrieve_graph_ms = retrieve_ms

        # 2. Prompt assembly
        t = time.perf_counter()
        prompt = PROMPT_TEMPLATE.format(
            system=SYSTEM_INSTRUCTION,
            context=ctx.formatted_context if ctx.formatted_context else "(no context retrieved)",
            question=question,
        )
        latency.format_ms = (time.perf_counter() - t) * 1000.0

        # 3. LLM generation
        gen = ollama_backend.generate(
            prompt,
            model=self.model,
            num_ctx=self.num_ctx,
            max_tokens=self.max_tokens,
            temperature=0.0,
        )
        latency.llm_request_ms = gen.request_ms
        latency.llm_prompt_eval_ms = gen.prompt_eval_ms
        latency.llm_decode_ms = gen.eval_ms

        latency.total_ms = (time.perf_counter() - t_total) * 1000.0

        return PipelineResult(
            question=question,
            answer=gen.response,
            context=ctx.formatted_context,
            graph_invocations=ctx.graph_invocations,
            vector_uids=ctx.vector_uids,
            context_chars=ctx.char_count,
            context_tokens=ctx.est_tokens,
            prompt_chars=len(prompt),
            model=self.model,
            latency=latency,
            eval_count=gen.eval_count,
            prompt_eval_count=gen.prompt_eval_count,
        )


# -----------------------
# CLI smoke test
# -----------------------

if __name__ == "__main__":
    PROBES = [
        "What certifications does PRC-001 require?",
        "What torque should I use for M20 bolts?",
        "Which procedures involve a centrifugal pump?",
        "Show me PRC-002 step 3",
        "How many steps are in the induction motor bearing replacement procedure?",
    ]

    pipeline = Pipeline()
    try:
        # Warmup call - load model into GPU memory before timing other calls
        print("[*] Warming up model ...")
        _ = pipeline.answer("Reply with: ready")
        print("    warmup done")
        print()

        for q in PROBES:
            print("=" * 72)
            print("Q: " + q)
            r = pipeline.answer(q)
            print()
            print("A: " + r.answer[:300])
            print()
            print("Stages (ms):")
            print("  retrieve (combined): {:7.1f}".format(r.latency.retrieve_graph_ms or 0))
            print("  format (assembly):   {:7.1f}".format(r.latency.format_ms or 0))
            print("  llm prompt_eval:     {:7.1f}  ({} tokens)".format(
                r.latency.llm_prompt_eval_ms or 0, r.prompt_eval_count or 0))
            print("  llm decode:          {:7.1f}  ({} tokens)".format(
                r.latency.llm_decode_ms or 0, r.eval_count or 0))
            print("  llm total request:   {:7.1f}".format(r.latency.llm_request_ms or 0))
            print("  TOTAL:               {:7.1f}".format(r.latency.total_ms or 0))
            print("  context: " + str(r.context_chars) + " chars / "
                  + str(r.prompt_chars) + " chars in prompt")
            print()
    finally:
        pipeline.close()
