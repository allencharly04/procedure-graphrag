"""Few-shot monolithic pipeline.

Same architecture as Pipeline (single LLM call) but with 3 in-context examples
prepended to the prompt. Tests whether small models can be coached toward
correct answer format via examples instead of instruction.

Examples cover the three eval categories: single_hop, multi_hop, arithmetic.
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


SYSTEM_INSTRUCTION = """You are an industrial maintenance assistant. Answer using ONLY the structured context provided.
Output rules: terse, no Markdown, no explanation. Use IDs (PRC-XXX, T-XXX, CERT-X, D-XXX, C-XXX, TS-XXX) not full names.

Study the examples below carefully and produce your answer in the exact same format."""


# Three carefully chosen examples covering the three categories.
# Each shows a complete: context -> question -> answer triple.
FEW_SHOT_EXAMPLES = """### EXAMPLE 1 (single-hop, list of IDs)

CONTEXT:
## Procedure PRC-099: Example Pump Service
- Equipment: centrifugal_pump
- Required certifications: Mechanical Maintenance Level 1, Pressure Systems Inspector Level 2

QUESTION: What certifications does PRC-099 require?

ANSWER:
CERT-1
CERT-6


### EXAMPLE 2 (multi-hop, filter set)

CONTEXT:
## Procedures matching equipment filter (3 procedures)
- PRC-097: Example Heat Exchanger Service [high]
- PRC-098: Example Heat Exchanger Cleaning [medium]
- PRC-099: Example Heat Exchanger Test [critical]

QUESTION: Which heat exchanger procedures have criticality 'high'?

ANSWER:
PRC-097


### EXAMPLE 3 (arithmetic, count)

CONTEXT:
## Procedure PRC-099: Example Pump Service
- Steps:
  1. Isolate pump.
  2. Remove cover. [HOLD POINT]
  3. Replace bearing. [HOLD POINT]
  4. Reassemble. [HOLD POINT]
  5. Restart.

QUESTION: How many steps in PRC-099 are hold points?

ANSWER:
3
"""


PROMPT_TEMPLATE = """{system}

{examples}

### YOUR TASK

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


@dataclass
class StageLatency:
    embed_ms: Optional[float] = None
    retrieve_vector_ms: Optional[float] = None
    retrieve_graph_ms: Optional[float] = None
    format_ms: Optional[float] = None
    llm_request_ms: Optional[float] = None
    llm_prompt_eval_ms: Optional[float] = None
    llm_decode_ms: Optional[float] = None
    total_ms: Optional[float] = None


@dataclass
class FewShotResult:
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


class FewShotPipeline:
    def __init__(self, model: str = "llama3.2:1b",
                 num_ctx: int = 4096, max_tokens: int = 256,
                 vector_top_k: int = 5):
        self.model = model
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens
        self.retriever = HybridRetriever(vector_top_k=vector_top_k)

    def close(self):
        self.retriever.close()

    def answer(self, question: str) -> FewShotResult:
        latency = StageLatency()
        t_total = time.perf_counter()

        t = time.perf_counter()
        ctx = self.retriever.retrieve(question)
        latency.retrieve_graph_ms = (time.perf_counter() - t) * 1000.0

        t = time.perf_counter()
        prompt = PROMPT_TEMPLATE.format(
            system=SYSTEM_INSTRUCTION,
            examples=FEW_SHOT_EXAMPLES,
            context=ctx.formatted_context if ctx.formatted_context else "(no context retrieved)",
            question=question,
        )
        latency.format_ms = (time.perf_counter() - t) * 1000.0

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

        return FewShotResult(
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.2:1b")
    args = parser.parse_args()

    PROBES = [
        "What certifications does PRC-001 require?",
        "How many steps are in the induction motor bearing replacement procedure?",
        "What torque should I use for M20 bolts?",
        "Which procedures involve a centrifugal pump?",
    ]

    pipeline = FewShotPipeline(model=args.model)
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
            print("A: " + r.answer[:200])
            print()
            print("Stages (ms): retrieve={:.0f}, prefill={:.0f}, decode={:.0f}, total={:.0f}".format(
                r.latency.retrieve_graph_ms or 0,
                r.latency.llm_prompt_eval_ms or 0,
                r.latency.llm_decode_ms or 0,
                r.latency.total_ms or 0))
            print("Prompt: " + str(r.prompt_chars) + " chars, " + str(r.prompt_eval_count) + " tokens")
            print()
    finally:
        pipeline.close()
