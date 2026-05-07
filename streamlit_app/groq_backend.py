"""Groq inference backend for the live tab.

Mirror of ollama_backend but using Groq API for cloud deployment.
Provides per-stage latency telemetry.
"""
import time
from dataclasses import dataclass
from typing import Optional

from groq import Groq


@dataclass
class GroqResult:
    """One LLM generation result."""
    response: str
    model: str
    request_ms: float
    eval_ms: Optional[float]          # Groq calls this completion_time
    prompt_eval_ms: Optional[float]   # Groq calls this prompt_time
    eval_count: Optional[int]         # output tokens
    prompt_eval_count: Optional[int]  # input tokens


def generate(prompt: str, api_key: str, model: str = "llama-3.1-8b-instant",
             temperature: float = 0.0, max_tokens: int = 512) -> GroqResult:
    """Send a prompt to Groq and return response + latency breakdown."""
    client = Groq(api_key=api_key)

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    text = resp.choices[0].message.content or ""
    usage = resp.usage

    # Groq returns timings in seconds, not nanoseconds. Convert to ms.
    completion_time_ms = (usage.completion_time * 1000) if hasattr(usage, "completion_time") and usage.completion_time else None
    prompt_time_ms = (usage.prompt_time * 1000) if hasattr(usage, "prompt_time") and usage.prompt_time else None

    return GroqResult(
        response=text.strip(),
        model=model,
        request_ms=elapsed_ms,
        eval_ms=completion_time_ms,
        prompt_eval_ms=prompt_time_ms,
        eval_count=usage.completion_tokens if usage else None,
        prompt_eval_count=usage.prompt_tokens if usage else None,
    )
