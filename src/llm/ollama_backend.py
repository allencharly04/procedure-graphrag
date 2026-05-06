"""Ollama backend for the LLM stage.

Reads OLLAMA_HOST from .env (default: auto-detect WSL2 host gateway).
Provides a simple .generate(prompt, model) -> dict with response + latency.
"""
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()


def _detect_host() -> str:
    """In WSL2, the Windows host is the default gateway. On Linux/Mac,
    fall back to localhost."""
    explicit = os.getenv("OLLAMA_HOST")
    if explicit:
        return explicit
    try:
        result = subprocess.run(
            ["ip", "route", "show"],
            capture_output=True, text=True, check=True, timeout=2,
        )
        for line in result.stdout.splitlines():
            if line.startswith("default"):
                return line.split()[2]
    except Exception:
        pass
    return "127.0.0.1"


HOST = _detect_host()
PORT = int(os.getenv("OLLAMA_PORT", "11434"))
BASE_URL = "http://" + HOST + ":" + str(PORT)


@dataclass
class GenerateResult:
    """One LLM generation result."""
    response: str
    model: str
    # Latency stages, all in milliseconds
    request_ms: float          # total wall clock from request to response
    eval_ms: Optional[float]   # Ollama-reported eval (decode) time
    prompt_eval_ms: Optional[float]  # Ollama-reported prompt eval (prefill) time
    eval_count: Optional[int]  # tokens generated
    prompt_eval_count: Optional[int]  # tokens in prompt


def generate(prompt: str, model: str = "llama3.1:latest",
             temperature: float = 0.0, max_tokens: int = 512,
             num_ctx: int = 4096,
             timeout: float = 120.0) -> GenerateResult:
    """Send a prompt to Ollama and return the response + latency breakdown."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": num_ctx,
        },
    }

    t0 = time.perf_counter()
    r = requests.post(BASE_URL + "/api/generate", json=payload, timeout=timeout)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    r.raise_for_status()
    data = r.json()

    # Ollama returns nanoseconds; convert to milliseconds
    eval_ms = data.get("eval_duration")
    eval_ms = eval_ms / 1_000_000.0 if eval_ms else None
    prompt_eval_ms = data.get("prompt_eval_duration")
    prompt_eval_ms = prompt_eval_ms / 1_000_000.0 if prompt_eval_ms else None

    return GenerateResult(
        response=data.get("response", "").strip(),
        model=model,
        request_ms=elapsed_ms,
        eval_ms=eval_ms,
        prompt_eval_ms=prompt_eval_ms,
        eval_count=data.get("eval_count"),
        prompt_eval_count=data.get("prompt_eval_count"),
    )


if __name__ == "__main__":
    print("[*] Ollama base URL: " + BASE_URL)
    print("[*] Test generation ...")
    r = generate("Reply with exactly: PIPELINE_OK", max_tokens=16)
    print("    response: " + repr(r.response))
    print("    request: {:.1f} ms".format(r.request_ms))
    if r.prompt_eval_ms:
        print("    prompt_eval: {:.1f} ms ({} tokens)".format(
            r.prompt_eval_ms, r.prompt_eval_count or 0))
    if r.eval_ms:
        print("    eval (decode): {:.1f} ms ({} tokens)".format(
            r.eval_ms, r.eval_count or 0))
