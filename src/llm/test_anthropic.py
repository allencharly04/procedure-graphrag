"""Smoke test: confirm Anthropic API works."""
import os

from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

resp = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=64,
    messages=[
        {"role": "user", "content": "Reply with exactly: PROCEDURE_GRAPHRAG_OK"},
    ],
)
text = resp.content[0].text.strip()
print("[*] Response:", repr(text))
print("[*] Tokens used:", resp.usage.input_tokens, "in /", resp.usage.output_tokens, "out")
print("[OK] Anthropic API smoke test passed" if "PROCEDURE_GRAPHRAG_OK" in text else "[FAIL]")
