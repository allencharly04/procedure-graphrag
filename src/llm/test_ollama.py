"""Smoke test: confirm we can reach Ollama on the Windows host from WSL2."""
import subprocess
import time
import requests


def get_windows_host_ip():
    """In WSL2, the default gateway is the Windows host."""
    result = subprocess.run(
        ["ip", "route", "show"],
        capture_output=True, text=True, check=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("default"):
            return line.split()[2]
    raise RuntimeError("No default route found")


def main():
    host = get_windows_host_ip()
    base_url = "http://" + host + ":11434"
    print("[*] Windows host:", host)
    print("[*] Ollama base URL:", base_url)

    r = requests.get(base_url + "/api/tags", timeout=10)
    r.raise_for_status()
    models = [m["name"] for m in r.json().get("models", [])]
    print("[*] Available models:", models)

    if "llama3.1:latest" not in models:
        print("[!] llama3.1:latest not found; aborting generate test")
        return

    print("[*] Sending test prompt to llama3.1:latest ...")
    t0 = time.perf_counter()
    r = requests.post(
        base_url + "/api/generate",
        json={
            "model": "llama3.1:latest",
            "prompt": "Reply with exactly: ASSEMBLY_RAG_OK",
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 16},
        },
        timeout=120,
    )
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    response = r.json()["response"].strip()
    print("[*] Response:", repr(response))
    print("[*] Latency: {:.2f}s".format(elapsed))
    print("[OK] Ollama smoke test passed" if response else "[FAIL] empty response")


if __name__ == "__main__":
    main()
