"""Keep the Ollama server alive: install it (Linux / Kaggle / Colab), start it, restart it when it stops answering.

    ensure_ollama()
       |-- server answers?            -> done (about 1 second, safe to call any time)
       |-- not installed?             -> download the install script, run it, check the result (3 tries)
       |-- start "ollama serve" in its own session (a notebook kernel crash does not kill it)
       |-- not answering after 60 s?  -> kill it and start again (3 tries)
       v
    still not answering -> OllamaError with the last lines of the log (the notebook stops with a clear message)

The log file is OLLAMA_LOG (default: ollama.log in the current folder; on Kaggle set it to /kaggle/working/ollama.log).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
INSTALL_SCRIPT = "https://ollama.com/install.sh"
START_WAIT = 60            # seconds to wait for a fresh server to answer


class OllamaError(RuntimeError):
    """Ollama could not be installed or started."""


def log_path() -> Path:
    return Path(os.environ.get("OLLAMA_LOG", "ollama.log"))


def log_tail(lines: int = 15) -> str:
    path = log_path()
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:]) if path.exists() else "(no log yet)"


def running(timeout: int = 3) -> bool:
    try:
        return requests.get(f"{OLLAMA_URL}/api/version", timeout=timeout).ok
    except requests.RequestException:
        return False


def install(tries: int = 3) -> None:
    """Install Ollama with the official script. Fails loudly instead of silently (no 'curl | sh')."""
    if shutil.which("ollama"):
        return
    if not sys.platform.startswith("linux"):
        raise OllamaError("Ollama is not installed - install the app from https://ollama.com and run it")
    if not shutil.which("zstd") and shutil.which("apt-get"):         # the installer unpacks a .tar.zst
        subprocess.run("apt-get -qq update && apt-get -qq install -y zstd pciutils lshw", shell=True, capture_output=True)
    script = Path(tempfile.gettempdir()) / "ollama_install.sh"
    problem = ""
    print("installing Ollama (1-3 min) ...")
    for attempt in range(1, tries + 1):
        steps = [["curl", "-fsSL", "--retry", "3", "-o", str(script), INSTALL_SCRIPT], ["sh", str(script)]]
        for step in steps:
            result = subprocess.run(step, capture_output=True, text=True, timeout=900)
            if result.returncode != 0:
                problem = f"{step[0]} failed: {(result.stderr or result.stdout).strip()[-300:]}"
                break
        if shutil.which("ollama"):
            print("Ollama installed")
            return
        print(f"install try {attempt}/{tries} failed | {problem}")
        time.sleep(15 * attempt)
    raise OllamaError(f"Ollama could not be installed after {tries} tries | {problem}")


def stop() -> None:
    if sys.platform.startswith("linux"):
        subprocess.run(["pkill", "-x", "ollama"], capture_output=True)     # the server and its model runners
        time.sleep(3)


def start() -> bool:
    log = open(log_path(), "a")
    subprocess.Popen(["ollama", "serve"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for second in range(START_WAIT):
        if running():
            return True
        time.sleep(1)
    return False


def ensure_ollama(tries: int = 3) -> None:
    """Make sure the server answers; install / start / restart it if needed. Raises OllamaError if it cannot."""
    if running():
        return
    install()
    for attempt in range(1, tries + 1):
        stop()                                     # a stuck server holds the port: kill it first
        if start():
            print(f"Ollama server running (start {attempt}) | log: {log_path()}")
            return
        print(f"start {attempt}/{tries}: no answer after {START_WAIT}s")
    raise OllamaError(f"Ollama server did not start after {tries} tries. Last log lines:\n{log_tail()}")
