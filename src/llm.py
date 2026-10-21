"""One way to call every LLM used in SwasthyaSetu: local Ollama models and free API models.

Model names
-----------
    "qwen2.5:7b", "llama3.1:8b", ...      -> Ollama (laptop or notebook GPU); a dead server is restarted automatically
    "groq:openai/gpt-oss-120b"            -> Groq (free tier), needs env GROQ_API_KEY

(GitHub Models was retired on 30 July 2026 - its endpoint answers "OK" to everything, so it is removed.)
Keys are read from environment variables only (laptop: the gitignored .env; Kaggle / Colab: the secrets panel).

Usage
-----
    from src.llm import chat, real_answer_check
    real_answer_check("groq:openai/gpt-oss-120b")          # stops with LLMError if the model does not really answer
    reply = chat("qwen2.5:7b", system="You are ...", user="Question: ...")
"""
from __future__ import annotations

import os
import re
import time

import requests

from src.ollama_server import OLLAMA_URL, OllamaError, ensure_ollama

API = {"groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
       "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "GEMINI_API_KEY"),
       "nvidia": ("https://integrate.api.nvidia.com/v1/chat/completions", "NVIDIA_API_KEY"),
        "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
       "sarvam": ("https://api.sarvam.ai/v1/chat/completions", "SARVAM_API_KEY")}
       
API_MAX_TOKENS = 4096                          # one generous limit for every API model, so a comparison is fair (C-036)
OPTIONS = {"temperature": 0, "num_ctx": 4096, "num_predict": 512}
RETRY_STATUS = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """The model did not answer after all retries (timeout, rate limit, server error, missing key, fake answers)."""

def provider_of(model: str) -> tuple[str, str]:
    """'groq:openai/gpt-oss-120b' -> ('groq', 'openai/gpt-oss-120b'); 'qwen2.5:7b' -> ('ollama', 'qwen2.5:7b')."""
    prefix, separator, rest = model.partition(":")
    return (prefix, rest) if separator and prefix in API else ("ollama", model)

def clean(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()

def _ollama(name: str, messages: list[dict], timeout: int):
    payload = {"model": name, "stream": False, "messages": messages, "options": OPTIONS}
    if name.startswith("qwen3"):
        payload["think"] = False                     # hybrid qwen3 models: answer directly, no reasoning text
    reply = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
    return reply, (lambda response: response.json()["message"]["content"])

def api_text(response) -> str:
    """OpenAI-style JSON reply -> its message text. Anything else is NOT an answer (a dead service can reply 'OK')."""
    if "json" not in response.headers.get("content-type", ""):
        raise LLMError(f"not a model reply (content-type {response.headers.get('content-type')!r}): {response.text[:100]!r}")
    data = response.json()
    if not data.get("choices"):
        raise LLMError(f"no answer in the reply: {str(data.get('error', data))[:200]}")
    return data["choices"][0]["message"].get("content") or ""

def _api(provider: str, name: str, messages: list[dict], timeout: int):
    url, key_name = API[provider]
    key = os.environ.get(key_name, "")
    if not key:
        raise LLMError(f"{key_name} is not set - add it to .env (laptop) or the notebook secrets")
    payload = {"model": name, "messages": messages, "temperature": 0, "max_tokens": API_MAX_TOKENS}
    if "gpt-oss" in name:                      # reasoning model: think briefly and leave room for the answer
        payload["reasoning_effort"] = "low"
    if provider == "gemini":                   # Gemini 3.x thinks by default: keep it low, leave room for long lists
        payload.update({"reasoning_effort": "low", "max_tokens": 4096})
    if provider == "sarvam":                   # Sarvam 105B reasons by default (medium); thinking counts in max_tokens
        payload.update({"reasoning_effort": "low", "max_tokens": 4096})
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json", "Content-Type": "application/json"}
    reply = requests.post(url, json=payload, timeout=timeout, headers=headers)
    return reply, api_text

def api_reply(response) -> dict:
    """OpenAI-style reply -> text, token counts and finish_reason ('stop' = finished, 'length' = cut off at max_tokens).
    A count the provider does not report is None. answer_tokens is completion_tokens as reported (for OpenAI-style
    reasoning models it includes the thinking; reasoning_tokens is that part, when reported)."""
    text = api_text(response)
    data = response.json()
    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return {"text": text, "prompt_tokens": usage.get("prompt_tokens"), "answer_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": details.get("reasoning_tokens"), "finish_reason": data["choices"][0].get("finish_reason")}

def ollama_reply(response) -> dict:
    """Ollama reply -> the same fields as api_reply (Ollama names them prompt_eval_count, eval_count, done_reason)."""
    data = response.json()
    return {"text": data["message"]["content"], "prompt_tokens": data.get("prompt_eval_count"),
            "answer_tokens": data.get("eval_count"), "reasoning_tokens": None, "finish_reason": data.get("done_reason")}

def chat_full(model: str, system: str, user: str, timeout: int = 300, tries: int = 3) -> dict:
    """Send one system + user message (same request as chat), return the reply text with token use and finish_reason.
    Retries timeouts, 429 and 5xx with a growing wait; an Ollama server that stopped answering is restarted."""
    provider, name = provider_of(model)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    problem = ""
    for attempt in range(1, tries + 1):
        try:
            reply = _ollama(name, messages, timeout)[0] if provider == "ollama" else _api(provider, name, messages, timeout)[0]
            read = ollama_reply if provider == "ollama" else api_reply
            if reply.status_code in RETRY_STATUS:
                problem = f"HTTP {reply.status_code}: {reply.text[:200]}"
                wait = int(reply.headers.get("retry-after", 0) or 0) or 10 * attempt
                time.sleep(min(wait, 120))
                continue
            reply.raise_for_status()
            result = read(reply)
            return {**result, "text": clean(result["text"])}
        except requests.RequestException as error:
            problem = f"{type(error).__name__}: {str(error)[:200]}"
            if provider == "ollama":
                try:
                    ensure_ollama()                  # restarts the server only if it no longer answers
                except OllamaError as start_error:
                    raise LLMError(str(start_error)) from error
            time.sleep(5 * attempt)
    raise LLMError(f"{model} gave no answer after {tries} tries | {problem}")

def chat(model: str, system: str, user: str, timeout: int = 300, tries: int = 3) -> str:
    """Send one system + user message, return the reply text only (as before; chat_full also returns token use)."""
    return chat_full(model, system, user, timeout, tries)["text"]

def real_answer_check(model: str, timeout: int = 300, tries: int = 3) -> str:
    """17 + 25 must give 42. A dead or fake endpoint ('OK', empty, an error page) fails here, before a long run."""
    reply = chat(model, "Answer with the number only.", "What is 17 + 25?", timeout, tries)
    if not re.search(r"\b42\b", reply):
        raise LLMError(f"{model} failed the real-answer check: 17 + 25 -> {reply[:100]!r}")
    return reply

class FakeAnswerGuard:
    """Stops a long run when the model keeps sending the SAME text (e.g. 288 x 'OK').
    The normal refusal (e.g. 'Not found in the guideline.') may repeat, so it is not counted."""

    def __init__(self, model: str, allowed_repeats: tuple[str, ...] = (), limit: int = 10):
        self.model, self.allowed, self.limit = model, {text.strip() for text in allowed_repeats}, limit
        self.last, self.repeats = "", 0

    def check(self, answer: str) -> None:
        text = answer.strip()
        if text in self.allowed:
            self.last, self.repeats = "", 0
            return
        self.repeats = self.repeats + 1 if text == self.last else 1
        self.last = text
        if self.repeats >= self.limit:
            raise LLMError(f"{self.model} sent the same reply {self.repeats} times in a row: {text[:80]!r} - "
                           f"probably not a real model answer; check the provider before re-running")
