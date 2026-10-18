"""Query rewriting before search (Level 1 decision, C-029): the question in plain English, asked live from an LLM
(prompts.REWRITES), the same way the chatbot will ask it for a user's question. No cache: every run asks again, so
the run shows the real waiting time and the real failures. Only the runner calls it, with --allow-api-calls (rule R1).
"""
from __future__ import annotations

import time

from src import prompts
from src.llm import LLMError, chat_full


def ask(question: str, config: dict) -> dict:
    """{model, version, text, seconds, error}. text is "" when the call failed or came back empty; the pipeline then
    falls back to the SpellFixer (rule R5)."""
    query, llm = config["query"], config["llm"]
    began = time.time()
    try:
        reply = chat_full(query["rewrite_model"], prompts.REWRITES[query["rewrite_version"]], question,
                          timeout=llm["timeout"], tries=llm["tries"])
        text = " ".join(reply["text"].split())
        error = "" if text else "empty rewrite"
    except LLMError as failure:
        text, error = "", str(failure)[:200]
    return {"model": query["rewrite_model"], "version": query["rewrite_version"], "text": text,
            "seconds": round(time.time() - began, 2), "error": error}