"""Query rewriting before search (Level 1 decision, C-029): the question in plain English, asked live from an LLM
(prompts.REWRITES), the same way the chatbot will ask it for a user's question. No cache: every run asks again, so
the run shows the real waiting time and the real failures. Only the runner calls it, with --allow-api-calls (rule R1).
"""
from __future__ import annotations

import time
import re

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

def unknown_words(question: str, vocab, english) -> list[str]:
    """Words of the question that are neither English (english: a SpellChecker) nor in the PDF's own vocabulary
    (vocab: lower-case words of the chunks). Any such word sends the question to the rewrite (C-032). Words the
    rewrite must keep as written (prompts.REWRITES) never count: ALL-CAPS abbreviations and capitalised words inside
    a sentence ("ICU", "Infosys"); the first word of a sentence still counts ("Probaton kitne ..."). Single letters,
    numbers and contractions ("I've") are left out."""
    found = []
    for match in re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?", question):
        word, before = match.group(), question[:match.start()].rstrip()
        starts_sentence = not before or before[-1] in ".?!"
        kept_as_written = word.isupper() or (word[0].isupper() and not starts_sentence)
        if len(word) > 1 and "'" not in word and not kept_as_written \
                and word.lower() not in vocab and not english.known([word.lower()]):
            found.append(word)
    return found

def needed(question: str, front_end: str, speller) -> bool:
    """Does this question go to the live rewrite? (config query.front_end) spellfixer: never; rewrite: always;
    rewrite_if_unknown: only when it has a word that is neither English nor in the PDF (unknown_words, C-032).
    speller: the gate's SpellFixer, which holds the PDF vocabulary and the English dictionary."""
    return front_end == "rewrite" or (front_end == "rewrite_if_unknown"
                                      and bool(unknown_words(question, speller.vocab, speller.english)))