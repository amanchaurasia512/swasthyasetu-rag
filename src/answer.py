"""SwasthyaSetu answers: retrieve the top chunks, then let a local LLM (Ollama) answer ONLY from them.

Usage
-----
    from src.answer import Answerer
    a = Answerer()                       # loads the retriever once
    result = a.ask("When will a leaving employee get the relieving letter?")
    print(result["answer"])
    for s in result["sources"]:
        print(s["n"], s["path"], "page", s["printed_pages"])

From a terminal:
    python -m src.answer "When will a leaving employee get the relieving letter?"
"""
from __future__ import annotations

import sys
import time

import requests

from src.retriever import Retriever

OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"
TOP_K        = 5           # chunks sent to the model
NOT_FOUND    = "Not found in the guideline."

SYSTEM_PROMPT = f"""You answer questions about the "Guidelines on Human Resources for Health for NHM (2022)".
Rules:
1. Use ONLY the numbered sources given to you. Do not use outside knowledge.
2. After every fact, cite its source number in square brackets, e.g. [2].
3. If the sources do not contain the answer, reply exactly: {NOT_FOUND}
4. Be short and clear: 1-4 sentences or a short list."""


def build_prompt(question: str, hits: list[dict]) -> str:
    """Number the chunks [1]..[k] with their section and page, then add the question."""
    blocks = []
    for n, h in enumerate(hits, 1):
        pages = ", ".join(str(p) for p in h["printed_pages"] if p is not None) or "?"
        blocks.append(f"[{n}] {h['path']} (page {pages})\n{h['text']}")
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\nAnswer:"


class Answerer:
    def __init__(self, model: str = OLLAMA_MODEL, retriever: Retriever | None = None):
        self.model = model
        self.retriever = retriever or Retriever()

    def generate(self, prompt: str) -> str:
        """Send one prompt to Ollama and return its reply text."""
        r = requests.post(OLLAMA_URL, timeout=300, json={
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 8192},   # 0 = same answer every time; 8192 tokens fits 5 chunks
        })
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    def ask(self, question: str, k: int = TOP_K) -> dict:
        t0 = time.time()
        hits = self.retriever.search(question, k=k)
        t1 = time.time()
        answer = self.generate(build_prompt(question, hits))
        t2 = time.time()
        sources = [{"n": n, "chunk_id": h["chunk_id"], "path": h["path"],
                    "printed_pages": h["printed_pages"], "pdf_pages": h["pdf_pages"]}
                   for n, h in enumerate(hits, 1)]
        return {"question": question, "answer": answer, "sources": sources,
                "not_found": answer.startswith(NOT_FOUND.rstrip(".")),
                "seconds_search": round(t1 - t0, 1), "seconds_answer": round(t2 - t1, 1)}


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "When will a leaving employee get the relieving letter?"
    res = Answerer().ask(question)
    print(f"Q: {res['question']}\n")
    print(res["answer"], "\n")
    for s in res["sources"]:
        print(f"[{s['n']}] {s['path'][:70]}  (page {', '.join(map(str, s['printed_pages']))})")
    print(f"\nsearch {res['seconds_search']} s | answer {res['seconds_answer']} s")