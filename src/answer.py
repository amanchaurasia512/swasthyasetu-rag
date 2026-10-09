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

from src.query import SpellFixer #build SpellingFixture 


import sys
import time
import re

import requests

from src.retriever import Retriever ,unit_of


OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"
TOP_K        = 5           # chunks sent to the model
NOT_FOUND    = "Not found in the guideline."
MIN_RERANK_SCORE = -5.0   # notebook 08.10: all 200 real questions pass, 19/20 off-topic blocked

## 1. Read ALL the numbered sources first. The answer can be in any source, not only [1]. #this system prompt was has it was casue the to much reasoning  

SYSTEM_PROMPT = f"""You answer questions about the "Guidelines on Human Resources for Health for NHM (2022)".
How to answer:
1. Use all provided sources as needed. The answer may be in any source.
2. Use ONLY these sources. No outside knowledge, no guessing.
3. Give the COMPLETE answer. If a source lists items (members, conditions, steps, documents), include every item.
4. After each fact write the number of the source you read it in, e.g. [4].
5. Tables: read row by row and keep each row with its column names. For "only" questions, choose rows where that column is Yes and the other columns are No.
6. If the sources do not answer the question, reply exactly: {NOT_FOUND}
7. Be short: 1-5 sentences or a short list. No extra advice."""


def build_prompt(question: str, hits: list[dict]) -> str:
    """Question first (so the model knows what to look for), numbered sources, then the question again."""
    blocks = []
    for n, h in enumerate(hits, 1):
        pages = ", ".join(str(p) for p in h["printed_pages"] if p is not None) or "?"
        kind = " (table)" if h["doc_type"] == "table" else ""
        blocks.append(f"[{n}]{kind} {h['path']} (page {pages})\n{h['text']}")
    return (f"Question: {question}\n\nSources:\n\n" + "\n\n".join(blocks)
            + f"\n\nAnswer this question using the sources above: {question}\nAnswer:")

CONTEXT_TOP  = 3      # expand only the top 3 hits
CONTEXT_TAIL = 700    # characters taken from the end of the previous chunk


def merge_pages(first: list, second: list) -> list:
    """Join two page lists without repeats: ['51'] + ['51', '52'] -> ['51', '52']."""
    pages = []
    for page in first + second:
        if page not in pages:
            pages.append(page)
    return pages


def add_context(hits: list[dict], retriever: Retriever,
                top: int = CONTEXT_TOP, tail: int = CONTEXT_TAIL) -> list[dict]:
    """For the top hits, put the end of the previous chunk (same section) in front of the text,
    so a chunk that starts in the middle of a paragraph gets its beginning back."""
    hit_ids = [hit["chunk_id"] for hit in hits]
    new_hits = []
    for rank, hit in enumerate(hits, 1):
        hit = dict(hit)                                    # a copy, so the retriever's result is not changed
        row = retriever.row_of[hit["chunk_id"]]
        if rank <= top and row > 0:
            prev = retriever.chunks[row - 1]               # the chunk just before it in chunks.jsonl
            same_section = unit_of(prev) == hit["unit"] and prev["doc_type"] == hit["doc_type"]
            already_a_hit = prev["chunk_id"] in hit_ids
            if same_section and not already_a_hit:
                hit["text"] = "..." + prev["text"][-tail:] + "\n" + hit["text"]
                hit["printed_pages"] = merge_pages(prev["printed_pages"], hit["printed_pages"])
                hit["pdf_pages"] = merge_pages(prev["pdf_pages"], hit["pdf_pages"])
        new_hits.append(hit)
    return new_hits



class Answerer:
    # def __init__(self, model: str = OLLAMA_MODEL, retriever: Retriever | None = None):
    #     self.model = model
    #     self.retriever = retriever or Retriever()

    def __init__(self, model: str = OLLAMA_MODEL, retriever: Retriever | None = None,
                 system_prompt: str = SYSTEM_PROMPT):
        self.model = model
        self.retriever = retriever or Retriever()
        self.system_prompt = system_prompt
        self.speller = SpellFixer(self.retriever.chunks)

    # def generate(self, prompt: str) -> str:
    #     """Send one prompt to Ollama and return its reply text."""
    #     r = requests.post(OLLAMA_URL, timeout=300, json={
    #         "model": self.model,
    #         # "messages": [{"role": "system", "content": SYSTEM_PROMPT},
    #         "messages": [{"role": "system", "content": self.system_prompt},
    #                      {"role": "user", "content": prompt}],
    #         "stream": False,
    #         "options": {"temperature": 0, "num_ctx": 8192},   # 0 = same answer every time; 8192 tokens fits 5 chunks
    #     })
    #     r.raise_for_status()
    #     return r.json()["message"]["content"].strip()
    def generate(self, prompt: str) -> str:
        """Send one prompt to Ollama and return its reply text."""
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.system_prompt},
                         {"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 8192},
        }
        if "qwen3" in self.model:
            payload["think"] = False          # qwen3: skip the slow thinking step
        r = requests.post(OLLAMA_URL, timeout=600, json=payload)
        r.raise_for_status()
        text = r.json()["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()    

    # def ask(self, question: str, k: int = TOP_K) -> dict:
    #     t0 = time.time()
    #     hits = self.retriever.search(question, k=k)
    #     t1 = time.time()
    #     answer = self.generate(build_prompt(question, hits))
    #     t2 = time.time()
    #     sources = [{"n": n, "chunk_id": h["chunk_id"], "path": h["path"],
    #                 "printed_pages": h["printed_pages"], "pdf_pages": h["pdf_pages"]}
    #                for n, h in enumerate(hits, 1)]
    #     return {"question": question, "answer": answer, "sources": sources,
    #             "not_found": answer.startswith(NOT_FOUND.rstrip(".")),
    #             "seconds_search": round(t1 - t0, 1), "seconds_answer": round(t2 - t1, 1)}

    # def ask(self, question: str, k: int = TOP_K) -> dict:
    #     t0 = time.time()
    #     fixed, spelling = self.speller.fix(question)              # 1. query processing (spelling only)
    #     hits = self.retriever.search(fixed, k=k)                  # 2. BM25 + vector -> RRF -> rerank
    #     best = max(h["score"] for h in hits)
    #     t1 = time.time()
    #     if best < MIN_RERANK_SCORE:                               # 3. corpus gate: nothing relevant
    #         answer, hits, by = NOT_FOUND, [], "gate"
    #     else:                                                     # 4. LLM: sufficiency + answer + citations
    #         answer = self.generate(build_prompt(fixed, hits))
    #         by = "llm" if answer.startswith(NOT_FOUND.rstrip(".")) else None

    #     t2 = time.time()
    #     sources = [{"n": n, "chunk_id": h["chunk_id"], "path": h["path"],
    #                 "printed_pages": h["printed_pages"], "pdf_pages": h["pdf_pages"]}
    #                for n, h in enumerate(hits, 1)]
    #     return {"question": question, "fixed_question": fixed, "spelling": spelling,
    #             "answer": answer, "sources": sources,
    #             "not_found": by is not None, "not_found_by": by, "best_score": round(best, 2),
    #             "seconds_search": round(t1 - t0, 1), "seconds_answer": round(t2 - t1, 1)}
     

    def ask(self, question: str, k: int = TOP_K) -> dict:
        t0 = time.time()

        # 1. query processing: fix spelling only
        fixed, spelling = self.speller.fix(question)

        # 2. search: BM25 + vector -> RRF -> rerank
        hits = self.retriever.search(fixed, k=k)
        best = max(hit["score"] for hit in hits)
        t1 = time.time()

        # 3. corpus gate, then 4. LLM
        if best < MIN_RERANK_SCORE:
            answer = NOT_FOUND
            hits = []
            not_found_by = "gate"
        else:
            hits = add_context(hits, self.retriever)
            answer = self.generate(build_prompt(fixed, hits))
            if answer.startswith("Not found"):
                not_found_by = "llm"
            else:
                not_found_by = None
        t2 = time.time()

        sources = []
        for n, hit in enumerate(hits, 1):
            sources.append({"n": n, "chunk_id": hit["chunk_id"], "path": hit["path"],
                            "printed_pages": hit["printed_pages"], "pdf_pages": hit["pdf_pages"]})

        return {"question": question, "fixed_question": fixed, "spelling": spelling,
                "answer": answer, "sources": sources,
                "not_found": not_found_by is not None, "not_found_by": not_found_by,
                "best_score": round(best, 2),
                "seconds_search": round(t1 - t0, 1), "seconds_answer": round(t2 - t1, 1)}        

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "When will a leaving employee get the relieving letter?"
    res = Answerer().ask(question)
    print(f"Q: {res['question']}\n")
    if res["spelling"]:
        print("Did you mean:", res["fixed_question"], "\n")
    print(res["answer"], "\n")
    for s in res["sources"]:
        print(f"[{s['n']}] {s['path'][:70]}  (page {', '.join(map(str, s['printed_pages']))})")
    print(f"\nsearch {res['seconds_search']} s | answer {res['seconds_answer']} s")
    