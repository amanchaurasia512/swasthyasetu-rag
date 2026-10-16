"""Run trace: one line per question per run, facts only - no answer key, so it works for a user's question too.

Long texts are not repeated: chunk text lives in evidence.jsonl, prompts and answers in answers.csv; the trace
points to them by chunk_id, prompt_hash and the answer key (model, version, format, run, qid).
Rules: only the runner writes it; a line is appended when the question's stages 1-6 finish; it is never edited;
every stage key is present, None when the stage did not run.
"""
from __future__ import annotations

import json
from pathlib import Path

HIT_FACTS = ("rank", "chunk_id", "score", "pdf_pages", "printed_pages", "path")


def trace_line(run_id: str, doc_id: str, record: dict, built: list[dict], seconds: dict) -> dict:
    prepared = "action" in record
    return {
        "run_id": run_id, "qid": record["qid"], "doc_id": doc_id,
        "normalize": {"fixed": record["fixed"], "changes": record["changes"]} if prepared else None,
        "retrieve": {"hits": [{key: hit[key] for key in HIT_FACTS} for hit in record["hits"]]} if prepared else None,
        "gate": {"signals": record["signals"], "first": record["first"], "final": record["final"],
                 "rounds": 2 if record["first"] == "P" else 1, "action": record["action"]} if prepared else None,
        "evidence": {"hit_ids": [hit["chunk_id"] for hit in record["hits"]],
                     "fixed_answer": record["fixed_answer"]} if prepared else None,
        "prompt": [{key: item[key] for key in ("prompt_version", "format", "prompt_hash")}
                   | {"length": len(item["system"]) + len(item["prompt"])} for item in built] or None,
        "seconds": seconds,
        "error": record["error"],
    }


def log_line(record: dict, seconds: float) -> str:
    """'NHM09 S answer 5 hits | Chapter 6 > 6.1 Joining Formalities p.17 (PDF 37) | 2.1 s'"""
    if "action" not in record:
        return f"{record['qid']} ERROR {record['error']['type']}: {record['error']['message'][:80]} | {seconds:.1f} s"
    top = record["hits"][0] if record["hits"] else {"path": "-", "printed_pages": [], "pdf_pages": []}
    pages = ", ".join(str(page) for page in top["printed_pages"] if page is not None) or "?"
    pdf = ", ".join(str(page) for page in top["pdf_pages"])
    return (f"{record['qid']} {record['final']} {record['action']} {len(record['hits'])} hits | "
            f"{top['path']} p.{pages} (PDF {pdf}) | {seconds:.1f} s")


def append(path: Path, line: dict) -> None:
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(line, ensure_ascii=False) + "\n")