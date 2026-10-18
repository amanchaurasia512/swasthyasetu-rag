"""Diagnosis of one run (C-017): for every question, the first stage where it was lost - judged by whether the
expected facts are present at each stage, with the scorer's own fact matcher (no LLM judge, no API calls).
Works for any document: it reads only the run folder, the chunks file named in the run's config, and the expected
answers stored in the run's evidence.

Ideas credited (see the restructure report, References): RAGChecker (Ru et al., NeurIPS 2024) - judge retriever and
generator separately by claim recall and context utilization; Seven Failure Points (Barnett et al., CAIN 2024) - the
stage labels (missing content, missed top-ranked documents, not in context, not extracted / incomplete).

    python -m src.diagnose e3-20261017-1200     writes runs/<run_id>/report.csv and rebuilds runs/index.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src import evaluate, pipeline
from src.config import path_of
from src.run import RUNS

STAGES = ("key", "index", "retrieval", "gate", "context", "llm_refused", "llm_incomplete", "over_answer", "ok")


def lost_at(row: dict) -> str:
    """The first stage where the question failed (Seven Failure Points order); 'ok' when every run was right."""
    if row["label"] != "answerable":
        return "ok" if row["action"] not in pipeline.TO_LLM or all(row["refused"]) else "over_answer"
    if not row["facts_total"]:
        return "key"
    if not row["facts_in_index"]:
        return "index"
    if not row["facts_in_hits"]:
        return "retrieval"
    if row["action"] not in pipeline.TO_LLM:
        return "gate"
    if not row["facts_in_prompt"]:
        return "context"
    if all(row["correct"]):
        return "ok"
    return "llm_refused" if any(row["refused"]) else "llm_incomplete"


def found_in_any(facts: list[str], texts: list[str]) -> list[str]:
    """Facts that appear (by the scorer's rule) inside at least one single text - never in the texts glued together."""
    return [fact for fact in facts if any(evaluate.found_in([fact], text) for text in texts)]


def labelled(item: dict) -> str:
    """A chunk or hit as the model sees it: path and both page numbers, then the text (page facts live in metadata)."""
    path = " > ".join(item["path"]) if isinstance(item["path"], list) else item["path"]
    printed = ", ".join(str(page) for page in item["printed_pages"] if page is not None)
    return f"{path} | printed page {printed} | PDF page {', '.join(map(str, item['pdf_pages']))}\n{item['text']}"


def question_row(record: dict, answers: pd.DataFrame, chunk_text: dict) -> dict:
    facts = evaluate.facts(record["expected"]) if record["label"] == "answerable" and record["expected"] else []
    hits = record.get("hits", [])
    in_hits = found_in_any(facts, [chunk_text.get(hit["chunk_id"], "") for hit in hits])
    in_prompt = found_in_any(facts, [labelled(hit) for hit in hits])
    runs = answers.sort_values("run")
    return {
        "qid": record["qid"], "source": record["source"], "label": record["label"], "question": record["question"],
        "fixed": record.get("fixed", ""), "spelling_changes": json.dumps(record.get("changes", []), ensure_ascii=False),
        "grade": record.get("final", ""), "action": record.get("action", ""),
        "best_score": record.get("signals", {}).get("best"), "hard_out": ", ".join(record.get("signals", {}).get("hard_out", [])),
        "top_hit": f"{hits[0]['path']} | printed {hits[0]['printed_pages']} | PDF {hits[0]['pdf_pages']}" if hits else "",
        "facts_total": len(facts), "facts_in_index": len(found_in_any(facts, list(chunk_text.values()))),
        "facts_in_hits": len(in_hits), "facts_in_prompt": len(in_prompt),
        "facts_in_answer": [len(evaluate.found_in(in_prompt, answer)) for answer in runs.answer.fillna("")],
        "refused": runs.refused.astype(bool).tolist(),
        "correct": (~runs.refused.astype(bool) & runs.correctness.astype(float).fillna(0).ge(evaluate.TAU)).tolist(),
        "answers": runs.answer.fillna("").tolist(),
        "fixed_answer": record.get("fixed_answer", ""),
        "expected": record["expected"],
    }


def report(folder: Path) -> pd.DataFrame:
    """One row per question of the run, with lost_at."""
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    with open(path_of(manifest["config"]["merged"]["files"]["chunks"]), encoding="utf-8") as file:
        chunk_text = {chunk["chunk_id"]: labelled(chunk) for chunk in map(json.loads, file)}
    with open(folder / "evidence.jsonl", encoding="utf-8") as file:
        records = [json.loads(line) for line in file]
    scored = pd.read_csv(folder / "scored.csv") if (folder / "scored.csv").exists() else pd.DataFrame(
        columns=["qid", "run", "answer", "refused", "correctness"])
    rows = [question_row(record, scored[scored.qid == record["qid"]], chunk_text) for record in records]
    table = pd.DataFrame(rows)
    table.insert(3, "lost_at", [lost_at(row) for row in rows])
    for name in ("facts_in_answer", "refused", "correct", "answers"):
        for run in range(max((len(row[name]) for row in rows), default=0)):
            table[f"{name}_r{run + 1}"] = [row[name][run] if run < len(row[name]) else None for row in rows]
        table = table.drop(columns=name)
    return table


def index_row(folder: Path) -> dict:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    row = {"date": manifest["started_at"][:10], "run_id": manifest["run_id"], "experiment": manifest["experiment"],
           "status": manifest["status"], "commit": manifest["code"]["git_commit"], "dirty": manifest["code"]["git_dirty"],
           "questions": manifest["plan"].get("questions"), "calls": manifest["plan"].get("planned_calls"),
           "answers": manifest["progress"].get("answers_saved"), "truncated": manifest.get("scoring", {}).get("truncated")}
    if (folder / "scorecard.csv").exists():
        cards = pd.read_csv(folder / "scorecard.csv")
        row["scorecard"] = "; ".join(f"{card['model']} {card['version']}: {card['correct on answerable %']}% correct, "
                                     f"{card['over-answer on unanswerable %']}% over-answer"
                                     for card in cards.to_dict("records"))
    if (folder / "report.csv").exists():
        counts = pd.read_csv(folder / "report.csv").lost_at.value_counts()
        row |= {f"lost_at_{stage}": int(counts.get(stage, 0)) for stage in STAGES}
    return row


def main(argv: list[str] | None = None, runs_dir: Path = RUNS) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="Diagnose one run (no API calls)")
    parser.add_argument("run_id")
    args = parser.parse_args(argv)
    folder = runs_dir / args.run_id
    table = report(folder)
    table.to_csv(folder / "report.csv", index=False)
    index = pd.DataFrame([index_row(run) for run in sorted(runs_dir.iterdir()) if (run / "manifest.json").exists()])
    index.sort_values(["date", "run_id"]).to_csv(runs_dir / "index.csv", index=False)
    print(f"{args.run_id}: {len(table)} questions -> report.csv")
    print(table.lost_at.value_counts().reindex(STAGES, fill_value=0).to_string())
    return table


if __name__ == "__main__":
    main()