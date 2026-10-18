"""Level 1 check - spelling fix 1 (query.SpellFixer): does it help search?
For every question the fixer changes, search with the original question and with the fixed one (same retriever,
top 5 after rerank) and put both lists side by side for reading. Only the spelling fix is applied - no aliases,
no second typo fix, no gate - so the difference comes from this level alone. No API calls.

    python -m src.checks.level1_spelling        writes runs/check-l1-<YYYYMMDD>-<HHMM>/compare.xlsx
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from src import pipeline
from src.config import load
from src.run import RUNS

E3 = "config/experiments/e3.yaml"
TOP_K = 5


def all_questions() -> list[dict]:
    """The 173 E2 questions and the 12 new E3 ones, each once."""
    return list({question["qid"]: question for config in (load(), load(E3))
                 for question in pipeline.load_questions(config)}.values())


def hit_lines(hits: list[dict]) -> str:
    return "\n".join(f"{rank}. {hit['chunk_id']} ({hit['score']:.2f}) {hit['path']} | {hit['text'][:120]}"
                     for rank, hit in enumerate(hits, 1))


def compare_row(question: dict, gate) -> dict | None:
    fixed, changes = gate.SP.fix(question["question"])
    if not changes:
        return None
    without, with_fix = gate.R.search(question["question"], k=TOP_K), gate.R.search(fixed, k=TOP_K)
    ids_without, ids_with = [hit["chunk_id"] for hit in without], [hit["chunk_id"] for hit in with_fix]
    return {"qid": question["qid"], "label": question["label"], "question": question["question"], "fixed": fixed,
            "changes": ", ".join(f"{old} -> {new}" for old, new in changes.items()),
            "same_top5": ids_without == ids_with, "same_set": set(ids_without) == set(ids_with),
            "new_in_top5": ", ".join(chunk for chunk in ids_with if chunk not in ids_without),
            "lost_from_top5": ", ".join(chunk for chunk in ids_without if chunk not in ids_with),
            "best_without": round(without[0]["score"], 2), "best_with": round(with_fix[0]["score"], 2),
            "top5_without_fix": hit_lines(without), "top5_with_fix": hit_lines(with_fix),
            "expected": question.get("expected", ""), "verdict": "", "note": ""}


def summary(table: pd.DataFrame, total: int) -> pd.DataFrame:
    return pd.DataFrame([
        ("questions checked", total),
        ("changed by the fixer", len(table)),
        ("same top 5, same order", int(table.same_top5.sum())),
        ("same top 5, other order", int((table.same_set & ~table.same_top5).sum())),
        ("different chunks in top 5 - read these", int((~table.same_set).sum())),
        ("verdict values", "helped / hurt / no effect"),
    ], columns=["item", "value"])


def main() -> pd.DataFrame:
    gate = pipeline.open_gate(load())
    questions = all_questions()
    table = pd.DataFrame([row for row in (compare_row(question, gate) for question in questions) if row])
    table = table.sort_values(["same_set", "qid"]).reset_index(drop=True)
    folder = RUNS / f"check-l1-{datetime.now():%Y%m%d-%H%M}"
    folder.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(folder / "compare.xlsx") as writer:
        table.to_excel(writer, sheet_name="compare", index=False)
        summary(table, len(questions)).to_excel(writer, sheet_name="summary", index=False)
    print(summary(table, len(questions)).to_string(index=False))
    print(f"-> {folder / 'compare.xlsx'}")
    return table


if __name__ == "__main__":
    main()