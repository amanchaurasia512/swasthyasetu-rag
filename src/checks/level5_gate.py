"""Level 5 check - the gate (corrective.Corrective.signals + grade + re_retrieve), on the input it will get from now
on: the Gemini English rewrite of each question (Level 1 decision). Offline - rewrites are read from the Level 1b
and Level 1d folders, no API calls. Four sheets:
    rules       which rule decided each question, counted for answerable (blocked = wrong) and unanswerable (blocked = right)
    thresholds  for every cut-off of the best rerank score: answerable passing vs unanswerable blocked (data, not a hand pick)
    hard_words  every M decision with the word or number that triggered it
    re_retrieve P questions: grade and gold rank before and after the second search

    python -m src.checks.level5_gate check-l1b-20261017-2215 check-forms-20261018-0902
"""
from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from src import pipeline
from src.checks.level1_spelling import all_questions
from src.checks.level1c_translate_search import gold_chunks, gold_rank
from src.checks.level1d_forms import FILE, queries
from src.config import load
from src.corrective import LOWER, UPPER
from src.run import RUNS

MODEL = "gemini:gemini-3.5-flash-lite"


def rewrites_of(folder: str) -> dict[str, str]:
    replies = pd.read_csv(RUNS / folder / "rewrites.csv", dtype=str).fillna("")
    replies = replies[(replies.model == MODEL) & (replies.error == "")].drop_duplicates("qid", keep="last")
    return dict(zip(replies.qid, replies.rewrite))


def items(benchmark_folder: str, forms_folder: str) -> list[dict]:
    """The 185 benchmark questions and the 92 form queries, each with its rewrite, label and gold chunks."""
    benchmark, forms, gold = rewrites_of(benchmark_folder), rewrites_of(forms_folder), gold_chunks()
    return ([{"set": "benchmark", "form": "", "qid": question["qid"], "label": question["label"],
              "rewrite": benchmark[question["qid"]], "gold": gold.get(question["qid"], [])}
             for question in all_questions() if benchmark.get(question["qid"])]
            + [{"set": "forms", "form": item["form"], "qid": item["qid"], "label": item["label"],
                "rewrite": forms[item["qid"]], "gold": item["gold"]}
               for item in queries(FILE) if forms.get(item["qid"])])


def reasons(signals: dict) -> list[str]:
    """Every rule of grade() that fires, in grade()'s order; the first one is the deciding rule."""
    fired = [f"I: best {signals['best']} < {LOWER}" if signals["best"] < LOWER else "",
             f"M: not in PDF {signals['hard_out']}" if signals["hard_out"] else "",
             f"P: soft word not in PDF {signals['soft_out']}" if signals["soft_out"] else "",
             f"P: not in hits {signals['missing']}" if signals["missing"] else "",
             f"P: best {signals['best']} < {UPPER}" if signals["best"] < UPPER else ""]
    return [reason for reason in fired if reason] or ["S"]


def gate_row(gate, item: dict) -> dict:
    hits = gate.R.search(item["rewrite"], k=5)
    first_signals = gate.signals(item["rewrite"], hits)
    first = gate.grade(first_signals)
    row = {**{key: item[key] for key in ("set", "form", "qid", "label", "rewrite")}, "gold": ", ".join(item["gold"]),
           "best": first_signals["best"], "first": first, "rule": reasons(first_signals)[0],
           "all_rules": " | ".join(reasons(first_signals)), "hard_out": ", ".join(first_signals["hard_out"]),
           "gold_rank_first": gold_rank(hits, item["gold"]) if item["gold"] else None,
           "final": first, "gold_rank_final": None, "best_final": None}
    if first == "P":
        hits = gate.re_retrieve(item["rewrite"], first_signals)
        final_signals = gate.signals(item["rewrite"], hits)
        row |= {"final": gate.grade(final_signals), "best_final": final_signals["best"]}
    row["gold_rank_final"] = gold_rank(hits, item["gold"]) if item["gold"] else None
    return row


def rules(table: pd.DataFrame) -> pd.DataFrame:
    kind = table.rule.str.replace(r"\s*\[.*\]|\s*-?[\d.]+ < -?[\d.]+", "", regex=True)       # "M: not in PDF ['DMC']" -> "M: not in PDF"
    counts = pd.crosstab(kind, table.label).reindex(columns=["answerable", "unanswerable"], fill_value=0)
    return counts.rename_axis("deciding rule").reset_index()


def thresholds(table: pd.DataFrame) -> pd.DataFrame:
    answerable, unanswerable = table[table.label == "answerable"].best, table[table.label != "answerable"].best
    return pd.DataFrame([{"cut-off": cut, "answerable passing %": round(100 * (answerable >= cut).mean(), 1),
                          "unanswerable blocked %": round(100 * (unanswerable < cut).mean(), 1),
                          "answerable lost": int((answerable < cut).sum()), "unanswerable let through": int((unanswerable >= cut).sum())}
                         for cut in np.arange(-12, 8.5, 0.5)])


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="Level 5: the gate, on Gemini rewrites (no API calls)")
    parser.add_argument("benchmark_folder", help="check-l1b-* folder (185 questions)")
    parser.add_argument("forms_folder", help="check-forms-* folder (92 form queries)")
    args = parser.parse_args(argv)
    work = items(args.benchmark_folder, args.forms_folder)
    print(f"{len(work)} queries")
    gate = pipeline.open_gate(load())
    table = pd.DataFrame([gate_row(gate, item) for item in work])
    folder = RUNS / f"check-l5-{datetime.now():%Y%m%d-%H%M}"
    folder.mkdir(parents=True, exist_ok=True)
    rule_counts = rules(table)
    with pd.ExcelWriter(folder / "gate.xlsx") as writer:
        rule_counts.to_excel(writer, sheet_name="rules", index=False)
        thresholds(table).to_excel(writer, sheet_name="thresholds", index=False)
        table[table["first"] == "M"].to_excel(writer, sheet_name="hard_words", index=False)
        table[table["first"] == "P"].to_excel(writer, sheet_name="re_retrieve", index=False)
        table.to_excel(writer, sheet_name="all", index=False)
    print(rule_counts.to_string(index=False))
    print(f"-> {folder / 'gate.xlsx'}")
    return table


if __name__ == "__main__":
    main()