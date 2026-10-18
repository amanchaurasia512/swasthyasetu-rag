"""English check (C-032, part 1): which questions would the rule send to the rewrite? No models, no API calls.

Rule (src.rewrite.unknown_words): a question goes to the rewrite when it has a word that is neither English nor in
the PDF's own vocabulary, not counting what the rewrite must keep as written (ALL-CAPS abbreviations, capitalised
words inside a sentence); otherwise SpellFixer handles it and only the answer costs a call.
Questions: the 4-form test (A1 clean English, A2 English typos, B1/B2 Hinglish), the manual check (manual 23, new 12,
spelling hold-out 15), set C and the negatives (written by Claude, mostly clean English; a few hold Hinglish or typos
on purpose, so the rule 3 share is an upper bound). "Rewrite helped" comes from the earlier checks: an answerable
question whose gate passed or gold chunk reached the top 5 only after the rewrite.

Pass rules (set by the author before running, 18 Oct 2026):
  1. every Hinglish question (B1, B2) is sent to the rewrite
  2. every question the rewrite helped is sent to the rewrite
  3. at most 10% of clean English questions (A1, set C, negatives) are sent to the rewrite

    python -m src.checks.english_check --forms check-forms-20261018-0902 --manual check-manual-20261018-1031
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime

import pandas as pd
from spellchecker import SpellChecker

from src.checks.level1_spelling import all_questions
from src.config import load, path_of
from src.rewrite import unknown_words
from src.run import RUNS

CLEAN_ENGLISH = ("A1_english", "set_C", "negatives")
HINGLISH = ("B1_hinglish", "B2_hinglish_typos")
MAX_CLEAN_SHARE = 10.0                                      # rule 3, chosen by the author


def helped(table: pd.DataFrame) -> set[str]:
    """qids of answerable questions that passed the gate or found the gold chunk only with the rewrite."""
    good = table.assign(ok=table.grade.isin(["S", "P"]) | table.gold_rank.notna())
    wide = good[good.label == "answerable"].pivot_table(index="qid", columns="method", values="ok", aggfunc="first")
    return set(wide.index[wide["rewrite"].astype(bool) & ~wide["spellfixer"].astype(bool)])


def questions(forms: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    """One row per question as typed: qid, group, label, question."""
    typed_forms = forms[forms.method == "original"].assign(group=lambda frame: frame.form)
    typed_manual = manual[manual.method == "original"].assign(group=lambda frame: frame.source)
    generated = pd.DataFrame([{"qid": item["qid"], "group": item["source"], "label": item["label"], "query": item["question"]}
                              for item in all_questions() if item["source"] in ("set_C", "negatives")])
    return pd.concat([frame[["qid", "group", "label", "query"]] for frame in (typed_forms, typed_manual, generated)],
                     ignore_index=True).rename(columns={"query": "question"})


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="English check: which questions would go to the rewrite")
    parser.add_argument("--forms", required=True, help="the check-forms-* folder (4-form test)")
    parser.add_argument("--manual", required=True, help="the check-manual-* folder (manual, new 12, hold-out)")
    args = parser.parse_args(argv)
    forms = pd.read_excel(RUNS / args.forms / "compare.xlsx", sheet_name="all")
    manual = pd.read_excel(RUNS / args.manual / "manual.xlsx", sheet_name="all")
    with open(path_of(load()["files"]["chunks"]), encoding="utf-8") as file:
        vocab = {word for chunk in map(json.loads, file) for word in re.findall(r"[a-z]+", chunk["embed_text"].lower())}
    english, gained = SpellChecker(), helped(forms) | helped(manual)
    table = questions(forms, manual)
    table["unknown"] = [", ".join(unknown_words(text, vocab, english)) for text in table.question]
    table["to_rewrite"] = table.unknown != ""
    table["rewrite_helped"] = table.qid.isin(gained)

    summary = table.groupby("group", sort=False).agg(questions=("qid", "size"), to_rewrite=("to_rewrite", "sum"))
    summary["to_rewrite %"] = (100 * summary.to_rewrite / summary.questions).round(1)
    clean = table[table.group.isin(CLEAN_ENGLISH)]
    clean_share = round(100 * clean.to_rewrite.mean(), 1)
    rules = pd.DataFrame([
        {"rule": "1. every Hinglish question goes to the rewrite",
         "result": f"{int(table[table.group.isin(HINGLISH)].to_rewrite.sum())} of {int(table.group.isin(HINGLISH).sum())}",
         "pass": bool(table[table.group.isin(HINGLISH)].to_rewrite.all())},
        {"rule": "2. every question the rewrite helped goes to the rewrite",
         "result": f"{int(table[table.rewrite_helped].to_rewrite.sum())} of {int(table.rewrite_helped.sum())}",
         "pass": bool(table[table.rewrite_helped].to_rewrite.all())},
        {"rule": f"3. at most {MAX_CLEAN_SHARE:g}% of clean English goes to the rewrite",
         "result": f"{clean_share}% ({int(clean.to_rewrite.sum())} of {len(clean)})", "pass": clean_share <= MAX_CLEAN_SHARE}])

    folder = RUNS / f"check-english-{datetime.now():%Y%m%d-%H%M}"
    folder.mkdir(parents=True)
    with pd.ExcelWriter(folder / "english.xlsx") as writer:
        rules.to_excel(writer, sheet_name="rules", index=False)
        summary.reset_index().to_excel(writer, sheet_name="summary", index=False)
        table.to_excel(writer, sheet_name="all", index=False)
    print(rules.to_string(index=False), "\n\n", summary.to_string(), f"\n\n-> {folder / 'english.xlsx'}", sep="")
    return table


if __name__ == "__main__":
    main()