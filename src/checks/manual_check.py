"""Manual-question check - every question written by hand (manual NHM01-23, the 12 new E3 questions, the 15-question
spelling hold-out, opened here for the first time) through the new front end and the gate, three ways:
    original    the question as typed
    spellfixer  today's SpellFixer output
    rewrite     the Gemini English rewrite (Level 1 decision)
Per question and method: gold chunk rank (where the file has gold chunks), best rerank score, gate grade and the
rule that decided it. For the hold-out: how many of its intended word fixes each method makes (word for word).
Rewrites already made in a Level 1b folder are reused (--reuse); only the rest are asked (dry run unless --allow-api-calls).

    python -m src.checks.manual_check --reuse check-l1b-20261017-2215                      dry run: counts
    python -m src.checks.manual_check --reuse check-l1b-20261017-2215 --allow-api-calls    run
"""
from __future__ import annotations

import argparse
import difflib
import re
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from src import pipeline
from src.checks.level1b_rewrite import COLUMNS, collect, saved
from src.checks.level1c_translate_search import gold_chunks, gold_rank
from src.checks.level5_gate import reasons
from src.config import load, path_of
from src.paths import ROOT
from src.run import RUNS

MODEL = "gemini:gemini-3.5-flash-lite"
FILES = {"manual": "data/eval/benchmark/manual_questions_with_answers.xlsx",
         "new_12": "data/eval/benchmark/new_questions_12.xlsx",
         "holdout": "data/eval/benchmark/spelling_holdout_15.xlsx"}


def manual_questions() -> list[dict]:
    gold = gold_chunks()
    rows = []
    for source, file in FILES.items():
        table = pipeline.read_table(path_of(file)).fillna("")
        for row in table.to_dict("records"):
            rows.append({"source": source, "qid": row["qid"], "label": row.get("label") or "answerable",
                         "question": str(row["question"]).strip(), "intended": str(row.get("intended", "")).strip(),
                         "written_by": row.get("written_by", "author"), "gold": gold.get(row["qid"], [])})
    return rows


def words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", str(text).lower())


def word_fixes(question: str, intended: str) -> list[tuple[str, str]]:
    """(typed word, intended word) for every one-word-for-one-word change between question and intended."""
    typed, wanted = words(question), words(intended)
    pairs = difflib.SequenceMatcher(a=typed, b=wanted, autojunk=False).get_opcodes()
    return [(typed[i], wanted[j]) for tag, i1, i2, j1, j2 in pairs if tag == "replace" and i2 - i1 == j2 - j1
            for i, j in zip(range(i1, i2), range(j1, j2))]


def method_rows(gate, item: dict, rewrite: str) -> list[dict]:
    texts = {"original": item["question"], "spellfixer": gate.SP.fix(item["question"])[0], "rewrite": rewrite}
    rows = []
    for method, text in texts.items():
        hits = gate.R.search(text, k=5)
        signals = gate.signals(text, hits)
        fixes = word_fixes(item["question"], item["intended"]) if item["intended"] else []
        rows.append({"source": item["source"], "written_by": item["written_by"], "qid": item["qid"], "label": item["label"],
                     "method": method, "query": text, "gold": ", ".join(item["gold"]),
                     "gold_rank": gold_rank(hits, item["gold"]) if item["gold"] else None,
                     "best": signals["best"], "grade": gate.grade(signals), "rule": reasons(signals)[0],
                     "fixes_wanted": len(fixes), "fixes_made": sum(wanted in words(text) for typed, wanted in fixes),
                     "top5": ", ".join(hit["chunk_id"] for hit in hits)})
    return rows


def summary(table: pd.DataFrame) -> pd.DataFrame:
    def one(group: pd.DataFrame) -> pd.Series:
        answerable, unanswerable = group[group.label == "answerable"], group[group.label != "answerable"]
        with_gold = answerable[answerable.gold != ""]
        return pd.Series({
            "answerable": len(answerable), "answerable passing gate (S+P)": int(answerable.grade.isin(["S", "P"]).sum()),
            **{f"answerable {grade}": int((answerable.grade == grade).sum()) for grade in "SPIM"},
            "gold in top 5": f"{int(with_gold.gold_rank.notna().sum())}/{len(with_gold)}",
            "unanswerable": len(unanswerable), "unanswerable passing gate (S+P)": int(unanswerable.grade.isin(["S", "P"]).sum()),
            "word fixes made": f"{int(group.fixes_made.sum())}/{int(group.fixes_wanted.sum())}"})
    return table.groupby(["source", "method"], sort=False).apply(one, include_groups=False).reset_index()


def main(argv: list[str] | None = None) -> pd.DataFrame | None:
    parser = argparse.ArgumentParser(description="Manual questions: spelling/translation and gate (dry run by default)")
    parser.add_argument("--reuse", help="a check-l1b-* folder whose Gemini rewrites are reused")
    parser.add_argument("--folder", help="an existing check-manual-* folder to continue")
    parser.add_argument("--allow-api-calls", action="store_true", help="really call the model")
    args = parser.parse_args(argv)
    config, items = load(), manual_questions()
    reused = saved(RUNS / args.reuse / "rewrites.csv").query("model == @MODEL and error == ''") if args.reuse else pd.DataFrame(columns=COLUMNS)
    reused = reused[reused.qid.isin([item["qid"] for item in items])]
    counts = ", ".join(f"{source} {sum(item['source'] == source for item in items)}" for source in FILES)
    print(f"{len(items)} questions ({counts}); {reused.qid.nunique()} rewrites reused, {len(items) - reused.qid.nunique()} to ask")
    if not args.allow_api_calls:
        print("No calls made (add --allow-api-calls).")
        return None
    load_dotenv(ROOT / ".env", override=True)
    folder = RUNS / (args.folder or f"check-manual-{datetime.now():%Y%m%d-%H%M}")
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / "rewrites.csv").exists():
        reused[COLUMNS].to_csv(folder / "rewrites.csv", index=False)
    collect([MODEL], items, folder / "rewrites.csv", config["llm"]["pause"])
    replies = saved(folder / "rewrites.csv").query("model == @MODEL and error == ''").drop_duplicates("qid", keep="last")
    rewrites = dict(zip(replies.qid, replies.rewrite))
    if any(not rewrites.get(item["qid"]) for item in items):
        print(f"some questions have no rewrite yet - run again with --folder {folder.name}")
        return None
    gate = pipeline.open_gate(config)
    table = pd.DataFrame([row for item in items for row in method_rows(gate, item, rewrites[item["qid"]])])
    result = summary(table)
    with pd.ExcelWriter(folder / "manual.xlsx") as writer:
        result.to_excel(writer, sheet_name="summary", index=False)
        table.to_excel(writer, sheet_name="all", index=False)
    print(result.to_string(index=False))
    print(f"-> {folder / 'manual.xlsx'}")
    return result


if __name__ == "__main__":
    main()