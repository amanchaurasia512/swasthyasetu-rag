"""Search-only test on two documents (C-042): each question set on its own document's index, and on the other one.
No answers, no LLM call - only the front end, search, rerank and the gate, with the chatbot's settings (config/app.yaml).

    NHM manual questions  -> NHM index      gold chunk in the top 5? gate S/P? (a check that C-040/C-041 changed nothing)
    NFHS-6 questions      -> NFHS-6 index   right row and right area in the top 5? unanswerable -> I?
    NHM questions         -> NFHS-6 index   every question should be refused by the gate (I)
    NFHS-6 questions      -> NHM index      every question should be refused by the gate (I)

The right NFHS row comes from the indicator table (area + indicator number -> chunk N<page>-<indicator>), not typed
by hand. Questions with a word that is neither English nor in the document go to the live Gemini rewrite only with
--allow-api-calls (rule R1); the dry run uses the SpellFixer for them and says how many would be rewritten. A rewrite
is asked once per question and used for both indexes.

    python -m src.checks.search_two_docs                      dry run, no API calls
    python -m src.checks.search_two_docs --allow-api-calls    with the rewrite (about 6 calls)
writes runs/check-search-<stamp>/results.csv and summary.csv
"""
from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from src import pipeline, rewrite
from src.checks.level1c_translate_search import gold_rank
from src.config import load, path_of
from src.run import RUNS

APP = "config/app.yaml"
SETS = {"nhm": ("nhm_hrh_2022", "data/eval/benchmark/manual_questions_with_answers.xlsx"),
        "nfhs6": ("nfhs6", "data/eval/nfhs6/nfhs6_questions_claude20.csv")}


def nhm_questions(file: str) -> list[dict]:
    """The 23 hand-written NHM questions with their gold chunks (all answerable)."""
    return [{"set": "nhm", "qid": row.qid, "question": row.question, "label": "answerable", "kind": row.qtype,
             "language": "en", "gold": [chunk.strip() for chunk in str(row.gold_chunks).replace(",", ";").split(";")
                                        if chunk.strip() and pd.notna(row.gold_chunks)],
             "areas": []} for row in pipeline.read_table(path_of(file)).itertuples()]


def nfhs_questions(file: str, indicators: pd.DataFrame) -> list[dict]:
    """The NFHS-6 questions; the right chunk of each area + indicator is looked up in the indicator table."""
    pages = {(row.area, str(row.indicator)): row.pdf_page for row in indicators.itertuples()}
    records = []
    for row in pipeline.read_table(path_of(file)).fillna("").itertuples():
        areas = [area for area in str(row.area).split(";") if area]
        number = int(float(row.indicator)) if areas else 0          # read_table may give 18.0 for 18
        gold = [f"N{pages[(area, str(number))]}-{number:03d}" for area in areas]
        records.append({"set": "nfhs6", "qid": row.qid, "question": row.question, "label": row.label, "kind": row.kind,
                        "language": row.language, "gold": gold, "areas": areas})
    return records


def search_one(gate, question: dict, index: str, asked: dict | None) -> dict:
    """One question on one index: grade, best score, gold rank, right area in the top 5."""
    prepared = gate.prepare(question["question"], asked["text"] if asked and asked["text"] else None)
    hit_ids = prepared["hit_ids"].split(";") if prepared["hit_ids"] else []
    hits = [{"chunk_id": chunk_id} for chunk_id in hit_ids]
    areas = {gate.R.chunks[gate.R.row_of[chunk_id]]["section_id"] for chunk_id in hit_ids}
    own = question["set"] == index
    return {"questions": question["set"], "index": index, "qid": question["qid"], "label": question["label"],
            "kind": question["kind"], "language": question["language"], "question": question["question"],
            "searched": prepared["fixed"], "rewritten": bool(asked and asked["text"]), "grade": prepared["final"],
            "best": prepared["best"], "gold_rank": gold_rank(hits, question["gold"]) if own and question["gold"] else None,
            "all_gold_top5": all(chunk in hit_ids for chunk in question["gold"]) if own and question["areas"] else None,
            "area_top5": all(area in areas for area in question["areas"]) if own and question["areas"] else None,
            "expected": ("found" if question["label"] == "answerable" else "refused") if own else "refused",
            "top5": ";".join(hit_ids)}


def summary(results: pd.DataFrame) -> pd.DataFrame:
    """Per question set and index: gold in top 5, right area, gate grades, and refusals where a refusal is expected."""
    rows = []
    for (questions, index), group in results.groupby(["questions", "index"], sort=False):
        found, refused = group[group.expected == "found"], group[group.expected == "refused"]
        rows.append({"questions": questions, "index": index, "n": len(group), "rewritten": int(group.rewritten.sum()),
                     "should_find": len(found), "gold_top5": int(found.gold_rank.notna().sum()),
                     "all_gold_top5": int(found.all_gold_top5.eq(True).sum()) if found.all_gold_top5.notna().any() else None,
                     "area_top5": int(found.area_top5.eq(True).sum()) if found.area_top5.notna().any() else None,
                     "found_S_or_P": int(found.grade.isin(["S", "P"]).sum()),
                     "should_refuse": len(refused), "refused_I": int((refused.grade == "I").sum()),
                     "grades": dict(group.grade.value_counts())})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="Search-only test of NHM and NFHS-6 questions on both indexes")
    parser.add_argument("--allow-api-calls", action="store_true", help="send unknown-word questions to the live rewrite")
    args = parser.parse_args(argv)
    load_dotenv()
    configs = {name: load(APP, doc=doc) for name, (doc, file) in SETS.items()}
    gates = {name: pipeline.open_gate(config) for name, config in configs.items()}
    indicators = pd.read_csv(path_of(configs["nfhs6"]["files"]["indicators"]), dtype={"indicator": str})
    questions = nhm_questions(SETS["nhm"][1]) + nfhs_questions(SETS["nfhs6"][1], indicators)
    front_end = configs["nhm"]["query"]["front_end"]
    rows, calls, would = [], 0, 0
    for question in questions:
        needs = any(rewrite.needed(question["question"], front_end, gate.SP) for gate in gates.values())
        would += needs
        asked = rewrite.ask(question["question"], configs["nhm"]) if needs and args.allow_api_calls else None
        calls += asked is not None
        rows += [search_one(gates[index], question, index, asked) for index in gates]
    results = pd.DataFrame(rows)
    folder = RUNS / f"check-search-{datetime.now():%Y%m%d-%H%M}"
    folder.mkdir(parents=True)
    table = summary(results)
    results.to_csv(folder / "results.csv", index=False, encoding="utf-8-sig")
    table.to_csv(folder / "summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 60)
    print(f"{len(questions)} questions | need a rewrite: {would} | rewrite calls made: {calls}"
          f"{'' if args.allow_api_calls else ' (dry run: SpellFixer used instead)'}\n")
    print(table.to_string(index=False))
    misses = results[((results.expected == "found") & results.gold_rank.isna())
                     | ((results.expected == "refused") & (results.grade != "I"))]
    print("\nNot as expected:")
    print(misses[["questions", "index", "qid", "label", "grade", "best", "gold_rank", "searched"]].to_string(index=False))
    print(f"\n-> {folder}")
    return table


if __name__ == "__main__":
    main()