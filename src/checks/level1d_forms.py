"""Level 1d check - the 4-form translation test (one question written as A1 English, A2 English with typos,
B1 Hinglish, B2 Hinglish with typos; same gold chunk for all four). Every query is rewritten by the LLM
(dry run unless --allow-api-calls; replies saved at once, continue with --folder), then searched four ways
(original / spellfixer / rewrite / dual, as in Level 1c). Result per form x method: gold in top 5, gold at rank 1,
gate grades S/P/I/M.

    python -m src.checks.level1d_forms                                     dry run: counts
    python -m src.checks.level1d_forms --allow-api-calls                   rewrite + search, new folder
    python -m src.checks.level1d_forms --allow-api-calls --folder check-forms-20261018-0900     continue
"""
from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from src import pipeline
from src.checks.level1b_rewrite import collect, saved
from src.checks.level1c_translate_search import question_rows
from src.config import load, path_of
from src.paths import ROOT
from src.run import RUNS

FILE = "data/eval/benchmark/translation_test_23x4.xlsx"
FORMS = ["A1_english", "A2_english_typos", "B1_hinglish", "B2_hinglish_typos"]
MODEL = "gemini:gemini-3.5-flash-lite"


def queries(file: str) -> list[dict]:
    """One query per question and form: qid 'HG09:B1_hinglish'."""
    table = pipeline.read_table(path_of(file)).fillna("")
    return [{"qid": f"{row['qid']}:{form}", "base": row["qid"], "form": form, "label": row["label"],
             "question": str(row[form]).strip(), "gold": [chunk.strip() for chunk in str(row["gold_chunk"]).split(",") if chunk.strip()]}
            for row in table.to_dict("records") for form in FORMS]


def summary(table: pd.DataFrame) -> pd.DataFrame:
    def one(group: pd.DataFrame) -> pd.Series:
        answerable, unanswerable = group[group.label == "answerable"], group[group.label != "answerable"]
        return pd.Series({
            "gold in top 5 %": round(100 * answerable.gold_rank.notna().mean(), 1),
            "gold at rank 1 %": round(100 * (answerable.gold_rank == 1).mean(), 1),
            "answerable passing gate (S+P)": int(answerable.grade.isin(["S", "P"]).sum()),
            **{f"answerable {grade}": int((answerable.grade == grade).sum()) for grade in "SPIM"},
            "unanswerable passing gate (S+P)": int(unanswerable.grade.isin(["S", "P"]).sum())})
    return table.groupby(["form", "method"], sort=False).apply(one, include_groups=False).reset_index()


def main(argv: list[str] | None = None) -> pd.DataFrame | None:
    parser = argparse.ArgumentParser(description="Level 1d: 4-form translation test (dry run by default)")
    parser.add_argument("--file", default=FILE)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--folder", help="an existing check-forms-* folder to continue")
    parser.add_argument("--allow-api-calls", action="store_true", help="really call the model")
    args = parser.parse_args(argv)
    config, items = load(), queries(args.file)
    print(f"{len(items)} queries ({len(items) // len(FORMS)} questions x {len(FORMS)} forms), model {args.model}")
    if not args.allow_api_calls:
        print("No calls made (add --allow-api-calls).")
        return None
    load_dotenv(ROOT / ".env", override=True)
    folder = RUNS / (args.folder or f"check-forms-{datetime.now():%Y%m%d-%H%M}")
    folder.mkdir(parents=True, exist_ok=True)
    collect([args.model], items, folder / "rewrites.csv", config["llm"]["pause"])
    replies = saved(folder / "rewrites.csv").query("model == @args.model and error == ''").drop_duplicates("qid", keep="last")
    rewrites = dict(zip(replies.qid, replies.rewrite))
    missing = [item["qid"] for item in items if not rewrites.get(item["qid"])]
    if missing:
        print(f"{len(missing)} queries have no rewrite yet (e.g. {missing[:3]}) - run again with --folder {folder.name}")
        return None
    gate = pipeline.open_gate(config)
    table = pd.DataFrame([{"base": item["base"], "form": item["form"], **row} for item in items
                          for row in question_rows(gate, item, rewrites[item["qid"]], item["gold"])])
    result = summary(table)
    with pd.ExcelWriter(folder / "compare.xlsx") as writer:
        result.to_excel(writer, sheet_name="summary", index=False)
        table.to_excel(writer, sheet_name="all", index=False)
    print(result.to_string(index=False))
    print(f"-> {folder / 'compare.xlsx'}")
    return result


if __name__ == "__main__":
    main()