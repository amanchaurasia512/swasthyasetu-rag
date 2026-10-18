"""Level 1c check - translate before search (Flipkart's method, arXiv 2208.03713) against today's SpellFixer.
For every question, four ways to search, same retriever and gate, no answers and no API calls (the rewrites are
read from a Level 1b folder):
    original      the question as typed
    spellfixer    today's SpellFixer output
    rewrite       the LLM's English rewrite alone
    dual          original + rewrite searched together, pooled and reranked against the rewrite
Measured: is a gold chunk in the top 5 (and at which rank), and the gate grade S/P/I/M of that search.

    python -m src.checks.level1c_translate_search check-l1b-20261017-2215
    python -m src.checks.level1c_translate_search check-l1b-20261017-2215 --model llama3.2:3b
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime

import pandas as pd

from src import pipeline
from src.checks.level1_spelling import all_questions
from src.config import load, path_of
from src.run import RUNS

E3 = "config/experiments/e3.yaml"
TOP_K, POOL = 5, 10
CHUNK_ID = re.compile(r"^[A-Z]+\d+(?:-\d+)*(?:-\*)?$")


def gold_chunks() -> dict[str, list[str]]:
    """{qid: gold chunk ids} from every question file that has a gold_chunks column."""
    sets = {**load()["questions"], **load(E3)["questions"]}
    tables = [pipeline.read_table(path_of(spec["file"])) for spec in sets.values()
              if isinstance(spec, dict) and "file" in spec]
    return {row.qid: [token for token in re.split(r"[;,\s]+", str(row.gold_chunks)) if CHUNK_ID.match(token)]
            for table in tables if "gold_chunks" in table for row in table.itertuples()}


def gold_rank(hits: list[dict], gold: list[str]) -> int | None:
    """Rank (1-5) of the first hit that is a gold chunk; 'T101-1-*' matches every chunk of that table."""
    matches = [rank for rank, hit in enumerate(hits, 1)
               if any(hit["chunk_id"] == chunk or (chunk.endswith("*") and hit["chunk_id"].startswith(chunk[:-1]))
                      for chunk in gold)]
    return matches[0] if matches else None


def dual_search(gate, original: str, rewrite: str) -> list[dict]:
    pool = list({hit["chunk_id"]: hit for query in (original, rewrite) for hit in gate.R.search(query, k=POOL)}.values())
    scores = gate.R.reranker.rerank(rewrite, [gate.R.chunks[gate.R.row_of[hit["chunk_id"]]]["embed_text"] for hit in pool])
    return sorted(({**hit, "score": float(score)} for hit, score in zip(pool, scores)), key=lambda hit: -hit["score"])[:TOP_K]


def searches(gate, question: str, rewrite: str) -> dict[str, tuple[str, list[dict]]]:
    fixed = gate.SP.fix(question)[0]
    return {"original": (question, gate.R.search(question, k=TOP_K)),
            "spellfixer": (fixed, gate.R.search(fixed, k=TOP_K)),
            "rewrite": (rewrite, gate.R.search(rewrite, k=TOP_K)),
            "dual": (rewrite, dual_search(gate, question, rewrite))}


def question_rows(gate, question: dict, rewrite: str, gold: list[str]) -> list[dict]:
    return [{"qid": question["qid"], "label": question["label"], "method": method, "query": query,
             "gold": ", ".join(gold), "gold_rank": gold_rank(hits, gold) if gold else None,
             "grade": gate.grade(gate.signals(query, hits)), "best": round(hits[0]["score"], 2),
             "top5": ", ".join(hit["chunk_id"] for hit in hits)}
            for method, (query, hits) in searches(gate, question["question"], rewrite).items()]


def summary(table: pd.DataFrame) -> pd.DataFrame:
    def one(group: pd.DataFrame) -> pd.Series:
        answerable, unanswerable = group[group.label == "answerable"], group[group.label != "answerable"]
        with_gold = answerable[answerable.gold != ""]
        return pd.Series({
            "answerable with gold": len(with_gold),
            "gold in top 5 %": round(100 * with_gold.gold_rank.notna().mean(), 1),
            "gold at rank 1 %": round(100 * (with_gold.gold_rank == 1).mean(), 1),
            **{f"answerable {grade}": int((answerable.grade == grade).sum()) for grade in "SPIM"},
            **{f"unanswerable {grade}": int((unanswerable.grade == grade).sum()) for grade in "SPIM"}})
    return table.groupby("method", sort=False).apply(one, include_groups=False).reset_index()


def changed(table: pd.DataFrame) -> pd.DataFrame:
    """One row per question whose gold rank or grade differs between methods, methods side by side."""
    wide = table.pivot(index=["qid", "label", "gold"], columns="method", values=["gold_rank", "grade", "query"])
    wide.columns = [f"{field}_{method}" for field, method in wide.columns]
    differs = wide.filter(like="gold_rank_").astype(str).nunique(axis=1).gt(1) | wide.filter(like="grade_").nunique(axis=1).gt(1)
    return wide[differs].reset_index()


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="Level 1c: translate before search vs SpellFixer (no API calls)")
    parser.add_argument("folder", help="the check-l1b-* folder holding rewrites.csv")
    parser.add_argument("--model", default="gemini:gemini-3.5-flash-lite", help="whose rewrites to use")
    args = parser.parse_args(argv)
    replies = pd.read_csv(RUNS / args.folder / "rewrites.csv", dtype=str).fillna("")
    replies = replies[(replies.model == args.model) & (replies.error == "")].drop_duplicates("qid", keep="last")
    rewrites, gold = dict(zip(replies.qid, replies.rewrite)), gold_chunks()
    questions = [question for question in all_questions() if rewrites.get(question["qid"])]
    print(f"{len(questions)} questions with a {args.model} rewrite")
    gate = pipeline.open_gate(load())
    table = pd.DataFrame([row for question in questions
                          for row in question_rows(gate, question, rewrites[question["qid"]], gold.get(question["qid"], []))])
    folder = RUNS / f"check-l1c-{datetime.now():%Y%m%d-%H%M}"
    folder.mkdir(parents=True, exist_ok=True)
    result = summary(table)
    with pd.ExcelWriter(folder / "compare.xlsx") as writer:
        result.to_excel(writer, sheet_name="summary", index=False)
        changed(table).to_excel(writer, sheet_name="changed", index=False)
        table.to_excel(writer, sheet_name="all", index=False)
    print(result.to_string(index=False))
    print(f"-> {folder / 'compare.xlsx'}")
    return result


if __name__ == "__main__":
    main()