"""Level 1b check - what an LLM does with a question when asked only to clean it for search (query rewriting).
Every question goes to each model with one fixed instruction; the rewrite is saved next to the original and the
current SpellFixer output, for reading. Nothing is searched or answered yet. Dry run unless --allow-api-calls
(the same safety switch as the runner, rule R1); every reply is saved at once, so a stopped run continues
where it left off with --folder.

    python -m src.checks.level1b_rewrite                                    dry run: counts + the exact prompt
    python -m src.checks.level1b_rewrite --allow-api-calls                  new folder runs/check-l1b-<stamp>/
    python -m src.checks.level1b_rewrite --allow-api-calls --folder check-l1b-20261017-2215    continue
    python -m src.checks.level1b_rewrite --models qwen2.5:7b --allow-api-calls                one model only
"""
from __future__ import annotations

import argparse
import re
import time
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from src.checks.level1_spelling import all_questions
from src.config import load, path_of
from src.llm import LLMError, chat_full, provider_of
from src.paths import ROOT
from src.query import SpellFixer
from src.retriever import load_chunks
from src.run import RUNS

MODELS = ["gemini:gemini-3.5-flash-lite", "qwen2.5:7b"]
SYSTEM = ("You rewrite a user's question so it can be used to search a document. Fix spelling mistakes. "
          "Translate any Hindi or Hinglish words into English. Keep every name, number, date, abbreviation and place "
          "exactly as written. Do not answer the question. Do not add, remove or guess any information. "
          "Reply with the rewritten question only, on one line.")
COLUMNS = ["model", "qid", "rewrite", "prompt_tokens", "answer_tokens", "finish_reason", "error"]


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+", str(text)))


def saved(file) -> pd.DataFrame:
    return pd.read_csv(file, dtype=str).fillna("") if file.exists() else pd.DataFrame(columns=COLUMNS)


def ask(model: str, question: dict) -> dict:
    try:
        reply = chat_full(model, SYSTEM, question["question"])
        return {"model": model, "qid": question["qid"], "rewrite": reply["text"].strip(), "error": "",
                **{key: reply[key] for key in ("prompt_tokens", "answer_tokens", "finish_reason")}}
    except LLMError as failure:
        return {"model": model, "qid": question["qid"], "rewrite": "", "error": str(failure)[:300]}


def collect(models: list[str], questions: list[dict], file, pauses: dict) -> None:
    for model in models:
        done = set(saved(file).query("model == @model and error == ''").qid)
        todo = [question for question in questions if question["qid"] not in done]
        print(f"{model}: {len(todo)} to ask")
        for count, question in enumerate(todo, 1):
            row = ask(model, question)
            pd.DataFrame([row], columns=COLUMNS).to_csv(file, mode="a", header=not file.exists(), index=False)
            if row["error"]:
                print(f"  {question['qid']}: {row['error'][:120]}")
            if count % 20 == 0:
                print(f"  {count}/{len(todo)}")
            time.sleep(pauses.get(provider_of(model)[0], 0))


def compare(questions: list[dict], file, fixer: SpellFixer) -> pd.DataFrame:
    replies = saved(file).query("error == ''").drop_duplicates(["model", "qid"], keep="last")
    table = pd.DataFrame([{"qid": question["qid"], "label": question["label"], "question": question["question"],
                           "spellfixer": fixer.fix(question["question"])[0]} for question in questions])
    for model, group in replies.groupby("model"):
        short = provider_of(model)[1]
        table[f"rewrite_{short}"] = table.qid.map(dict(zip(group.qid, group.rewrite))).fillna("")
        table[f"numbers_kept_{short}"] = [numbers(question) <= numbers(rewrite) for question, rewrite
                                          in zip(table.question, table[f"rewrite_{short}"])]
        table[f"verdict_{short}"] = ""
    return table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Level 1b: LLM query rewriting, side by side (dry run by default)")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--folder", help="an existing check-l1b-* folder to continue")
    parser.add_argument("--allow-api-calls", action="store_true", help="really call the models")
    args = parser.parse_args(argv)
    config, questions = load(), all_questions()
    print(f"{len(questions)} questions x {len(args.models)} models = {len(questions) * len(args.models)} calls")
    if not args.allow_api_calls:
        print(f"\nSYSTEM:\n{SYSTEM}\n\nUSER (example):\n{questions[0]['question']}\n\nNo calls made (add --allow-api-calls).")
        return
    load_dotenv(ROOT / ".env", override=True)
    folder = RUNS / (args.folder or f"check-l1b-{datetime.now():%Y%m%d-%H%M}")
    folder.mkdir(parents=True, exist_ok=True)
    collect(args.models, questions, folder / "rewrites.csv", config["llm"]["pause"])
    table = compare(questions, folder / "rewrites.csv", SpellFixer(load_chunks(path_of(config["files"]["chunks"]))))
    table.to_excel(folder / "compare.xlsx", index=False)
    print(f"-> {folder / 'compare.xlsx'}")


if __name__ == "__main__":
    main()