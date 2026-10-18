"""Level 5b check - can Gemini do the M rule's job? Every query the gate graded M (a hard word or year not in the PDF)
is sent to Gemini WITH its evidence instead of getting the fixed "does not cover" sentence: same prompt as E3
(v3b_med + f0p), no warning note, the Gemini rewrite as the question. Answers are saved for reading
(dry run unless --allow-api-calls; every answer saved at once, continue with --folder).

    python -m src.checks.level5b_m_answers check-l5-20261018-1000 check-manual-20261018-1031                      dry run
    python -m src.checks.level5b_m_answers check-l5-20261018-1000 check-manual-20261018-1031 --allow-api-calls
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from src import answer, evaluate, pipeline, prompts
from src.config import load
from src.llm import LLMError, chat_full, provider_of
from src.paths import ROOT
from src.run import RUNS

MODEL, VERSION, FORMAT, RUNS_PER_QUESTION = "gemini:gemini-3.5-flash-lite", "v3b_med", "f0p", 2
COLUMNS = ["qid", "run", "answer", "finish_reason", "error"]


def m_queries(gate_folder: str, manual_folder: str | None) -> pd.DataFrame:
    """Every M query of the Level 5 check, plus the manual-check questions still graded M after the rewrite."""
    gate = pd.read_excel(RUNS / gate_folder / "gate.xlsx", sheet_name="hard_words")[["qid", "label", "rewrite", "hard_out", "best"]]
    if not manual_folder:
        return gate
    manual = pd.read_excel(RUNS / manual_folder / "manual.xlsx", sheet_name="all")
    manual = manual[(manual.method == "rewrite") & (manual.grade == "M")]
    manual = manual.assign(rewrite=manual["query"], hard_out=manual.rule.str.extract(r"\[(.*)\]", expand=False))
    return pd.concat([gate, manual[["qid", "label", "rewrite", "hard_out", "best"]]]).drop_duplicates("qid").reset_index(drop=True)


def m_prompt(gate, question: str) -> tuple[str, str]:
    """The prompt the question would get as an ordinary 'answer' (grade S): context added, refined, no notes."""
    hits = answer.add_context(gate.R.search(question, k=5), gate.R)
    hits = gate.refine(hits, question) if gate.refine_on else hits
    return prompts.build({"fixed": question, "hits": hits, "final": "S", "signals": {}}, VERSION, FORMAT)


def ask(prompt: tuple[str, str], qid: str, run: int) -> dict:
    try:
        reply = chat_full(MODEL, *prompt)
        return {"qid": qid, "run": run, "answer": reply["text"], "finish_reason": reply["finish_reason"], "error": ""}
    except LLMError as failure:
        return {"qid": qid, "run": run, "answer": "", "finish_reason": "", "error": str(failure)[:300]}


def main(argv: list[str] | None = None) -> pd.DataFrame | None:
    parser = argparse.ArgumentParser(description="Level 5b: M queries answered by Gemini with evidence (dry run by default)")
    parser.add_argument("gate_folder", help="check-l5-* folder (gate.xlsx)")
    parser.add_argument("manual_folder", nargs="?", help="check-manual-* folder (manual.xlsx), optional")
    parser.add_argument("--folder", help="an existing check-m-* folder to continue")
    parser.add_argument("--allow-api-calls", action="store_true", help="really call the model")
    args = parser.parse_args(argv)
    config, queries = load(), m_queries(args.gate_folder, args.manual_folder)
    print(f"{len(queries)} M queries ({(queries.label == 'answerable').sum()} answerable) x {RUNS_PER_QUESTION} runs = "
          f"{len(queries) * RUNS_PER_QUESTION} calls")
    if not args.allow_api_calls:
        print("No calls made (add --allow-api-calls).")
        return None
    load_dotenv(ROOT / ".env", override=True)
    folder = RUNS / (args.folder or f"check-m-{datetime.now():%Y%m%d-%H%M}")
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / "answers.csv"
    done = set() if not file.exists() else {(row.qid, row.run) for row in pd.read_csv(file).fillna("").itertuples() if not row.error}
    gate, pause = pipeline.open_gate(config), config["llm"]["pause"][provider_of(MODEL)[0]]
    for row in queries.itertuples():
        prompt = m_prompt(gate, row.rewrite)
        for run in range(1, RUNS_PER_QUESTION + 1):
            if (row.qid, run) in done:
                continue
            pd.DataFrame([ask(prompt, row.qid, run)], columns=COLUMNS).to_csv(file, mode="a", header=not file.exists(), index=False)
            time.sleep(pause)
    answers = pd.read_csv(file).fillna("").query("error == ''").drop_duplicates(["qid", "run"], keep="last")
    table = queries.merge(answers, on="qid").assign(refused_by_rule=lambda frame: frame.answer.map(evaluate.refused), verdict="")
    table.to_excel(folder / "answers.xlsx", index=False)
    print(table.groupby("label").refused_by_rule.agg(["sum", "count"]).rename(columns={"sum": "refused (phrase rule)", "count": "answers"}))
    print(f"-> {folder / 'answers.xlsx'}  (read every answer; refused_by_rule is only a hint)")
    return table


if __name__ == "__main__":
    main()