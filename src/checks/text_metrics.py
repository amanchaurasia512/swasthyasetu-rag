"""Similarity to the expected answer (C-037): ROUGE-L, BLEU and token F1 for every answer of a run, per model.
These are the usual text-overlap metrics of summarisation, translation and QA papers. They compare the answer with
the answer key, NOT with the sources, so they measure closeness to the expected answer, not hallucination (that is
the over-answer rate and, later, a faithfulness check). No models, no API calls.

Only answerable answers are scored (llm_label = answerable: the expected facts were in the prompt). Citation marks
([1], [1, 3]) are removed first, because the key has none. A run with several models is compared on the answers
every model has (same question and run), so a model that stopped early is not judged on an easier subset.

    python -m src.checks.text_metrics e6-20261019-1557       writes text_metrics.csv and text_metrics_summary.csv
"""
from __future__ import annotations

import argparse
import re
from collections import Counter

import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer

from src.evaluate import CITE
from src.run import RUNS

ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def plain(text: str) -> str:
    """The answer without citation marks and extra spaces."""
    return re.sub(r"\s+([.,;:!?])", r"\1", " ".join(CITE.sub("", str(text)).split()))


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def token_f1(answer: str, expected: str) -> float:
    """Word-overlap F1 as in SQuAD: shared words counted with their multiplicity."""
    shared = sum((Counter(words(answer)) & Counter(words(expected))).values())
    if not shared:
        return 0.0
    precision, recall = shared / len(words(answer)), shared / len(words(expected))
    return 2 * precision * recall / (precision + recall)


def measures(answer: str, expected: str) -> dict:
    text = plain(answer)
    return {"rouge_l": ROUGE.score(expected, text)["rougeL"].fmeasure if text else 0.0,
            "bleu": sacrebleu.sentence_bleu(text, [expected]).score / 100 if text else 0.0,
            "token_f1": token_f1(text, expected), "empty": not text}


def common_answers(scored: pd.DataFrame) -> pd.DataFrame:
    """The answers (question + run) that every model in the run has."""
    keys = [set(zip(group.qid, group.run)) for model, group in scored.groupby("model")]
    shared = set.intersection(*keys) if keys else set()
    return scored[[(qid, run) in shared for qid, run in zip(scored.qid, scored.run)]]


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="ROUGE-L, BLEU and token F1 against the expected answers")
    parser.add_argument("run_id")
    args = parser.parse_args(argv)
    folder = RUNS / args.run_id
    scored = pd.read_csv(folder / "scored.csv", keep_default_na=False)
    rows = common_answers(scored[scored.llm_label == "answerable"])
    table = pd.concat([rows[["model", "version", "run", "qid"]].reset_index(drop=True),
                       pd.DataFrame([measures(answer, expected) for answer, expected in zip(rows.answer, rows.expected)])],
                      axis=1)
    summary = table.groupby(["model", "version"]).agg(answers=("qid", "size"), rouge_l=("rouge_l", "mean"),
                                                     bleu=("bleu", "mean"), token_f1=("token_f1", "mean"),
                                                     empty=("empty", "sum")).round(3)
    table.to_csv(folder / "text_metrics.csv", index=False)
    summary.to_csv(folder / "text_metrics_summary.csv")
    print(summary.sort_values("rouge_l", ascending=False).to_string(), f"\n-> {folder / 'text_metrics_summary.csv'}")
    return summary


if __name__ == "__main__":
    main()