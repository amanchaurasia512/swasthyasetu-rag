"""NFHS-6 questions in the runner's format (C-045): the expected answer of each question is built from the indicator
table, never typed by hand. One fact per value the question asks for: the area, the indicator's words and the value,
e.g. "Bihar Total fertility rate 2.7". The scorer (src.evaluate) then needs that exact number and half of the words.
Numbers inside indicator names (ages, "BMI <18.5") are left out of the words, so the only numbers to find are the values.

    python -m src.checks.nfhs_questions        data/eval/nfhs6/nfhs6_questions_claude20.csv -> questions_e7.csv
"""
from __future__ import annotations

import re

import pandas as pd

from src.config import load, path_of

SOURCE = "data/eval/nfhs6/nfhs6_questions_claude20.csv"
TARGET = "data/eval/nfhs6/questions_e7.csv"


def name_words(name: str) -> str:
    """The indicator's words without numbers and brackets: 'Women age 20-24 years married before age 18 years (%)'
    -> 'Women age years married before age years'."""
    return " ".join(re.sub(r"\([^)]*\)|[\d.,<>≥%+-]+", " ", name).split())


def expected_answer(row, values: dict) -> str:
    """One fact per area and asked column, separated by ';' (src.evaluate.facts splits on it)."""
    if row.label != "answerable":
        return ""
    number = int(float(row.indicator))
    facts = [f"{area} {name_words(values[(area, number)]['name'])} {values[(area, number)][column]}"
             for area in str(row.area).split(";") for column in str(row.column).split(";")]
    return "; ".join(facts)


def build(source: str = SOURCE, indicators: pd.DataFrame | None = None) -> pd.DataFrame:
    table = indicators if indicators is not None else pd.read_csv(path_of(load(doc="nfhs6")["files"]["indicators"]), dtype=str)
    values = {(row["area"], int(row["indicator"])): row for row in table.to_dict("records")}
    questions = pd.read_csv(path_of(source), dtype=str).fillna("")
    return questions.assign(expected_answer=[expected_answer(row, values) for row in questions.itertuples()])


if __name__ == "__main__":
    built = build()
    built.to_csv(path_of(TARGET), index=False, encoding="utf-8-sig")
    print(built[["qid", "label", "expected_answer"]].to_string(index=False), f"\n-> {TARGET}")