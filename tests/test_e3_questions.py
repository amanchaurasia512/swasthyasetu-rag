"""E3 question set (Phase 5, C-016): the full set with 10 questions swapped for 12 new ones (new_questions_12.xlsx).
The 10 removed are not hand-picked: the rule in config/experiments/e3.yaml is recomputed here.
Run from the project root:   python -m pytest tests/test_e3_questions.py -q      (no models, no API calls; seconds)"""
import json

import pandas as pd
import pytest

from src import pipeline, run
from src.config import load, path_of

E3 = load("config/experiments/e3.yaml")
NEW_TYPES = {"general": 4, "spelling": 3, "table": 1, "letter": 1, "ocr": 1, "irrelevant": 2}


@pytest.fixture(scope="module")
def questions():
    return {question["qid"]: question for question in pipeline.load_questions(E3)}


def gold_chunks(qid, gold):
    return {chunk.strip() for chunk in str(gold.get(qid, "")).replace(",", ";").split(";") if chunk.strip() not in ("", "nan")}


def test_removal_rule_gives_the_listed_ten():
    """Answerable, reached the LLM in E2's frozen evidence, a gold chunk already in its hits; 10 drawn, random_state=7."""
    sets = E3["questions"]
    labels = {question["qid"]: question["label"] for question in pipeline.load_questions(load())}
    with open(path_of(sets["frozen_evidence"]), encoding="utf-8") as file:
        frozen = [json.loads(line) for line in file]
    manual, set_c = pd.read_excel(path_of(sets["manual"]["file"])), pd.read_csv(path_of(sets["set_C"]["file"]))
    gold = {**dict(zip(manual.qid, manual.gold_chunks)), **dict(zip(set_c.qid, set_c.gold_chunks))}
    candidates = sorted(record["qid"] for record in frozen
                        if labels[record["qid"]] == "answerable" and record["action"] in pipeline.TO_LLM
                        and gold_chunks(record["qid"], gold) & {hit["chunk_id"] for hit in record["hits"]})
    assert len(candidates) == 71
    assert pd.Series(candidates).sample(10, random_state=7).tolist() == sets["removed"]


def test_set_is_173_minus_10_plus_12(questions):
    assert len(questions) == 175
    assert not set(E3["questions"]["removed"]) & set(questions)


def test_new_questions(questions):
    new = pd.read_excel(path_of(E3["questions"]["new"]["file"]))
    assert new.qtype.value_counts().to_dict() == NEW_TYPES
    assert all(questions[row.qid]["label"] == row.label and questions[row.qid]["source"] == "new" for row in new.itertuples())
    assert all(questions[row.qid]["expected"] == "" for row in new.itertuples() if row.label == "unanswerable")


def test_new_file_is_fingerprinted():
    assert "new" in run.fingerprints(E3)


def test_e2_set_unchanged():
    assert len(pipeline.load_questions(load())) == 173