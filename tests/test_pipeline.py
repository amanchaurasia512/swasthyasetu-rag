"""Pipeline regression (Phase 2, step 5a): stages 1-6 from src/pipeline.py, driven by the config, must give E2's
questions and labels, the frozen gate decisions, and every saved E2 prompt character for character.
Run from the project root:   python -m pytest tests/test_pipeline.py -q      (local models, no API calls; ~6 min)"""
import json

import pandas as pd
import pytest

from src import pipeline
from src.config import load, path_of

CONFIG = load()
BENCHMARK = path_of("data/eval/benchmark")
HIT_FIELDS = {"rank", "score", "chunk_id", "unit", "doc_type", "path", "pdf_pages", "printed_pages", "text"}


@pytest.fixture(scope="module")
def frozen():
    with open(path_of(CONFIG["questions"]["frozen_evidence"]), encoding="utf-8") as file:
        return [json.loads(line) for line in file]


@pytest.fixture(scope="module")
def saved():
    return pd.read_json(BENCHMARK / "e2_scoring_input.jsonl", lines=True, dtype=False)


@pytest.fixture(scope="module")
def questions():
    return pipeline.load_questions(CONFIG)


@pytest.fixture(scope="module")
def records(prepared):
    """pipeline.prepare() on every question (shared with test_retrieval.py, see conftest.py)."""
    return prepared


def test_same_173_questions_in_the_same_order(questions, frozen):
    pick = lambda record: (record["source"], record["qid"], record["question"])
    assert [pick(question) for question in questions] == [pick(record) for record in frozen]


def test_only_the_7_false_premises_change_label(questions, frozen):
    changed = {question["qid"] for question, record in zip(questions, frozen) if question["label"] != record["label"]}
    assert changed == set(CONFIG["questions"]["premise_qids"])


def test_labels_and_expected_answers_match_e2(questions, saved):
    by_qid = {question["qid"]: question for question in questions}
    wrong = [row.qid for row in saved.itertuples()
             if (by_qid[row.qid]["label"], by_qid[row.qid]["expected"]) != (row.label, row.expected)]
    assert not wrong, f"{len(wrong)} differ, e.g. {wrong[:5]}"


def test_no_stage_errors(records):
    failed = [record["error"] for record in records.values() if record["error"]]
    assert not failed, failed[:3]


def test_same_gate_decisions(records, frozen):
    pick = lambda record: (record["fixed"], record["final"], record["action"], record["signals"])
    wrong = [record["qid"] for record in frozen if pick(record) != pick(records[record["qid"]])]
    assert not wrong, f"{len(wrong)} differ, e.g. {wrong[:5]}"
    assert sum(record["action"] in pipeline.TO_LLM for record in records.values()) == 104


def test_every_e2_prompt_rebuilt_exactly(records, saved):
    rebuilt = [pipeline.prompt(records[row.qid], row.version, row.format) for row in saved.itertuples()]
    wrong = [row.qid for row, new in zip(saved.itertuples(), rebuilt)
             if (new["system"], new["prompt"], new["prompt_hash"]) != (row.system, row.prompt, row.prompt_hash)]
    assert len(rebuilt) == 1297
    assert not wrong, f"{len(wrong)} differ, e.g. {sorted(set(wrong))[:5]}"


def test_hits_keep_every_retriever_field(records):
    to_llm = [record for record in records.values() if record["action"] in pipeline.TO_LLM]
    missing = [record["qid"] for record in to_llm if any(set(hit) != HIT_FIELDS for hit in record["hits"])]
    assert not missing, f"{len(missing)} records lost hit fields, e.g. {missing[:5]}"