"""Retrieval + gate regression (Phase 2, step 3): Corrective.prepare() on every frozen question must give the
same fixed question, grade, action, signals and hits as the frozen E2 evidence.
Run from the project root:   python -m pytest tests/test_retrieval.py -q     (local models, no API calls; 1-2 min)"""
import json

import pytest

from src.corrective import Corrective
from src.paths import EVAL_DIR
from src.query import SpellFixer
from src.retriever import Retriever

EVIDENCE_FILE = EVAL_DIR / "benchmark" / "prompt_ab_evidence.jsonl"
HIT_KEYS = ("chunk_id", "doc_type", "path", "printed_pages", "text")


@pytest.fixture(scope="module")
def frozen():
    with open(EVIDENCE_FILE, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


@pytest.fixture(scope="module")
def retriever():
    return Retriever()


@pytest.fixture(scope="module")
def fresh(frozen, retriever):
    gate = Corrective(retriever, SpellFixer(retriever.chunks))
    return {record["qid"]: gate.prepare(record["question"]) for record in frozen}


def differences(frozen, fresh, read):
    """qids (with frozen and fresh value) where read(record) differs between the frozen and the fresh run."""
    return [(record["qid"], read(record), read(fresh[record["qid"]]))
            for record in frozen if read(record) != read(fresh[record["qid"]])]


def test_index_has_427_chunks(retriever):
    assert len(retriever.chunks) == 427


def test_all_173_questions_present(frozen):
    assert len(frozen) == 173


def test_same_fixed_question(frozen, fresh):
    found = differences(frozen, fresh, lambda record: record["fixed"])
    assert not found, f"{len(found)} differ: {found[:3]}"


def test_same_grade_and_action(frozen, fresh):
    found = differences(frozen, fresh, lambda record: (record["final"], record["action"]))
    assert not found, f"{len(found)} differ: {found[:5]}"


def test_same_signals(frozen, fresh):
    found = differences(frozen, fresh, lambda record: record["signals"])
    assert not found, f"{len(found)} differ: {found[:3]}"


TO_LLM = ("answer", "answer_with_note")       # only these questions send their hits to the model


def test_same_hit_ids_for_questions_that_reach_the_llm(frozen, fresh):
    to_llm = [record for record in frozen if record["action"] in TO_LLM]
    found = differences(to_llm, fresh, lambda record: [hit["chunk_id"] for hit in record["hits"]])
    assert len(to_llm) == 104
    assert not found, f"{len(found)} differ: {found[:3]}"


def test_same_hit_text_and_pages_for_questions_that_reach_the_llm(frozen, fresh):
    to_llm = [record for record in frozen if record["action"] in TO_LLM]
    found = differences(to_llm, fresh, lambda record: [{key: hit[key] for key in HIT_KEYS} for hit in record["hits"]])
    assert not found, f"{len(found)} differ: {[qid for qid, old, new in found][:10]}"