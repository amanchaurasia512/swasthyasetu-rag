"""Gate switches (C-028): config gate.m_action / gate.p_notes / gate.re_retrieve change only what they name, and
their old values keep today's behaviour. The grade is forced, so the tests hold for any document and question.
Run from the project root:   python -m pytest tests/test_switches.py -q        (local models, no API calls)"""
import pytest

from src import prompts
from src.corrective import Corrective

QUESTION = "How many members should the recruitment committee have?"
WEAK = {"best": -2.0, "hard_out": ["XYZ"], "soft_out": [], "missing": []}     # weak evidence + a word not in the PDF


@pytest.fixture
def make_gate(gate, monkeypatch):
    """A fresh gate on the shared retriever, with forced signals and grade and the given switches."""
    def build(grade: str, **switches) -> Corrective:
        fresh = Corrective(gate.R, gate.SP, refine=gate.refine_on, **switches)
        monkeypatch.setattr(fresh, "signals", lambda fixed, hits: WEAK)
        monkeypatch.setattr(fresh, "grade", lambda signals: grade)
        return fresh
    return build


def test_m_scoped_keeps_the_fixed_answer(make_gate):
    record = make_gate("M").prepare(QUESTION)
    assert record["action"] == "scoped" and record["fixed_answer"].startswith("The guideline does not cover")


def test_m_answer_sends_the_question_to_the_llm(make_gate):
    record = make_gate("M", m_action="answer").prepare(QUESTION)
    assert (record["final"], record["action"], record["fixed_answer"]) == ("M", "answer", "")
    assert record["prompt"] and "Important:" not in record["prompt"]


def test_p_without_notes_is_a_plain_answer(make_gate):
    record = make_gate("P", p_notes=False).prepare(QUESTION)
    assert record["action"] == "answer" and "Important:" not in record["prompt"]
    assert "Important:" not in prompts.build(record, "v3b_med", "f0p", notes=False)[1]


def test_p_with_notes_keeps_the_warning(make_gate):
    record = make_gate("P").prepare(QUESTION)
    assert record["action"] == "answer_with_note" and "Important:" in prompts.build(record, "v3b_med", "f0p")[1]


def test_re_retrieve_runs_only_when_on(make_gate, monkeypatch):
    calls = []
    for switch in (True, False):
        fresh = make_gate("P", re_retrieve=switch)
        monkeypatch.setattr(fresh, "re_retrieve", lambda question, signals: calls.append(switch) or fresh.R.search(question, k=5))
        fresh.prepare(QUESTION)
    assert calls == [True]