"""Similarity metrics (C-037): citation marks removed, identical text scores 1, an empty answer scores 0, and a run
with several models is compared on the answers all of them have.
Run from the project root:   python -m pytest tests/test_text_metrics.py -q      (no models, no API calls; seconds)"""
import pandas as pd

from src.checks import text_metrics

KEY = "The probation period is six months."


def test_citations_are_removed():
    assert text_metrics.plain("The probation period is six months [1, 2].") == KEY


def test_identical_answer_scores_one_and_empty_scores_zero():
    same, empty = text_metrics.measures(KEY + " [2]", KEY), text_metrics.measures("", KEY)
    assert (round(same["rouge_l"], 3), round(same["token_f1"], 3), same["empty"]) == (1.0, 1.0, False)
    assert (empty["rouge_l"], empty["bleu"], empty["token_f1"], empty["empty"]) == (0.0, 0.0, 0.0, True)


def test_models_are_compared_on_shared_answers():
    scored = pd.DataFrame({"model": ["a", "a", "b"], "qid": ["Q1", "Q2", "Q1"], "run": [1, 1, 1]})
    assert list(text_metrics.common_answers(scored).qid) == ["Q1", "Q1"]