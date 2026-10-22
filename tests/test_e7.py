"""E7 (C-044..C-046): the NFHS-6 experiment differs from E5 only in its document and questions, its expected answers
come from the indicator table, and an experiment with only a 'new' question set loads.
Run from the project root:   python -m pytest tests/test_e7.py -q      (no models, no API calls)"""
import pandas as pd
import yaml

from src import pipeline
from src.checks import nfhs_questions
from src.config import ROOT


def test_e7_differs_from_e5_only_in_document_and_questions():
    e5, e7 = (yaml.safe_load(open(ROOT / f"config/experiments/{name}.yaml", encoding="utf-8")) for name in ("e5", "e7"))
    same = {key for key in e5 if key not in ("experiment", "purpose", "docs", "questions")}
    assert {key: e5[key] for key in same} == {key: e7[key] for key in same} and e7["docs"] == ["nfhs6"]


def test_expected_answer_is_built_from_the_indicator_table(tmp_path):
    table = pd.DataFrame({"area": ["Bihar", "Kerala"], "indicator": ["18", "18"],
                          "name": ["Total fertility rate (children per woman)"] * 2,
                          "total_6": ["2.7", "1.6"], "total_5": ["3.0", "1.8"]})
    source = tmp_path / "questions.csv"
    pd.DataFrame({"qid": ["F01", "F02"], "question": ["TFR change in Bihar and Kerala?", "Anaemia?"],
                  "label": ["answerable", "unanswerable"], "area": ["Bihar;Kerala", ""], "indicator": ["18", ""],
                  "column": ["total_6;total_5", ""]}).to_csv(source, index=False)
    built = nfhs_questions.build(str(source), table)
    assert built.expected_answer.tolist() == [
        "Bihar Total fertility rate 2.7; Bihar Total fertility rate 3.0; Kerala Total fertility rate 1.6; "
        "Kerala Total fertility rate 1.8", ""]


def test_an_experiment_with_only_new_questions_loads(tmp_path):
    file = tmp_path / "new.csv"
    pd.DataFrame({"qid": ["F01"], "label": ["answerable"], "question": ["TFR in Bihar?"],
                  "expected_answer": ["Bihar Total fertility rate 2.7"]}).to_csv(file, index=False)
    loaded = pipeline.load_questions({"questions": {"new": {"file": str(file)}}})
    assert [(item["source"], item["qid"], item["expected"]) for item in loaded] == [("new", "F01", "Bihar Total fertility rate 2.7")]