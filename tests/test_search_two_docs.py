"""Search test on two documents (C-042): the right NFHS-6 chunk comes from the indicator table, and the summary counts
finds and refusals per question set and index.
Run from the project root:   python -m pytest tests/test_search_two_docs.py -q      (no models, no API calls)"""
import pandas as pd

from src.checks import search_two_docs

INDICATORS = pd.DataFrame({"area": ["Bihar", "Kerala"], "indicator": ["18", "18"], "pdf_page": [43, 75]})


def test_right_chunk_comes_from_the_indicator_table(tmp_path):
    file = tmp_path / "questions.csv"
    pd.DataFrame({"qid": ["F01", "F02"], "question": ["TFR in Bihar and Kerala?", "Anaemia in Patna?"],
                  "label": ["answerable", "unanswerable"], "area": ["Bihar;Kerala", ""], "indicator": [18, None],
                  "column": ["total_6", ""], "kind": ["compare", "unanswerable"], "language": ["en", "en"]}).to_csv(file, index=False)
    answerable, unanswerable = search_two_docs.nfhs_questions(str(file), INDICATORS)
    assert answerable["gold"] == ["N43-018", "N75-018"] and unanswerable["gold"] == []


def test_summary_counts_finds_and_refusals():
    results = pd.DataFrame({"questions": ["nfhs6"] * 3, "index": ["nfhs6"] * 3, "rewritten": [False, True, False],
                            "expected": ["found", "found", "refused"], "gold_rank": [1, None, None],
                            "all_gold_top5": [True, False, None], "area_top5": [True, True, None], "grade": ["S", "P", "I"]})
    row = search_two_docs.summary(results).iloc[0]
    assert (row.gold_top5, row.area_top5, row.found_S_or_P, row.refused_I, row.rewritten) == (1, 2, 2, 1, 1)
