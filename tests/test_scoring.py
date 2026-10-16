"""E2 regression for scoring (Phase 2, step 1c): scoring the saved E2 answers must give the frozen E2 results.
Run from the project root:   python -m pytest tests/test_scoring.py -q        (no API calls)"""
import pandas as pd
import pytest

from src import evaluate as ev
from src.paths import EVAL_DIR

BENCHMARK = EVAL_DIR / "benchmark"
SCORING_INPUT = BENCHMARK / "e2_scoring_input.jsonl"

SCREEN_MODELS = ["gemini:gemini-3.5-flash-lite", "groq:openai/gpt-oss-120b", "groq:openai/gpt-oss-20b",
                 "groq:qwen/qwen3.8-27b", "nvidia:google/gemma-4-31b-it",
                 "nvidia:nvidia/nemotron-3-super-120b-a12b", "nvidia:z-ai/glm-5.3-flash"]
TIER1 = ["gemini:gemini-3.5-flash-lite", "groq:qwen/qwen3.8-27b", "nvidia:z-ai/glm-5.3-flash"]
E2_PAIRS = ([(model, "v1") for model in SCREEN_MODELS]
            + [(model, version) for version in ("v3b_med", "v3b_strict") for model in TIER1])
E2_RUNS = 2

# (correct on answerable %, over-answer on unanswerable %, statements cited %): 40 questions x 2 runs per pair
E2_SCORECARD = {
    ("gemini:gemini-3.5-flash-lite", "v1"): (78.6, 8.3, 80.9),
    ("gemini:gemini-3.5-flash-lite", "v3b_med"): (87.5, 8.3, 75.5),
    ("gemini:gemini-3.5-flash-lite", "v3b_strict"): (75.0, 8.3, 75.7),
    ("groq:openai/gpt-oss-120b", "v1"): (82.1, 8.3, 63.9),
    ("groq:openai/gpt-oss-20b", "v1"): (87.5, 25.0, 63.0),
    ("groq:qwen/qwen3.8-27b", "v1"): (82.1, 8.3, 78.7),
    ("groq:qwen/qwen3.8-27b", "v3b_med"): (78.6, 8.3, 80.6),
    ("groq:qwen/qwen3.8-27b", "v3b_strict"): (64.3, 8.3, 84.9),
    ("nvidia:google/gemma-4-31b-it", "v1"): (91.1, 16.7, 76.2),
    ("nvidia:nvidia/nemotron-3-super-120b-a12b", "v1"): (78.6, 8.3, 66.7),
    ("nvidia:z-ai/glm-5.3-flash", "v1"): (94.6, 16.7, 76.7),
    ("nvidia:z-ai/glm-5.3-flash", "v3b_med"): (92.9, 16.7, 73.1),
    ("nvidia:z-ai/glm-5.3-flash", "v3b_strict"): (89.3, 16.7, 74.8),
}


@pytest.fixture(scope="module")
def scored():
    rows = ev.load_scoring_input(SCORING_INPUT)
    screen = pd.read_csv(BENCHMARK / "screen_40.csv")
    chosen = ev.select(rows, E2_PAIRS, screen.qid, fmt="f0", max_run=E2_RUNS)
    return ev.score_run(chosen[ev.SCORING_COLUMNS])


def test_scoring_input_is_complete():
    rows = ev.load_scoring_input(SCORING_INPUT)
    assert len(rows) == 1297
    assert rows.prompt.str.strip().ne("").all()


def test_every_pair_has_80_answers(scored):
    sizes = scored.groupby(["model", "version"]).size()
    assert set(sizes.index) == set(E2_PAIRS)
    assert sizes.eq(80).all() and len(scored) == 1040


def test_scorecard_matches_e2(scored):
    cards = ev.scorecards(scored)
    for pair, expected in E2_SCORECARD.items():
        card = cards.loc[pair]
        found = (card["correct on answerable %"], card["over-answer on unanswerable %"], card["statements cited %"])
        assert found == pytest.approx(expected, abs=0.05), pair


def test_trade_off_matches_e2(scored):
    table, selected = ev.trade_off(scored, TIER1, ["v1", "v3b_med", "v3b_strict"], retrieval_miss_qids={"NHM09"},
                                   accepted_corrections=set(), p_answerable=0.7, k_values=(1, 2, 3), chosen_k=3)
    assert selected == ("gemini:gemini-3.5-flash-lite", "v3b_med")
    assert table.loc[selected, "cost k=3"] == pytest.approx(8.8)
    assert table["beats selected if k <"].dropna().to_dict() == pytest.approx({
        ("nvidia:z-ai/glm-5.3-flash", "v1"): 1.83,
        ("nvidia:z-ai/glm-5.3-flash", "v3b_med"): 1.38,
        ("nvidia:z-ai/glm-5.3-flash", "v3b_strict"): 0.46})