"""Full regression (Phase 3): E2 end to end through the new code path must give E2's results.

    question files -> pipeline (fresh retrieval, gate) -> runner builds and "sends" every prompt
    -> a stand-in model replies with E2's saved answer for exactly that prompt -> runner saves, scores, scorecard

No API calls. The stand-in only knows E2's prompts, so a prompt the new path builds differently fails the test.
Run from the project root:   python -m pytest tests/test_regression.py -q     (local models; ~3 min)"""
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

from src import evaluate as ev
from src import run
from src.config import path_of
from test_scoring import E2_SCORECARD

E2 = "config/experiments/e2.yaml"
LABELS = ["refused", "llm_label"]
MEASURES = ["correctness", "statements", "cite_recall", "cite_precision"]   # floats: np.isclose (the CSV round trip changes the last digit)


@pytest.fixture(scope="module")
def e2_saved():
    """E2's 1,040 answers with the prompt each one was given (C-003), as test_scoring.py selects them."""
    rows = ev.load_scoring_input(path_of("data/eval/benchmark/e2_scoring_input.jsonl"))
    screen = pd.read_csv(path_of("data/eval/benchmark/screen_40.csv"))
    pairs = run.pairs(run.load(E2))
    return ev.select(rows, pairs, screen.qid, fmt="f0", max_run=2).sort_values("run")


@pytest.fixture(scope="module")
def e2_run(tmp_path_factory, e2_saved):
    """The runner's folder after a dry run and a resumed run whose 'model' replays E2's answers."""
    runs_dir = tmp_path_factory.mktemp("runs")
    replies = defaultdict(list)                  # (model, system, prompt) -> E2 answers, run 1 first
    for row in e2_saved.itertuples():
        replies[(row.model, row.system, row.prompt)].append(row.answer)

    def stand_in(model, system, user, timeout, tries):
        assert (model, system, user) in replies, f"{model}: the new path built a prompt E2 never sent"
        return replies[(model, system, user)].pop(0)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(run, "chat", stand_in)
        patch.setattr(run, "real_answer_check", lambda model, timeout, tries: "42")
        patch.setattr(run.time, "sleep", lambda seconds: None)
        folder = run.main(["--config", E2], runs_dir=runs_dir)
        run.main(["--resume", folder.name, "--allow-api-calls"], runs_dir=runs_dir)
    return folder, replies


def test_every_e2_answer_replayed_once(e2_run):
    folder, replies = e2_run
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete" and manifest["progress"]["answers_saved"] == 1040
    assert not any(replies.values()), "some E2 answers were never asked for"


def test_per_answer_scores_equal_e2(e2_run, e2_saved):
    folder, replies = e2_run
    new = pd.read_csv(folder / "scored.csv")
    old = ev.score_run(e2_saved[ev.SCORING_COLUMNS])
    joined = new.merge(old, on=["model", "version", "run", "qid"], suffixes=("_new", "_e2"))
    assert len(joined) == 1040
    differ = {label: int((joined[f"{label}_new"] != joined[f"{label}_e2"]).sum()) for label in LABELS}
    differ |= {measure: int((~np.isclose(joined[f"{measure}_new"].astype(float), joined[f"{measure}_e2"].astype(float),
                                         equal_nan=True)).sum()) for measure in MEASURES}
    assert not any(differ.values()), differ


def test_scorecard_equals_e2(e2_run):
    folder, replies = e2_run
    cards = pd.read_csv(folder / "scorecard.csv", index_col=["model", "version"])
    for pair, expected in E2_SCORECARD.items():
        card = cards.loc[pair]
        found = (card["correct on answerable %"], card["over-answer on unanswerable %"], card["statements cited %"])
        assert found == pytest.approx(expected, abs=0.05), pair


def test_trade_off_selects_e2_choice(e2_run):
    folder, replies = e2_run
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    table = pd.read_csv(folder / "tradeoff.csv", index_col=["model", "version"])
    selected = ("gemini:gemini-3.5-flash-lite", "v3b_med")
    assert tuple(manifest["scoring"]["selected"]) == selected
    assert table.loc[selected, "cost k=3"] == pytest.approx(8.8)
    assert table["beats selected if k <"].dropna().to_dict() == pytest.approx({
        ("nvidia:z-ai/glm-5.3-flash", "v1"): 1.83,
        ("nvidia:z-ai/glm-5.3-flash", "v3b_med"): 1.38,
        ("nvidia:z-ai/glm-5.3-flash", "v3b_strict"): 0.46})