"""Prompt regression (Phase 2, step 2): src/prompts.py must rebuild every E2 prompt character for character.
Run from the project root:   python -m pytest tests/test_prompts.py -q        (no API calls)"""
import json

import pytest

from src import evaluate as ev
from src import prompts
from src.paths import EVAL_DIR


from src import pipeline, prompts
from src.config import load

BENCHMARK = EVAL_DIR / "benchmark"


@pytest.fixture(scope="module")
def saved():
    return ev.load_scoring_input(BENCHMARK / "e2_scoring_input.jsonl")


@pytest.fixture(scope="module")
def evidence():
    with open(BENCHMARK / "prompt_ab_evidence.jsonl", encoding="utf-8") as file:
        return {record["qid"]: record for record in map(json.loads, file)}


def test_every_saved_prompt_is_rebuilt_exactly(saved, evidence):
    rebuilt = [prompts.build(evidence[row.qid], row.version, row.format) for row in saved.itertuples()]
    wrong_system = [row.qid for row, (system, prompt) in zip(saved.itertuples(), rebuilt) if system != row.system]
    wrong_prompt = [row.qid for row, (system, prompt) in zip(saved.itertuples(), rebuilt) if prompt != row.prompt]
    assert not wrong_system, f"{len(wrong_system)} system prompts differ, e.g. {wrong_system[:5]}"
    assert not wrong_prompt, f"{len(wrong_prompt)} user prompts differ, e.g. {wrong_prompt[:5]}"


def test_versions_and_formats_cover_e2(saved):
    assert set(saved.version) <= set(prompts.VERSIONS)
    assert set(saved.format) <= set(prompts.FORMATS)

def test_nhm_document_config_gives_the_e2_system_prompts():
    """C-040: the system prompts built from config/docs/nhm_hrh_2022.yaml are the E2 prompts character for character."""
    doc = pipeline.document(load())
    assert (prompts.system_v3b_med(doc), prompts.system_v3b_strict(doc)) == (prompts.SYSTEM_V3B_MED, prompts.SYSTEM_V3B_STRICT)


def test_another_document_changes_only_its_names():
    """C-040: a second document gets its own title, words and not-found reply, and nothing of NHM is left."""
    doc = {"title": "NFHS-6 Fact Sheets", "word": "fact sheet", "outside": "values from other surveys",
           "not_found": "Not found in the fact sheets."}
    system = prompts.system_v3b_med(doc)
    assert doc["title"] in system and doc["not_found"] in system and "NHM" not in system and "guideline" not in system