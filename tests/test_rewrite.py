"""Live query rewriting (C-029): the rewrite is one clean line with its time, a failed call falls back to the
SpellFixer, the gate searches the rewrite when there is one, and a dry run with the rewrite front end makes no call.
No API calls - the LLM is replaced by a stand-in.
Run from the project root:   python -m pytest tests/test_rewrite.py -q"""
import json
from types import SimpleNamespace
import pytest
import yaml
from spellchecker import SpellChecker
from src import pipeline, rewrite, run
from src.config import load
from src.llm import LLMError

TYPED = "how lng is the probaton period"
CLEAN = "How long is the probation period?"


def forbidden(*args, **kwargs):
    raise AssertionError("an API call was attempted")


def asked(text, error=""):
    return {"model": "gemini:test", "version": "r1", "text": text, "seconds": 0.5, "error": error}


def test_ask_returns_one_line_and_its_time(monkeypatch):
    monkeypatch.setattr(rewrite, "chat_full", lambda model, system, user, **options: {"text": " How long\n is it? "})
    result = rewrite.ask(TYPED, load())
    assert (result["text"], result["error"], result["version"]) == ("How long is it?", "", "r1")
    assert result["seconds"] >= 0


def test_ask_failure_gives_empty_text_and_the_error(monkeypatch):
    def fail(model, system, user, **options):
        raise LLMError("quota")
    monkeypatch.setattr(rewrite, "chat_full", fail)
    result = rewrite.ask(TYPED, load())
    assert (result["text"], result["error"]) == ("", "quota")


def test_prepare_searches_the_rewrite(gate):
    record = pipeline.prepare({"qid": "T1", "question": TYPED, "error": None}, gate, asked(CLEAN))
    assert (record["fixed"], record["changes"]) == (CLEAN, [])
    assert record["rewrite"]["text"] == CLEAN


def test_failed_rewrite_falls_back_to_the_spellfixer(gate):
    question = {"qid": "T1", "question": TYPED, "error": None}
    record = pipeline.prepare(question, gate, asked("", "quota"))
    assert record["fixed"] == pipeline.prepare(question, gate)["fixed"]
    assert record["rewrite"]["error"] == "quota"

@pytest.mark.parametrize("front_end", ["rewrite", "rewrite_if_unknown"])
def test_dry_run_with_a_rewrite_front_end_calls_nothing_and_writes_nothing(tmp_path, monkeypatch, front_end):
    experiment = tmp_path / "live.yaml"
    experiment.write_text(yaml.safe_dump({
        "experiment": "live", "purpose": "test", "docs": ["nhm_hrh_2022"], "question_set": "all",
        "query": {"front_end": front_end}, "plan": [{"versions": ["v3b_med"], "models": ["gemini:test"]}],
        "format": "f0p", "runs": 2}), encoding="utf-8")
    monkeypatch.setattr(rewrite, "chat_full", forbidden)
    monkeypatch.setattr(run, "chat_full", forbidden)
    assert run.main(["--config", str(experiment)], runs_dir=tmp_path / "runs") is None
    assert not (tmp_path / "runs").exists()


def test_live_start_rewrites_every_question_and_counts_failures(tmp_path, monkeypatch):
    """Five questions with a stand-in LLM that echoes the question and fails once: 5 calls, 1 failure, every rewrite
    kept in the evidence and the trace; no pause, no answer call (start only prepares)."""
    experiment = tmp_path / "live.yaml"
    experiment.write_text(yaml.safe_dump({
        "experiment": "live", "purpose": "test", "docs": ["nhm_hrh_2022"], "question_set": "all",
        "query": {"front_end": "rewrite"}, "llm": {"pause": {"gemini": 0}},
        "plan": [{"versions": ["v3b_med"], "models": ["gemini:test"]}], "format": "f0p", "runs": 2}), encoding="utf-8")
    calls = []

    def stand_in(model, system, user, **options):
        calls.append(user)
        if len(calls) == 3:
            raise LLMError("quota")
        return {"text": user}
    monkeypatch.setattr(rewrite, "chat_full", stand_in)
    monkeypatch.setattr(run, "chat_full", forbidden)
    monkeypatch.setattr(run, "select", lambda questions, config: questions[:5])
    folder, manifest, config, records = run.start(str(experiment), tmp_path / "runs")
    assert len(calls) == 5
    assert (manifest["plan"]["rewrites"]["calls"], manifest["plan"]["rewrites"]["failed"]) == (5, 1)
    assert sum(record["rewrite"]["error"] == "quota" for record in records) == 1
    assert "rewrite failed, SpellFixer used" in (folder / "run.log").read_text(encoding="utf-8")
    assert all(line["normalize"]["rewrite"] for line in map(json.loads, (folder / "traces.jsonl").open(encoding="utf-8")))

def test_unknown_words_decide_the_rewrite():
    """C-032: Hindi words and typos count; what the rewrite must keep as written (abbreviations, names inside a
    sentence) and contractions do not; the first word of a sentence still counts."""
    english, vocab = SpellChecker(), {"nhm", "anm", "probation"}
    assert rewrite.unknown_words("probation period kitna hota hai", vocab, english) == ["kitna", "hota", "hai"]
    assert rewrite.unknown_words("is ther a parking alowance", vocab, english) == ["ther", "alowance"]
    assert rewrite.unknown_words("Probaton kitne din ka hota hai?", vocab, english) == ["Probaton", "kitne", "hota", "hai"]
    assert rewrite.unknown_words("Can an ANM from Infosys work in the ICU? I've asked NHM.", vocab, english) == []

def test_needed_follows_the_front_end():
    """C-032: spellfixer never asks, rewrite always asks, rewrite_if_unknown asks only for unknown words."""
    speller = SimpleNamespace(vocab={"probation", "period"}, english=SpellChecker())
    clean, hinglish = "How long is the probation period?", "probation period kitna hota hai"
    assert [rewrite.needed(clean, front_end, speller) for front_end in ("spellfixer", "rewrite", "rewrite_if_unknown")] \
        == [False, True, False]
    assert rewrite.needed(hinglish, "rewrite_if_unknown", speller) and not rewrite.needed(hinglish, "spellfixer", speller)


def test_live_start_asks_only_questions_with_unknown_words(tmp_path, monkeypatch):
    """NHM01 has no unknown word, NHM13 has one ("aviable"): with rewrite_if_unknown only NHM13 is sent."""
    experiment = tmp_path / "live.yaml"
    experiment.write_text(yaml.safe_dump({
        "experiment": "live", "purpose": "test", "docs": ["nhm_hrh_2022"], "question_set": "all",
        "query": {"front_end": "rewrite_if_unknown"}, "llm": {"pause": {"gemini": 0}},
        "plan": [{"versions": ["v3b_med"], "models": ["gemini:test"]}], "format": "f0p", "runs": 2}), encoding="utf-8")
    calls = []
    monkeypatch.setattr(rewrite, "chat_full", lambda model, system, user, **options: calls.append(user) or {"text": user})
    monkeypatch.setattr(run, "chat_full", forbidden)
    monkeypatch.setattr(run, "select", lambda questions, config: [item for item in questions if item["qid"] in ("NHM01", "NHM13")])
    folder, manifest, config, records = run.start(str(experiment), tmp_path / "runs")
    by_qid = {record["qid"]: record for record in records}
    assert calls == [by_qid["NHM13"]["question"]] and "rewrite" not in by_qid["NHM01"]
    assert manifest["plan"]["rewrites"]["calls"] == 1 and manifest["plan"]["front_end"] == "rewrite_if_unknown"