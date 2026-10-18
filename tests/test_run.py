"""Runner (Phase 2, step 5b): a dry run of config/experiments/e2.yaml must plan exactly E2's 1,040 calls with E2's
prompts, make no API call, and write the manifest, evidence, trace and log; --resume must skip saved answers and
refuse when the settings or an input file changed.
Run from the project root:   python -m pytest tests/test_run.py -q     (local models, no API calls; ~2 min)"""
import json
import shutil

import pandas as pd
import pytest

from src import pipeline, run
from src.config import path_of

E2 = "config/experiments/e2.yaml"
STAGE_KEYS = {"run_id", "qid", "doc_id", "normalize", "retrieve", "gate", "evidence", "prompt", "seconds", "error"}


def forbidden(*args, **kwargs):
    raise AssertionError("an API call was attempted")


def runner(argv, runs_dir):
    """run.main with every API entry point blocked."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(run, "chat_full", forbidden)
        patch.setattr(run, "real_answer_check", forbidden)
        return run.main(argv, runs_dir=runs_dir)

def read_lines(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]
    
def reply(text, finish_reason="stop", answer_tokens=12):
    """What llm.chat_full returns."""
    return {"text": text, "prompt_tokens": 900, "answer_tokens": answer_tokens, "reasoning_tokens": None,
            "finish_reason": finish_reason}

@pytest.fixture(scope="module")
def dry(tmp_path_factory):
    return runner(["--config", E2], tmp_path_factory.mktemp("runs"))


@pytest.fixture(scope="module")
def manifest(dry):
    return json.loads((dry / "manifest.json").read_text(encoding="utf-8"))


def copy_of(dry, tmp_path):
    shutil.copytree(dry, tmp_path / dry.name)
    return tmp_path / dry.name


def test_dry_run_plans_exactly_e2(manifest):
    assert manifest["status"] == "dry_run"
    assert (manifest["plan"]["questions"], manifest["plan"]["to_llm"]) == (40, 40)
    assert len(manifest["plan"]["pairs"]) == 13
    assert manifest["plan"]["planned_calls"] == manifest["progress"]["pending"] == 1040
    assert manifest["progress"]["answers_saved"] == 0


def test_planned_prompts_are_e2_prompts(dry, manifest):
    evidence = {record["qid"]: record for record in read_lines(dry / "evidence.jsonl")}
    saved = pd.read_json(path_of("data/eval/benchmark/e2_scoring_input.jsonl"), lines=True, dtype=False)
    plan = {tuple(pair) for pair in manifest["plan"]["pairs"]}
    in_plan = pd.Series([(model, version) in plan for model, version in zip(saved.model, saved.version)], index=saved.index)
    e2 = saved[in_plan & saved.qid.isin(evidence)]
    wrong = [row.qid for row in e2.itertuples()
             if pipeline.prompt(evidence[row.qid], row.version, row.format)["prompt_hash"] != row.prompt_hash]
    assert len(e2) == 1040
    assert not wrong, f"{len(wrong)} differ, e.g. {wrong[:5]}"


def test_run_folder_files(dry):
    traces = read_lines(dry / "traces.jsonl")
    assert len(read_lines(dry / "evidence.jsonl")) == len(traces) == 40
    assert all(set(line) == STAGE_KEYS for line in traces)
    assert all(len(line["prompt"]) == 3 for line in traces)          # v1, v3b_med, v3b_strict
    assert (dry / "run.log").exists() and not (dry / "answers.csv").exists()


def test_resume_skips_saved_answers(dry, tmp_path):
    folder = copy_of(dry, tmp_path)
    saved = [{"model": "gemini:gemini-3.5-flash-lite", "version": "v1", "format": "f0", "run": run_number, "qid": qid,
              "answer": "Not found in the guideline.", "prompt": ""}
             for run_number, qid in [(1, "NHM02"), (1, "NHM21"), (2, "NHM02")]]    
    pd.DataFrame(saved).to_csv(folder / "answers.csv", index=False)
    runner(["--resume", folder.name], tmp_path)
    progress = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))["progress"]
    assert (progress["answers_saved"], progress["pending"]) == (3, 1037)


def test_calls_save_every_answer_with_its_prompt(dry, tmp_path):
    """--allow-api-calls with a stand-in model: all 1,040 answers saved, each with the exact prompt it was sent."""
    folder = copy_of(dry, tmp_path)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(run, "chat_full", lambda model, system, user, timeout, tries: reply("Not found in the guideline."))
        patch.setattr(run, "real_answer_check", lambda model, timeout, tries: "42")
        patch.setattr(run.time, "sleep", lambda seconds: None)
        run.main(["--resume", folder.name, "--allow-api-calls"], runs_dir=tmp_path)
    answers = pd.read_csv(folder / "answers.csv")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    rehashed = [pipeline.prompt_hash(system, prompt) for system, prompt in zip(answers.system, answers.prompt)]
    assert len(answers) == 1040 and manifest["status"] == "complete" and manifest["progress"]["pending"] == 0
    assert list(answers.prompt_hash) == rehashed

def test_truncated_answers_are_saved_flagged_and_counted(dry, tmp_path):
    """Option A: a cut-off answer is scored as usual, flagged in answers.csv and counted in scorecard and manifest."""
    folder = copy_of(dry, tmp_path)
    cut_off = {"NHM23", "NHM03"}

    def stand_in(model, system, user, timeout, tries):
        long_list = any(f"Question: {question}" in user for question in questions_of(folder, cut_off))
        return reply("1. Stage one [1]\n2. Stage", "length", 2048) if long_list else reply("Not found in the guideline.")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(run, "chat_full", stand_in)
        patch.setattr(run, "real_answer_check", lambda model, timeout, tries: "42")
        patch.setattr(run.time, "sleep", lambda seconds: None)
        run.main(["--resume", folder.name, "--allow-api-calls"], runs_dir=tmp_path)
    answers = pd.read_csv(folder / "answers.csv")
    scoring = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))["scoring"]
    cards = pd.read_csv(folder / "scorecard.csv", index_col=["model", "version"])
    assert set(answers.columns) >= {"prompt_tokens", "answer_tokens", "reasoning_tokens", "finish_reason", "truncated"}
    assert set(answers[answers.truncated].qid) == cut_off
    assert scoring["truncated"] == cards["truncated"].sum() == 2 * 13 * 2          # 2 questions x 13 pairs x 2 runs
    assert scoring["scored_answers"] == 1040                                       # still scored (option A)
    assert scoring["answer_tokens"]["gemini:gemini-3.5-flash-lite"] == {"min": 12, "median": 12.0, "max": 2048}


def questions_of(folder, qids):
    return [record["fixed"] for record in read_lines(folder / "evidence.jsonl") if record["qid"] in qids]

def test_resume_refuses_changed_input_file(dry, tmp_path):
    folder = copy_of(dry, tmp_path)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    manifest["data"]["fingerprints"]["set_C"] = "000000000000"
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit, match="set_C"):
        runner(["--resume", folder.name], tmp_path)


def test_resume_refuses_changed_settings(dry, tmp_path):
    folder = copy_of(dry, tmp_path)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    manifest["config"]["hash"] = "000000000000"
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit, match="settings"):
        runner(["--resume", folder.name], tmp_path)

def test_runner_keeps_its_command_line_entry_point():
    """python -m src.run needs these two lines at the end of src/run.py; they were lost twice when main was
    replaced by copy-paste (C-013, C-029), and every other test still passed because they call main() directly."""
    with open(run.__file__, encoding="utf-8") as file:
        source = file.read()
    assert source.rstrip().endswith('if __name__ == "__main__":\n    main()')