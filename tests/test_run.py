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
        patch.setattr(run, "chat", forbidden)
        patch.setattr(run, "real_answer_check", forbidden)
        return run.main(argv, runs_dir=runs_dir)


def read_lines(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


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
    saved = [{"model": "gemini:gemini-3.5-flash-lite", "version": "v1", "format": "f0", "run": run_number, "qid": qid}
             for run_number, qid in [(1, "NHM02"), (1, "NHM21"), (2, "NHM02")]]
    pd.DataFrame(saved).to_csv(folder / "answers.csv", index=False)
    runner(["--resume", folder.name], tmp_path)
    progress = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))["progress"]
    assert (progress["answers_saved"], progress["pending"]) == (3, 1037)


def test_calls_save_every_answer_with_its_prompt(dry, tmp_path):
    """--allow-api-calls with a stand-in model: all 1,040 answers saved, each with the exact prompt it was sent."""
    folder = copy_of(dry, tmp_path)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(run, "chat", lambda model, system, user, timeout, tries: "Not found in the guideline.")
        patch.setattr(run, "real_answer_check", lambda model, timeout, tries: "42")
        patch.setattr(run.time, "sleep", lambda seconds: None)
        run.main(["--resume", folder.name, "--allow-api-calls"], runs_dir=tmp_path)
    answers = pd.read_csv(folder / "answers.csv")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    rehashed = [pipeline.prompt_hash(system, prompt) for system, prompt in zip(answers.system, answers.prompt)]
    assert len(answers) == 1040 and manifest["status"] == "complete" and manifest["progress"]["pending"] == 0
    assert list(answers.prompt_hash) == rehashed


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