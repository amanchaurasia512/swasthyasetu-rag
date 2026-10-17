"""Runner: the only code that calls LLM APIs (rule R1). Without --allow-api-calls it is a dry run.

    python -m src.run --config config/experiments/e3.yaml                 new run: stages 1-6, plan, no calls
    python -m src.run --resume e3-20261020-1430                           what is still missing, no calls
    python -m src.run --resume e3-20261020-1430 --allow-api-calls         answer only the missing keys

Writes runs/<run_id>/: manifest.json (the run's ID card), evidence.jsonl (stages 1-5, every hit field),
traces.jsonl (one line per question), answers.csv (one row per answer, saved at once, with its exact prompt), run.log,
and once answers exist: scored.csv, scorecard.csv and (if the experiment sets a trade-off) tradeoff.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

import pandas as pd

from src import evaluate, pipeline, trace
from src.config import config_hash, load, path_of
from src.llm import LLMError, chat_full, provider_of, real_answer_check
from src.paths import ROOT
from dotenv import load_dotenv

RUNS = ROOT / "runs"
PACKAGES = ("chromadb", "fastembed", "onnxruntime", "rank-bm25", "snowballstemmer", "numpy", "pandas", "pyyaml")
ANSWER_KEY = ["model", "version", "format", "run", "qid"]
TOKEN_FIELDS = ["prompt_tokens", "answer_tokens", "reasoning_tokens", "finish_reason"]

# ------------------------------------------------------------------ small helpers
def now() -> str:
    return datetime.now().isoformat(timespec="seconds")

def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]

def package_version(name: str):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None

def log(folder: Path, text: str) -> None:
    print(text, flush=True)
    with open(folder / "run.log", "a", encoding="utf-8") as file:
        file.write(f"{now()}  {text}\n")

def save_manifest(folder: Path, manifest: dict) -> None:
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

# ------------------------------------------------------------------ what this run uses
def input_files(config: dict) -> dict:
    sets = config["questions"]
    return {**{name: sets[name]["file"] for name in ("manual", "set_C", "negatives")},
            **({"new": sets["new"]["file"]} if "new" in sets else {}),
            "screen": sets["screen"], "chunks": config["files"]["chunks"]}

def fingerprints(config: dict) -> dict:
    return {name: file_hash(path_of(relative)) for name, relative in input_files(config).items()}

def pairs(config: dict) -> list[tuple[str, str]]:
    """(model, prompt version) in plan order: group by group, version by version, model by model."""
    return [(model, version) for group in config["plan"] for version in group["versions"] for model in group["models"]]

def versions(config: dict) -> list[str]:
    return list(dict.fromkeys(version for model, version in pairs(config)))

def select(questions: list[dict], config: dict) -> list[dict]:
    if config["question_set"] == "all":
        return questions
    by_qid = {question["qid"]: question for question in questions}
    return [by_qid[qid] for qid in pd.read_csv(path_of(config["questions"]["screen"])).qid]

def to_llm(records: list[dict]) -> list[dict]:
    return [record for record in records if record.get("action") in pipeline.TO_LLM]

# ------------------------------------------------------------------ new run: stages 1-6
def start(experiment: str, runs_dir: Path) -> tuple[Path, dict, dict, list[dict]]:
    config = load(experiment)
    run_id = f"{config['experiment']}-{datetime.now():%Y%m%d-%H%M}"
    folder = runs_dir / run_id
    folder.mkdir(parents=True)                # fails if a run of this experiment started this same minute
    manifest = {
        "run_id": run_id, "experiment": config["experiment"], "purpose": config["purpose"],
        "status": "dry_run", "started_at": now(), "finished_at": None,
        "code": {"git_commit": git("rev-parse", "--short", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
                 "python": platform.python_version(), "packages": {name: package_version(name) for name in PACKAGES}},
        "config": {"file": str(experiment), "hash": config_hash(config), "merged": config},
        "data": {"doc_id": config["doc_id"], "collection": config["index"]["collection"],
                 "fingerprints": fingerprints(config)},
        "plan": {"question_set": config["question_set"], "pairs": pairs(config), "format": config["format"],
                 "runs": config["runs"]},
        "progress": {"answers_saved": 0, "pending": None, "history": []},
    }
    save_manifest(folder, manifest)
    log(folder, f"run {run_id} | {config['purpose']}")

    gate = pipeline.open_gate(config)
    records = []
    for question in select(pipeline.load_questions(config), config):
        began = time.time()
        record = pipeline.prepare(question, gate)
        prepared = time.time() - began
        built = [pipeline.prompt(record, version, config["format"]) for version in versions(config)] \
            if record.get("action") in pipeline.TO_LLM else []
        seconds = {"prepare": round(prepared, 2), "prompt": round(time.time() - began - prepared, 3)}
        trace.append(folder / "evidence.jsonl", record)
        trace.append(folder / "traces.jsonl", trace.trace_line(run_id, config["doc_id"], record, built, seconds))
        log(folder, trace.log_line(record, prepared))
        records.append(record)
    manifest["plan"] |= {"questions": len(records), "to_llm": len(to_llm(records)),
                         "planned_calls": len(to_llm(records)) * len(pairs(config)) * config["runs"]}
    save_manifest(folder, manifest)
    return folder, manifest, config, records

# ------------------------------------------------------------------ resumed run: same folder, same evidence
def reopen(folder: Path) -> tuple[Path, dict, dict, list[dict]]:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    config = load(manifest["config"]["file"])
    changed = [] if config_hash(config) == manifest["config"]["hash"] else ["settings (config hash)"]
    now_prints, then_prints = fingerprints(config), manifest["data"]["fingerprints"]
    changed += [f"{name} (hash {then_prints.get(name)} -> {now_prints[name]})"
                for name in now_prints if now_prints[name] != then_prints.get(name)]
    if changed:
        raise SystemExit(f"Cannot resume {manifest['run_id']}: changed since the run started: {', '.join(changed)}. "
                         f"Start a new run instead.")
    with open(folder / "evidence.jsonl", encoding="utf-8") as file:
        records = [json.loads(line) for line in file]
    log(folder, f"resumed {manifest['run_id']}")
    return folder, manifest, config, records

# ------------------------------------------------------------------ stage 7: generate
def saved_keys(folder: Path) -> set:
    answers = folder / "answers.csv"
    if not answers.exists():
        return set()
    return set(pd.read_csv(answers, usecols=ANSWER_KEY).itertuples(index=False, name=None))

def pending(records: list[dict], config: dict, done: set) -> dict:
    """{(model, version): [(run, record), ...]} - only answer keys not saved yet, in E2's order."""
    jobs = {pair: [(run, record) for run in range(1, config["runs"] + 1) for record in to_llm(records)
                   if (*pair, config["format"], run, record["qid"]) not in done] for pair in pairs(config)}
    return {pair: pair_jobs for pair, pair_jobs in jobs.items() if pair_jobs}

def answer_pair(folder: Path, run_id: str, config: dict, model: str, version: str, jobs: list) -> str:
    """Answers one model and prompt version; every answer is saved at once. An error stops this pair only."""
    llm, pause = config["llm"], config["llm"]["pause"][provider_of(model)[0]]
    try:
        real_answer_check(model, timeout=60, tries=1)
    except LLMError as error:
        return f"skipped - probe failed: {str(error)[:150]}"
    for number, (run, record) in enumerate(jobs, 1):
        built = pipeline.prompt(record, version, config["format"])
        started, began = now(), time.time()
        try:
            reply = chat_full(model, built["system"], built["prompt"], timeout=llm["timeout"], tries=llm["tries"])
        except LLMError as error:
            return f"stopped after {number - 1}/{len(jobs)} - {str(error)[:200]}"
        row = {"run_id": run_id, "model": model, "version": version, "format": config["format"], "run": run,
               "qid": record["qid"], "answer": reply["text"], "system": built["system"], "prompt": built["prompt"],
               "prompt_hash": built["prompt_hash"], "seconds": round(time.time() - began, 1), "started_at": started,
               **{key: reply[key] for key in TOKEN_FIELDS}, "truncated": reply["finish_reason"] == "length"}
        answers = folder / "answers.csv"
        pd.DataFrame([row]).to_csv(answers, mode="a", header=not answers.exists(), index=False)
        time.sleep(pause)
    return f"done - {len(jobs)} answers"

# ------------------------------------------------------------------ stages 8-9: score, scorecard (no API calls)
def scoring_rows(answers: pd.DataFrame, records: list[dict]) -> pd.DataFrame:
    """Each saved answer with its question data and sources from this run's evidence, and the prompt it was sent."""
    by_qid = {record["qid"]: record for record in records}
    question_data = pd.DataFrame([{"qid": qid, "source": record["source"], "label": record["label"],
                                   "question": record["question"], "action": record["action"],
                                   "expected": record["expected"] or "",
                                   "sources": [hit["text"] for hit in record["hits"]]}
                                  for qid, record in by_qid.items()])
    return answers.merge(question_data, on="qid", how="left")[evaluate.SCORING_COLUMNS]


def token_use(answers: pd.DataFrame) -> dict:
    """Truncated answers (option A: scored as usual, counted here) and answer tokens per model: min, median, max."""
    if "truncated" not in answers:
        return {}
    tokens = pd.to_numeric(answers.answer_tokens, errors="coerce")
    return {"truncated": int(answers.truncated.astype(str).eq("True").sum()),
            "answer_tokens": {model: {"min": int(used.min()), "median": float(used.median()), "max": int(used.max())}
                              for model, used in tokens.groupby(answers.model) if used.notna().any()}}


def score(folder: Path, config: dict, records: list[dict]) -> dict:
    """Scores every saved answer; writes scored.csv, scorecard.csv and tradeoff.csv. Returns a short summary."""
    answers = pd.read_csv(folder / "answers.csv", keep_default_na=False)
    scored = evaluate.score_run(scoring_rows(answers, records))
    scored.to_csv(folder / "scored.csv", index=False)
    cards = evaluate.scorecards(scored).round(1)
    if "truncated" in answers:
        cards["truncated"] = answers.truncated.astype(str).eq("True").groupby([answers.model, answers.version]).sum()
    cards.to_csv(folder / "scorecard.csv")
    summary = {"scored_answers": len(scored), **token_use(answers)}
    if "tradeoff" in config:
        settings = config["tradeoff"]
        table, selected = evaluate.trade_off(scored, settings["models"], settings["versions"],
                                             set(settings["retrieval_miss_qids"]), set(settings["accepted_corrections"]),
                                             settings["p_answerable"], settings["k_values"], settings["chosen_k"])
        table.to_csv(folder / "tradeoff.csv")
        summary["selected"] = list(selected)
    log(folder, f"scored {len(scored)} answers -> scorecard.csv"
                + (f" | truncated {summary['truncated']}" if "truncated" in summary else "")
                + (f" | selected {summary['selected']}" if "selected" in summary else ""))
    return summary

# ------------------------------------------------------------------ the command

def main(argv: list[str] | None = None, runs_dir: Path = RUNS) -> Path:
    parser = argparse.ArgumentParser(description="SwasthyaSetu runner (dry run unless --allow-api-calls)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="experiment file, e.g. config/experiments/e3.yaml (starts a new run)")
    source.add_argument("--resume", help="run_id of a run in runs/ to continue")
    parser.add_argument("--allow-api-calls", action="store_true", help="really call the models")
    args = parser.parse_args(argv)

    folder, manifest, config, records = reopen(runs_dir / args.resume) if args.resume else start(args.config, runs_dir)
    jobs = pending(records, config, saved_keys(folder))
    total = sum(len(pair_jobs) for pair_jobs in jobs.values())
    for (model, version), pair_jobs in jobs.items():
        log(folder, f"{model} | {version} | {config['format']}: {len(pair_jobs)} pending")
    log(folder, f"Total pending calls: {total}")
    entry = {"at": now(), "pending_before": total, "api_calls": args.allow_api_calls, "results": {}}

    if args.allow_api_calls:
        load_dotenv(ROOT / ".env", override=True)
        manifest["status"] = "running"
        save_manifest(folder, manifest)
        for (model, version), pair_jobs in jobs.items():
            began = time.time()
            result = answer_pair(folder, manifest["run_id"], config, model, version, pair_jobs)
            entry["results"][f"{model} | {version}"] = result
            log(folder, f"{model} | {version}: {result} ({time.time() - began:.0f} s)")
    else:
        log(folder, "No API calls made (add --allow-api-calls).")

    still = sum(len(pair_jobs) for pair_jobs in pending(records, config, saved_keys(folder)).values())
    manifest["progress"] |= {"answers_saved": len(saved_keys(folder)), "pending": still}
    manifest["progress"]["history"].append(entry)
    if (folder / "answers.csv").exists():
        manifest["scoring"] = score(folder, config, records)
    if args.allow_api_calls:
        manifest["status"] = "complete" if still == 0 else "stopped"
        manifest["finished_at"] = now()
    save_manifest(folder, manifest)
    return folder

if __name__ == "__main__":
    main()