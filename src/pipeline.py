"""SwasthyaSetu pipeline, stages 1-6 (local models only, no API calls).

    1    load      the document's question sets, labelled (notebook 15.1 + 15.2b)
    2-5  prepare   normalize -> retrieve + rerank -> gate -> evidence: one call to Corrective.prepare (tested, unchanged)
    6    prompt    exact system + user prompt for one prompt version and format (src/prompts.py)

Stages 7-9 (generate, score, scorecard) belong to the runner (src/run.py). Every record keeps all fields of the
stages before it; a failing stage writes {stage, qid, type, message} into "error" and the run goes on (rule R5).

Usage
-----
    from src.config import load
    from src import pipeline
    config = load()
    gate = pipeline.open_gate(config)
    records = [pipeline.prepare(question, gate) for question in pipeline.load_questions(config)]
    pipeline.prompt(records[0], "v3b_med", "f0")["prompt"]
"""
from __future__ import annotations

import hashlib
import re

import pandas as pd

from src import prompts
from src.config import path_of
from src.corrective import Corrective
from src.query import SpellFixer
from src.retriever import Retriever

TO_LLM = ("answer", "answer_with_note")         # actions whose hits are sent to the model
PREPARED_FIELDS = ("fixed", "changes", "first", "final", "signals", "action", "fixed_answer", "hits")
SET_C_LABELS = {"answer": "answerable", "refuse": "unanswerable"}
PREMISE_PREFIX = re.compile(r"^(No|Wrong premise)\s*-\s*")


# ------------------------------------------------------------------ stage 1: load
def read_table(path) -> pd.DataFrame:
    table = pd.read_excel(path) if path.suffix == ".xlsx" else pd.read_csv(path)
    table.columns = table.columns.str.strip()
    return table


def question(source, qid, label, text, expected) -> dict:
    return {"source": source, "qid": qid, "label": label, "question": text, "expected": expected, "error": None}


def load_questions(config: dict) -> list[dict]:
    """All questions in notebook 15.1's order (manual, set C, negatives), then an experiment's new set if it has one;
    false premises the guideline can correct are answerable, with the correction as the expected answer (15.2b);
    questions the experiment lists under 'removed' are left out."""
    sets = config["questions"]
    manual, set_c, negatives = (read_table(path_of(sets[name]["file"])) for name in ("manual", "set_C", "negatives"))
    corrections = {row.qid: PREMISE_PREFIX.sub("", row.reference_answer)
                   for row in set_c[set_c.qid.isin(sets["premise_qids"])].itertuples()}
    questions = (
        [question("manual", row.qid, sets["manual"]["label"], row.question, row.expected_answer) for row in manual.itertuples()]
        + [question("set_C", row.qid, SET_C_LABELS[row.expected], row.question,
                    row.reference_answer if row.expected == "answer" else "") for row in set_c.itertuples()]
        + [question("negatives", f"{sets['negatives']['qid_prefix']}{index:02d}", sets["negatives"]["label"], text, "")
           for index, text in enumerate(negatives.question)])
    if "new" in sets:
        questions += [question("new", row.qid, row.label, row.question,
                               row.expected_answer if row.label == "answerable" else "")
                      for row in read_table(path_of(sets["new"]["file"])).itertuples()]
    removed = set(sets.get("removed", []))
    return [{**item, "label": "answerable", "expected": corrections[item["qid"]]} if item["qid"] in corrections else item
            for item in questions if item["qid"] not in removed]

# ------------------------------------------------------------------ stages 2-5: prepare
def open_gate(config: dict) -> Corrective:
    """Retriever (BM25 + vectors + reranker) and the corrective gate, built from the document's files."""
    retriever = Retriever(path_of(config["files"]["chunks"]), path_of(config["index"]["vectordb"]),
                          collection=config["index"]["collection"])
    return Corrective(retriever, SpellFixer(retriever.chunks), refine=config["evidence"]["refine"])


def prepare(record: dict, gate: Corrective) -> dict:
    """The question record plus the fixed question, gate decision, signals and final hits (every hit field kept)."""
    try:
        prepared = gate.prepare(record["question"])
    except Exception as failure:              # rule R5: name the stage and question, keep going
        return {**record, "error": {"stage": "prepare", "qid": record["qid"],
                                    "type": type(failure).__name__, "message": str(failure)}}
    return {**record, **{field: prepared[field] for field in PREPARED_FIELDS}}


# ------------------------------------------------------------------ stage 6: prompt
def prompt_hash(system: str, user: str) -> str:
    """The same hash as E2's saved scoring input (C-003)."""
    return hashlib.sha256(f"{system}\n\n{user}".encode("utf-8")).hexdigest()[:12]


def prompt(record: dict, version: str, fmt: str) -> dict:
    """The exact prompt for one question, prompt version and format."""
    system, user = prompts.build(record, version, fmt)
    return {"qid": record["qid"], "prompt_version": version, "format": fmt,
            "system": system, "prompt": user, "prompt_hash": prompt_hash(system, user)}