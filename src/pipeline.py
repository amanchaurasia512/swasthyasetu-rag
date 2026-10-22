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
    a document's experiment may have only some of these sets (C-044: NFHS-6 has only 'new'). False premises the
    guideline can correct are answerable, with the correction as the expected answer (15.2b); questions the
    experiment lists under 'removed' are left out."""
    sets = config["questions"]
    tables = {name: read_table(path_of(sets[name]["file"])) for name in ("manual", "set_C", "negatives", "new") if name in sets}
    questions = []
    if "manual" in tables:
        questions += [question("manual", row.qid, sets["manual"]["label"], row.question, row.expected_answer)
                      for row in tables["manual"].itertuples()]
    if "set_C" in tables:
        questions += [question("set_C", row.qid, SET_C_LABELS[row.expected], row.question,
                               row.reference_answer if row.expected == "answer" else "") for row in tables["set_C"].itertuples()]
    if "negatives" in tables:
        questions += [question("negatives", f"{sets['negatives']['qid_prefix']}{index:02d}", sets["negatives"]["label"], text, "")
                      for index, text in enumerate(tables["negatives"].question)]
    if "new" in tables:
        questions += [question("new", row.qid, row.label, row.question, row.expected_answer if row.label == "answerable" else "")
                      for row in tables["new"].itertuples()]
    set_c = tables.get("set_C")
    corrections = {} if set_c is None else {row.qid: PREMISE_PREFIX.sub("", row.reference_answer)
                                            for row in set_c[set_c.qid.isin(sets["premise_qids"])].itertuples()}
    removed = set(sets.get("removed", []))
    return [{**item, "label": "answerable", "expected": corrections[item["qid"]]} if item["qid"] in corrections else item
            for item in questions if item["qid"] not in removed]

# ------------------------------------------------------------------ stages 2-5: prepare
def open_gate(config: dict) -> Corrective:
    """Retriever (BM25 + vectors + reranker) and the corrective gate, built from the document's files."""
    retriever = Retriever(path_of(config["files"]["chunks"]), path_of(config["index"]["vectordb"]),
                          collection=config["index"]["collection"])
    switches = config["gate"]
    return Corrective(retriever, SpellFixer(retriever.chunks), refine=config["evidence"]["refine"],
                      m_action=switches["m_action"], p_notes=switches["p_notes"], re_retrieve=switches["re_retrieve"])


def prepare(record: dict, gate: Corrective, asked: dict | None = None) -> dict:
    """The question record plus the fixed question, gate decision, signals and final hits (every hit field kept).
    asked: the live rewrite (src.rewrite.ask, config query.front_end = rewrite). It is kept in the record, so the
    evidence shows what was searched; an empty or failed rewrite falls back to the SpellFixer (rule R5)."""
    try:
        prepared = gate.prepare(record["question"], asked["text"] if asked else None)
    except Exception as failure:              # rule R5: name the stage and question, keep going
        return {**record, "error": {"stage": "prepare", "qid": record["qid"],
                                    "type": type(failure).__name__, "message": str(failure)}}
    return {**record, **({"rewrite": asked} if asked else {}), **{field: prepared[field] for field in PREPARED_FIELDS}}


# ------------------------------------------------------------------ stage 6: prompt
def prompt_hash(system: str, user: str) -> str:
    """The same hash as E2's saved scoring input (C-003)."""
    return hashlib.sha256(f"{system}\n\n{user}".encode("utf-8")).hexdigest()[:12]

def document(config: dict) -> dict:
    """How the prompts name this run's document (C-040): title and not-found reply from config/docs/<doc>.yaml,
    plus its prompt words (prompt.word, prompt.outside)."""
    return {"title": config["title"], "not_found": config["not_found"], **config["prompt"]}


def prompt(record: dict, version: str, fmt: str, notes: bool = True, doc: dict | None = None) -> dict:
    """The exact prompt for one question, prompt version and format (notes: config gate.p_notes; doc: document(config))."""
    system, user = prompts.build(record, version, fmt, notes, doc)
    return {"qid": record["qid"], "prompt_version": version, "format": fmt,
            "system": system, "prompt": user, "prompt_hash": prompt_hash(system, user)}