"""Diagnosis (C-017): a tiny made-up run - 3 chunks, 10 questions, each built to fail at one stage - must get exactly
the expected lost_at. It uses no NHM data, so it stays valid for any document.
Run from the project root:   python -m pytest tests/test_diagnose.py -q      (no models, no API calls; seconds)"""
import json
import shutil
import pandas as pd
import pytest

from src import diagnose

PROBATION = "Probation lasts 90 days for new staff."
CHUNKS = [
    {"chunk_id": "C1", "path": ["Chapter 1", "1.1 Leave"], "printed_pages": ["4"], "pdf_pages": [24],
     "text": "Casual leave is 12 days in a year for every employee."},
    {"chunk_id": "C2", "path": ["Chapter 2", "2.1 Probation"], "printed_pages": ["7"], "pdf_pages": [27],
     "text": "The probation period lasts 90 days for all new staff, extendable once."},
    {"chunk_id": "C3", "path": ["Chapter 3", "3.1 Travel"], "printed_pages": ["9"], "pdf_pages": [29],
     "text": "Travel by rail is allowed for field visits."},
]
EXPECTED_STAGE = {
    "Q-key": "key", "Q-index": "index", "Q-retrieval": "retrieval", "Q-gate": "gate", "Q-context": "context",
    "Q-refused": "llm_refused", "Q-incomplete": "llm_incomplete", "Q-ok": "ok", "Q-over": "over_answer", "Q-blocked": "ok",
}


def hit(chunk, text=None):
    return {**{key: chunk[key] for key in ("chunk_id", "path", "printed_pages", "pdf_pages")},
            "path": " > ".join(chunk["path"]), "text": text or chunk["text"]}


def record(qid, label, expected, action, hits):
    return {"qid": qid, "source": "test", "label": label, "question": qid, "expected": expected, "fixed": qid,
            "changes": [], "final": "S", "action": action, "signals": {"best": 1.0, "hard_out": []},
            "fixed_answer": "", "hits": hits}


@pytest.fixture(scope="module")
def runs_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    chunks_file = root / "chunks.jsonl"
    chunks_file.write_text("".join(json.dumps(chunk) + "\n" for chunk in CHUNKS), encoding="utf-8")
    folder = root / "t-20261017-1200"
    folder.mkdir()
    manifest = {"run_id": folder.name, "experiment": "t", "status": "complete", "started_at": "2026-10-17T12:00:00",
                "code": {"git_commit": "abc1234", "git_dirty": False},
                "config": {"merged": {"files": {"chunks": str(chunks_file)}}},
                "plan": {"questions": 10, "planned_calls": 12}, "progress": {"answers_saved": 12}, "scoring": {"truncated": 0}}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    c1, c2, c3 = CHUNKS
    records = [
        record("Q-key", "answerable", "", "answer", [hit(c2)]),
        record("Q-index", "answerable", "Pension is 75 percent of the last salary.", "answer", [hit(c1)]),
        record("Q-retrieval", "answerable", PROBATION, "answer", [hit(c1), hit(c3)]),
        record("Q-gate", "answerable", PROBATION, "reject", [hit(c2)]),
        record("Q-context", "answerable", PROBATION, "answer", [hit(c2, "… extendable once.")]),
        record("Q-refused", "answerable", PROBATION, "answer", [hit(c2)]),
        record("Q-incomplete", "answerable", PROBATION, "answer", [hit(c2)]),
        record("Q-ok", "answerable", PROBATION, "answer_with_note", [hit(c2)]),
        record("Q-over", "unanswerable", "", "answer", [hit(c3)]),
        record("Q-blocked", "unanswerable", "", "reject", [hit(c3)]),
    ]
    (folder / "evidence.jsonl").write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
    answers = {"Q-key": ("Probation is 90 days.", False, None), "Q-index": ("Not found in the guideline.", True, None),
               "Q-retrieval": ("Not found in the guideline.", True, None), "Q-context": ("It can be extended.", False, 0.0),
               "Q-refused": ("Not found in the guideline.", True, None), "Q-incomplete": ("It can be extended once.", False, 0.0),
               "Q-ok": ("Probation lasts 90 days for new staff [1].", False, 1.0), "Q-over": ("Rail travel is allowed.", False, None)}
    scored = pd.DataFrame([{"qid": qid, "run": run, "answer": text, "refused": refused, "correctness": correct}
                           for qid, (text, refused, correct) in answers.items() for run in (1, 2)])
    scored.to_csv(folder / "scored.csv", index=False)
    diagnose.main([folder.name], runs_dir=root)
    return root


@pytest.fixture(scope="module")
def table(runs_dir):
    return pd.read_csv(runs_dir / "t-20261017-1200" / "report.csv")


def test_every_question_lost_at_the_built_in_stage(table):
    assert dict(zip(table.qid, table.lost_at)) == EXPECTED_STAGE


def test_fact_counts_follow_the_stages(table):
    counts = table.set_index("qid")[["facts_in_index", "facts_in_hits", "facts_in_prompt"]]
    assert counts.loc["Q-retrieval"].tolist() == [1, 0, 0]
    assert counts.loc["Q-context"].tolist() == [1, 1, 0]
    assert counts.loc["Q-ok"].tolist() == [1, 1, 1]


def test_page_facts_are_read_from_metadata():
    page_fact = {"label": "answerable", "expected": "The probation section is on PDF page 27.", "qid": "Q", "source": "t",
                 "question": "Q", "action": "answer", "hits": [hit(CHUNKS[1])]}
    row = diagnose.question_row(page_fact, pd.DataFrame(columns=["run", "answer", "refused", "correctness"]),
                                {chunk["chunk_id"]: diagnose.labelled(chunk) for chunk in CHUNKS})
    assert row["facts_in_prompt"] == 1


def test_index_has_one_row_per_run_with_stage_counts(runs_dir):
    index = pd.read_csv(runs_dir / "index.csv")
    assert index.run_id.tolist() == ["t-20261017-1200"]
    assert index.loc[0, "lost_at_ok"] == 2 and index.loc[0, "lost_at_gate"] == 1 and index.loc[0, "date"] == "2026-10-17"

def test_a_run_with_several_models_is_diagnosed_one_model_at_a_time(runs_dir):
    """C-036: answers of two models are never mixed as runs of one; each model gets its own report."""
    folder = runs_dir / "t2-20261019-1200"
    shutil.copytree(runs_dir / "t-20261017-1200", folder)
    scored = pd.read_csv(folder / "scored.csv")
    both = pd.concat([scored.assign(model="groq:a/b"), scored.assign(model="sarvam:c", refused=True)])
    both.to_csv(folder / "scored.csv", index=False)
    with pytest.raises(SystemExit, match="2 models"):
        diagnose.main([folder.name], runs_dir=runs_dir)
    first = diagnose.main([folder.name, "--model", "groq:a/b"], runs_dir=runs_dir)
    assert dict(zip(first.qid, first.lost_at)) == EXPECTED_STAGE and (folder / "report_groq_a_b.csv").exists()
    second = diagnose.main([folder.name, "--model", "sarvam:c"], runs_dir=runs_dir)
    assert dict(zip(second.qid, second.lost_at))["Q-ok"] == "llm_refused"