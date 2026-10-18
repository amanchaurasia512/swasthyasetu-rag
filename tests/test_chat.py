"""Chat on the scored path (C-033): the answer uses the prompt E3 scored (config chat.version + chat.format), the
rewrite is asked only for unknown words, out-of-scope questions get the fixed reply with no LLM call, a failed call
gives a short message and the chat goes on, and every question leaves one row in the question log.
No API calls - the LLM is replaced by a stand-in.   Run:   python -m pytest tests/test_chat.py -q"""
import pandas as pd
import pytest

from src import chat, pipeline, prompts, rewrite
from src.config import load
from src.llm import LLMError


@pytest.fixture
def session(gate, tmp_path):
    """A chat on config/app.yaml with the shared retriever; sessions and log in a temporary folder."""
    config = load(chat.APP)
    config["chat"] = {**config["chat"], "sessions": str(tmp_path / "sessions"), "log": str(tmp_path / "questions.csv")}
    return chat.ChatSession(config, gate, session_id="test")


@pytest.fixture
def calls(monkeypatch):
    """Every LLM call as (who, system, user); the stand-in answers 'ANSWER' and rewrites by echoing."""
    seen = []

    def answer(model, system, user, **options):
        seen.append(("chat", system, user))
        return {"text": "ANSWER"}

    def rewriter(model, system, user, **options):
        seen.append(("rewrite", system, user))
        return {"text": user}
    monkeypatch.setattr(chat, "chat_full", answer)
    monkeypatch.setattr(rewrite, "chat_full", rewriter)
    return seen


def test_english_question_gets_the_scored_prompt_and_no_rewrite(session, calls):
    turn = session.ask("How long is the probation period?")
    assert [who for who, system, user in calls] == ["chat"] and turn["answer"] == "ANSWER"
    record = pipeline.prepare({"qid": "x", "question": turn["question"], "error": None}, session.gate)
    built = pipeline.prompt(record, "v3b_med", "f0p", False)
    assert calls[0][1:] == (built["system"], built["prompt"]) and calls[0][1] == prompts.VERSIONS["v3b_med"][0]
    assert (turn["calls"], turn["front_rewrite"], turn["grade"]) == (1, False, "S") and turn["sources"]


def test_hinglish_question_is_rewritten_first(session, calls):
    turn = session.ask("probation period kitna hota hai")
    assert [who for who, system, user in calls][:1] == ["rewrite"] and turn["front_rewrite"]
    assert turn["calls"] == len(calls)


def test_out_of_scope_question_gets_the_fixed_reply_without_a_call(session, calls):
    turn = session.ask("What is the share price of Tata Motors today?")
    assert calls == [] and (turn["action"], turn["calls"], turn["sources"]) == ("reject", 0, [])


def test_failed_call_gives_a_message_and_the_chat_goes_on(session, calls, monkeypatch):
    def down(model, system, user, **options):
        raise LLMError("quota")
    monkeypatch.setattr(chat, "chat_full", down)
    turn = session.ask("How long is the probation period?")
    assert (turn["action"], turn["answer"], turn["error"]) == ("error", chat.UNAVAILABLE, "quota")
    monkeypatch.setattr(chat, "chat_full", lambda model, system, user, **options: {"text": "ANSWER"})
    assert session.ask("What is the probation period for a staff nurse?")["answer"] == "ANSWER"


def test_every_question_leaves_one_log_row_without_private_numbers(session, calls):
    session.ask("How long is the probation period? call me on 98765 43210")
    session.ask("What is the share price of Tata Motors today?")
    log = pd.read_csv(session.settings["log"])
    assert list(log.columns) == chat.LOG_COLUMNS and len(log) == 2
    assert "98765" not in log.question[0] and "[hidden]" in log.question[0]
    assert list(log.action) == [session.turns[0]["action"], "reject"]


def test_session_is_saved_and_loaded(session, calls):
    session.ask("How long is the probation period?")
    again = chat.ChatSession(session.config, session.gate, session_id="test")
    assert [turn["user"] for turn in again.turns] == ["How long is the probation period?"]


def test_clarify_switch_asks_once_and_merges_the_reply(session, calls, monkeypatch):
    """With config chat.clarify on and a question that looks vague, the chat asks back, then answers the merged one."""
    replies = iter(["ASK: For which post?", "probation period of a staff nurse", "ANSWER"])
    monkeypatch.setattr(chat, "chat_full", lambda model, system, user, **options: {"text": next(replies)})
    monkeypatch.setattr(chat.ChatSession, "looks_vague", staticmethod(lambda question, record: True))
    session.settings["clarify"] = True
    first = session.ask("What is the probation period?")
    assert (first["action"], first["answer"]) == ("clarify", "For which post?") and session.pending
    second = session.ask("staff nurse")
    assert second["follow_up"] and second["question"] == "probation period of a staff nurse"
    assert second["answer"] == "ANSWER" and session.pending is None