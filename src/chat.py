"""SwasthyaSetu chat: short-term memory (follow-up rewriting) + one clarifying question, on top of the corrective layer.

    user message
       |-- answering our clarifying question?  -> merge it with the original question (LLM rewrite)
       |-- a follow-up ("and Bihar?", "can it be extended?") -> rewrite into a standalone question (LLM)
       v
    Corrective.prepare(standalone question)   -> reject / scoped (fixed reply, no LLM)
       |                                         answer / answer_with_note
       |-- vague question with many different candidate answers? -> ask ONE clarifying question (LLM), wait
       v
    answer from the sources (LLM)  -> saved to the session file (data/chat_sessions/<id>.json)

History is used ONLY to rewrite the question. Answers always come from freshly retrieved PDF sources,
never from earlier answers (so one wrong answer is not repeated).

Usage
-----
    from src.chat import ChatSession
    session = ChatSession(CR, model="qwen2.5:7b")          # CR = Corrective(R, SP)
    session.ask("What incentive do they get per C-section?")["answer"]   # -> a clarifying question
    session.ask("Staff nurses in the UP buddy model")["answer"]          # -> the answer
    session.ask("and OT technicians?")["answer"]                         # -> follow-up, rewritten first
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import src.answer as answer_mod
from src.llm import chat as llm_chat

HISTORY_TURNS = 3          # turns shown to the rewriter
SHORT_QUESTION = 6         # words: short messages are checked as possible follow-ups
CLARIFY_SPREAD = 3.0       # hits within this rerank distance of the best one count as "equally good"
CLARIFY_UNITS = 3          # ... and if they come from >= 3 different sections, the question may be vague
FOLLOW_UP = re.compile(r"\b(it|its|this|that|these|those|they|them|their|same|above|also)\b|^(and|or|what about|how about)\b", re.I)

REWRITE_SYSTEM = """You rewrite the user's latest message into ONE complete, standalone question about the
NHM Human Resources for Health guideline. Use the conversation only to fill in what the message refers to
(e.g. "it", "they", "and Bihar?"). If the message is already a complete question on a new topic, return it
unchanged. Do not answer. Output only the question."""

CLARIFY_SYSTEM = """You decide whether a question is too vague to answer from the numbered sources.
If the sources give DIFFERENT answers for different posts, states, programmes or situations and the question
does not say which one is meant, reply with exactly one line:
ASK: <one short clarifying question that lists up to 4 options taken from the sources>
Otherwise reply exactly: CLEAR"""


class ChatSession:
    def __init__(self, corrective, model: str, session_dir: str | Path = "data/chat_sessions",
                 session_id: str | None = None):
        self.corrective = corrective
        self.model = model
        self.session_id = session_id or time.strftime("%Y%m%d_%H%M%S")
        self.path = Path(session_dir) / f"{self.session_id}.json"
        self.turns: list[dict] = []
        self.pending: dict | None = None          # the question we asked the user to clarify
        if self.path.exists():
            self.load()

    # ------------------------------------------------------------------ memory: save / load
    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        state = {"session_id": self.session_id, "model": self.model, "turns": self.turns, "pending": self.pending}
        self.path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self):
        state = json.loads(self.path.read_text(encoding="utf-8"))
        self.turns, self.pending = state["turns"], state["pending"]

    def history_text(self) -> str:
        return "\n".join(f"User: {t['user']}\nAssistant: {t['answer'][:300]}" for t in self.turns[-HISTORY_TURNS:])

    # ------------------------------------------------------------------ memory: follow-up rewriting
    def is_follow_up(self, message: str) -> bool:
        return bool(self.turns) and (len(message.split()) <= SHORT_QUESTION or bool(FOLLOW_UP.search(message)))

    def rewrite(self, message: str) -> str:
        user = f"Conversation so far:\n{self.history_text()}\n\nLatest message: {message}\n\nStandalone question:"
        reply = llm_chat(self.model, REWRITE_SYSTEM, user).strip()
        rewritten = reply.splitlines()[0].strip().strip('"') if reply else ""
        return rewritten if 3 <= len(rewritten) <= 4 * len(message) + 200 else message

    # ------------------------------------------------------------------ clarification
    @staticmethod
    def looks_vague(question: str, prep: dict) -> bool:
        close = [h for h in prep["hits"] if h["score"] >= prep["best"] - CLARIFY_SPREAD]
        units = {h.get("unit", h["chunk_id"].split("-")[0]) for h in close}
        return (prep["final"] == "P" or len(question.split()) <= SHORT_QUESTION) and len(units) >= CLARIFY_UNITS

    def clarifying_question(self, prep: dict) -> str:
        reply = llm_chat(self.model, CLARIFY_SYSTEM, prep["prompt"])
        found = re.search(r"ASK:\s*(.+)", reply)
        return found.group(1).strip() if found else ""

    # ------------------------------------------------------------------ one chat turn
    def ask(self, message: str) -> dict:
        started = time.time()
        clarified = self.pending is not None
        if clarified or self.is_follow_up(message):
            question = self.rewrite(message)
        else:
            question = message
        self.pending = None

        prep = self.corrective.prepare(question)
        clarify = ""
        if prep["prompt"] and not clarified and self.looks_vague(question, prep):
            clarify = self.clarifying_question(prep)
        if clarify:
            answer, action = clarify, "clarify"
            self.pending = {"question": question, "asked": clarify}
        elif prep["prompt"]:
            answer, action = llm_chat(self.model, answer_mod.SYSTEM_PROMPT, prep["prompt"]), prep["action"]
        else:
            answer, action = prep["fixed_answer"], prep["action"]

        turn = {"user": message, "question": question, "rewritten": question != message, "action": action,
                "grade": prep["final"], "answer": answer, "seconds": round(time.time() - started, 1),
                "sources": [{"n": n, "chunk_id": h["chunk_id"], "pdf_pages": h["pdf_pages"]}
                            for n, h in enumerate(prep["hits"], 1)] if action not in ("reject", "scoped") else []}
        self.turns.append(turn)
        self.save()
        return turn
