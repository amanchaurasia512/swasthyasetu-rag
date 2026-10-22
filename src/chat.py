"""SwasthyaSetu chat (C-033): the chatbot on the same path the runner scores - config, front end, gate, prompt.

    user message
       |-- answering our clarifying question, or a follow-up ("and Bihar?")  -> standalone question (LLM, history only)
       |-- unknown words (config query.front_end)  -> English rewrite (src.rewrite, LLM)
       v
    pipeline.prepare (SpellFixer / rewrite -> retrieve -> gate)   -> reject / scoped: fixed reply, no LLM
       |-- config chat.clarify and a vague question with many different candidate answers?
       |       -> ask ONE clarifying question (LLM), wait
       v
    pipeline.prompt (config chat.version + chat.format, the prompt E3 scored) -> answer (LLM)
       -> saved to the session file and one row in the question log

History is used ONLY to rewrite the question. Answers always come from freshly retrieved PDF sources, never from
earlier answers (so one wrong answer is not repeated). The chat is the product, so a user's question is the
permission to call the model (experiments still call models only through the runner, rule R1). A failed call never
stops the chat (rule R5): the user gets a short message and the row says what failed.

Usage
-----
    from src.chat import open_chat
    session = open_chat()                                   # config/app.yaml
    session.ask("probation period kitna hota hai")["answer"]
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from src import pipeline, rewrite
from src.config import load, path_of
from src.corrective import Corrective
from src.llm import LLMError, chat_full
from src.paths import ROOT

APP = "config/app.yaml"
HISTORY_TURNS = 3          # turns shown to the follow-up rewriter
SHORT_QUESTION = 6         # words: short messages are checked as possible follow-ups
CLARIFY_SPREAD = 3.0       # hits within this rerank distance of the best one count as "equally good"
CLARIFY_UNITS = 3          # ... and if they come from >= 3 different sections, the question may be vague
FOLLOW_UP = re.compile(r"\b(it|its|this|that|these|those|they|them|their|same|above|also)\b|^(and|or|what about|how about)\b", re.I)
PRIVATE = re.compile(r"\S+@\S+|\+?\d[\d\s-]{6,}\d")    # e-mails and phone-like numbers never reach the log
LOG_COLUMNS = ["time", "session", "question", "follow_up", "front_rewrite", "grade", "action", "calls", "seconds", "error"]
UNAVAILABLE = "Sorry, the answer service is not reachable right now. Please try again in a minute."

REWRITE_SYSTEM = """You rewrite the user's latest message into ONE complete, standalone question about the
"{title}". Use the conversation only to fill in what the message refers to
(e.g. "it", "they", "and Bihar?"). If the message is already a complete question on a new topic, return it
unchanged. Do not answer. Output only the question."""


CLARIFY_SYSTEM = """You decide whether a question is too vague to answer from the numbered sources.
If the sources give DIFFERENT answers for different posts, states, programmes or situations and the question
does not say which one is meant, reply with exactly one line:
ASK: <one short clarifying question that lists up to 4 options taken from the sources>
Otherwise reply exactly: CLEAR"""


def open_chat(experiment: str = APP, session_id: str | None = None) -> "ChatSession":
    """A chat session on the settings of config/app.yaml (API keys from .env)."""
    load_dotenv(ROOT / ".env", override=True)
    config = load(experiment)
    return ChatSession(config, pipeline.open_gate(config), session_id)


class ChatSession:
    def __init__(self, config: dict, gate: Corrective, session_id: str | None = None):
        self.config, self.gate, self.settings = config, gate, config["chat"]
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = path_of(self.settings["sessions"]) / f"{self.session_id}.json"
        self.turns: list[dict] = []
        self.pending: dict | None = None          # the question we asked the user to clarify
        self.calls = 0                            # LLM calls in the current turn
        if self.path.exists():
            self.load()

    # ------------------------------------------------------------------ memory: save / load
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        state = {"session_id": self.session_id, "turns": self.turns, "pending": self.pending}
        self.path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self) -> None:
        state = json.loads(self.path.read_text(encoding="utf-8"))
        self.turns, self.pending = state["turns"], state["pending"]

    def history_text(self) -> str:
        return "\n".join(f"User: {turn['user']}\nAssistant: {turn['answer'][:300]}" for turn in self.turns[-HISTORY_TURNS:])

    # ------------------------------------------------------------------ one LLM call, counted
    def llm(self, system: str, user: str) -> str:
        self.calls += 1
        llm = self.config["llm"]
        return chat_full(self.settings["model"], system, user, timeout=llm["timeout"], tries=llm["tries"])["text"]

    # ------------------------------------------------------------------ memory: follow-up rewriting
    def is_follow_up(self, message: str) -> bool:
        return bool(self.turns) and (len(message.split()) <= SHORT_QUESTION or bool(FOLLOW_UP.search(message)))

    def standalone(self, message: str) -> str:
        user = f"Conversation so far:\n{self.history_text()}\n\nLatest message: {message}\n\nStandalone question:"
        reply = self.llm(REWRITE_SYSTEM.format(title=self.config["title"]), user).strip()
        rewritten = reply.splitlines()[0].strip().strip('"') if reply else ""
        return rewritten if 3 <= len(rewritten) <= 4 * len(message) + 200 else message
    # ------------------------------------------------------------------ clarification
    @staticmethod
    def looks_vague(question: str, record: dict) -> bool:
        close = [hit for hit in record["hits"] if hit["score"] >= record["signals"]["best"] - CLARIFY_SPREAD]
        units = {hit.get("unit", hit["chunk_id"].split("-")[0]) for hit in close}
        return (record["final"] == "P" or len(question.split()) <= SHORT_QUESTION) and len(units) >= CLARIFY_UNITS

    def clarifying_question(self, prompt: str) -> str:
        found = re.search(r"ASK:\s*(.+)", self.llm(CLARIFY_SYSTEM, prompt))
        return found.group(1).strip() if found else ""

    # ------------------------------------------------------------------ question log
    def log(self, turn: dict) -> None:
        path = path_of(self.settings["log"])
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {"time": turn["time"], "session": self.session_id, "question": PRIVATE.sub("[hidden]", turn["user"]),
               "follow_up": turn["follow_up"], "front_rewrite": turn["front_rewrite"], "grade": turn["grade"],
               "action": turn["action"], "calls": turn["calls"], "seconds": turn["seconds"], "error": turn["error"]}
        pd.DataFrame([row], columns=LOG_COLUMNS).to_csv(path, mode="a", header=not path.exists(), index=False)

    # ------------------------------------------------------------------ one chat turn
    def ask(self, message: str) -> dict:
        started, self.calls, error, question = time.time(), 0, "", message
        clarified, follow_up = self.pending is not None, self.pending is not None or self.is_follow_up(message)
        record, asked, answer, action = {"final": "", "hits": []}, None, UNAVAILABLE, "error"
        try:
            question = self.standalone(message) if follow_up else message
            self.pending = None
            if rewrite.needed(question, self.config["query"]["front_end"], self.gate.SP):
                self.calls += 1
                asked = rewrite.ask(question, self.config)
            record = pipeline.prepare({"qid": f"{self.session_id}-{len(self.turns) + 1}", "question": question,
                                       "error": None}, self.gate, asked)
            if record.get("error"):
                raise RuntimeError(record["error"]["message"])
            answer, action = record["fixed_answer"], record["action"]
            if action in pipeline.TO_LLM:
                built = pipeline.prompt(record, self.settings["version"], self.settings["format"],
                                        self.config["gate"]["p_notes"], pipeline.document(self.config))
                clarify = self.clarifying_question(built["prompt"]) \
                    if self.settings["clarify"] and not clarified and self.looks_vague(question, record) else ""
                if clarify:
                    answer, action, self.pending = clarify, "clarify", {"question": question, "asked": clarify}
                else:
                    answer = self.llm(built["system"], built["prompt"])
        except (LLMError, RuntimeError) as failure:      # rule R5: the user gets a message, the row says what failed
            answer, action, error = UNAVAILABLE, "error", str(failure)[:200]
        turn = {"time": datetime.now().isoformat(timespec="seconds"), "user": message, "question": question,
                "follow_up": follow_up, "front_rewrite": bool(asked and asked["text"]), "action": action,
                "grade": record.get("final", ""), "answer": answer, "calls": self.calls,
                "seconds": round(time.time() - started, 1), "error": error or (asked or {}).get("error", ""),
                "sources": [{"n": number, "chunk_id": hit["chunk_id"], "pdf_pages": hit["pdf_pages"]}
                            for number, hit in enumerate(record["hits"], 1)] if action in pipeline.TO_LLM else []}
        self.turns.append(turn)
        self.save()
        self.log(turn)
        return turn