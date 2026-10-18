"""SwasthyaSetu chat page (C-034): a Streamlit page on src.chat.ChatSession, so the page answers exactly the way the
scored pipeline does (config/app.yaml). Dark theme: .streamlit/config.toml.

    streamlit run app/streamlit_app.py          from the project root; opens http://localhost:8501

The retriever and gate load once per server (st.cache_resource); each browser tab gets its own conversation.
Set SWASTHYASETU_CHAT_DIR to keep sessions and the question log somewhere other than config chat.sessions / chat.log.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from src import pipeline
from src.chat import APP, ChatSession
from src.config import load
from src.paths import ROOT

ACTIONS = {"answer": "answered from the guideline", "answer_with_note": "answered from the guideline",
           "reject": "not covered by the guideline", "scoped": "outside the guideline's scope",
           "clarify": "asked a clarifying question", "error": "service not reachable"}


@st.cache_resource(show_spinner="Loading the guideline index ...")
def parts() -> tuple[dict, object]:
    """(config, gate), built once per server."""
    load_dotenv(ROOT / ".env", override=True)
    config = load(APP)
    folder = os.environ.get("SWASTHYASETU_CHAT_DIR")
    if folder:
        config["chat"] = {**config["chat"], "sessions": f"{folder}/sessions", "log": f"{folder}/questions.csv"}
    return config, pipeline.open_gate(config)


def new_session() -> ChatSession:
    config, gate = parts()
    return ChatSession(config, gate, session_id=f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}")


def show(turn: dict) -> None:
    """One question and its answer, with the sources and what the system did."""
    with st.chat_message("user"):
        st.markdown(turn["user"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        if turn["sources"]:
            with st.expander(f"Sources ({len(turn['sources'])})"):
                for source in turn["sources"]:
                    pages = ", ".join(str(page) for page in source["pdf_pages"])
                    st.markdown(f"**[{source['n']}]** PDF page {pages} - `{source['chunk_id']}`")
        searched = f' | searched as: "{turn["question"]}"' if turn["question"] != turn["user"] else ""
        st.caption(f"{ACTIONS.get(turn['action'], turn['action'])} | grade {turn['grade'] or '-'} | "
                   f"{turn['calls']} model call(s) | {turn['seconds']} s{searched}")


st.set_page_config(page_title="SwasthyaSetu", page_icon="🩺", layout="centered")
if "chat" not in st.session_state:
    st.session_state.chat = new_session()

with st.sidebar:
    st.title("SwasthyaSetu")
    st.write("Questions on the NHM guideline *Human Resources for Health* (2022), in English or Hinglish. "
             "Answers come only from the guideline, with page numbers.")
    if st.button("New conversation", width="stretch"):
        st.session_state.chat = new_session()
    st.caption("Known limits: names typed differently from the PDF may be missed; some letters and tables are not "
               "searchable yet.")

st.header("Ask the NHM HR guideline")
for turn in st.session_state.chat.turns:
    show(turn)
if message := st.chat_input("Type your question ..."):
    with st.spinner("Searching the guideline ..."):
        turn = st.session_state.chat.ask(message)
    show(turn)