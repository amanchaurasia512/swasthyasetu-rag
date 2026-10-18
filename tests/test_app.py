"""Chat page (C-034): the Streamlit page answers through ChatSession and shows the answer, its sources and what the
system did; out-of-scope questions show the fixed reply. Runs the page headless (streamlit.testing), no browser and
no API calls - the LLM is replaced by a stand-in.   Run:   python -m pytest tests/test_app.py -q"""
import pytest
from streamlit.testing.v1 import AppTest

from src import chat, rewrite
from src.paths import ROOT

PAGE = str(ROOT / "app" / "streamlit_app.py")


def status(page) -> str:
    """The caption under the last answer (what the system did)."""
    return [caption.value for caption in page.main.caption if "model call" in caption.value][-1]


@pytest.fixture
def page(tmp_path, monkeypatch):
    monkeypatch.setenv("SWASTHYASETU_CHAT_DIR", str(tmp_path))
    monkeypatch.setattr(chat, "chat_full", lambda model, system, user, **options: {"text": "ANSWER"})
    monkeypatch.setattr(rewrite, "chat_full", lambda model, system, user, **options: {"text": user})
    return AppTest.from_file(PAGE, default_timeout=120).run()


def test_page_opens_empty(page):
    assert not page.exception and page.header[0].value == "Ask the NHM HR guideline" and not page.chat_message


def test_question_shows_answer_sources_and_caption(page):
    page.chat_input[0].set_value("How long is the probation period?").run()
    texts = [block.value for block in page.markdown]
    assert "ANSWER" in texts and any(text.startswith("**[1]** PDF page") for text in texts)
    assert "answered from the guideline" in status(page) and "1 model call(s)" in status(page)


def test_out_of_scope_question_shows_the_fixed_reply(page):
    page.chat_input[0].set_value("What is the share price of Tata Motors today?").run()
    assert "not covered by the guideline" in status(page) and "0 model call(s)" in status(page)