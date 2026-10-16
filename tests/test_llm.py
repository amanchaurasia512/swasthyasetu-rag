"""Token use and cut-off answers (Phase 4, fix 2, C-015): llm.chat_full reads usage and finish_reason from a reply;
llm.chat still returns only the text. Fake replies, no network.
Run from the project root:   python -m pytest tests/test_llm.py -q      (under a second)"""
import pytest

from src import llm


class FakeResponse:
    """The parts of a requests.Response that llm.py reads."""

    def __init__(self, data):
        self.data, self.status_code, self.text = data, 200, str(data)
        self.headers = {"content-type": "application/json"}

    def json(self):
        return self.data

    def raise_for_status(self):
        return None


def openai_style(text, finish_reason, usage):
    return {"choices": [{"message": {"content": text}, "finish_reason": finish_reason}], **usage}


FINISHED = openai_style("Probation is 6 months [2].", "stop",
                        {"usage": {"prompt_tokens": 950, "completion_tokens": 40,
                                   "completion_tokens_details": {"reasoning_tokens": 25}}})
CUT_OFF = openai_style("1. Stage one [1]\n2. Stage", "length", {"usage": {"prompt_tokens": 1200, "completion_tokens": 2048}})
NO_USAGE = openai_style("Not found in the guideline.", "stop", {})
OLLAMA = {"message": {"content": "<think>short</think>Probation is 6 months [2]."}, "done_reason": "stop",
          "prompt_eval_count": 950, "eval_count": 18}


def test_finished_reply():
    assert llm.api_reply(FakeResponse(FINISHED)) == {"text": "Probation is 6 months [2].", "prompt_tokens": 950,
                                                     "answer_tokens": 40, "reasoning_tokens": 25, "finish_reason": "stop"}


def test_cut_off_reply():
    reply = llm.api_reply(FakeResponse(CUT_OFF))
    assert (reply["finish_reason"], reply["answer_tokens"], reply["reasoning_tokens"]) == ("length", 2048, None)


def test_reply_without_usage_gives_none():
    reply = llm.api_reply(FakeResponse(NO_USAGE))
    assert (reply["prompt_tokens"], reply["answer_tokens"], reply["reasoning_tokens"]) == (None, None, None)


def test_ollama_reply():
    assert llm.ollama_reply(FakeResponse(OLLAMA))["answer_tokens"] == 18


@pytest.fixture
def fake_api(monkeypatch):
    """llm._api / llm._ollama answer from the fakes instead of the network."""
    monkeypatch.setattr(llm, "_api", lambda provider, name, messages, timeout: (FakeResponse(CUT_OFF), llm.api_text))
    monkeypatch.setattr(llm, "_ollama", lambda name, messages, timeout: (FakeResponse(OLLAMA), None))


def test_chat_full_returns_text_and_token_use(fake_api):
    reply = llm.chat_full("groq:openai/gpt-oss-20b", "system", "user")
    assert reply["text"] == "1. Stage one [1]\n2. Stage" and reply["finish_reason"] == "length"


def test_chat_still_returns_only_the_text(fake_api):
    assert llm.chat("groq:openai/gpt-oss-20b", "system", "user") == "1. Stage one [1]\n2. Stage"
    assert llm.chat("qwen2.5:7b", "system", "user") == "Probation is 6 months [2]."      # <think> removed, as before