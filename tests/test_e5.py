"""E5 = the chatbot (C-035): the final run must score exactly what users get, so config/experiments/e5.yaml and
config/app.yaml must agree on the front end, the gate and the model / prompt / format, and E5 must use E3's questions.
Run from the project root:   python -m pytest tests/test_e5.py -q        (no models, no API calls; seconds)"""
from src.config import load

E3, E5, APP = load("config/experiments/e3.yaml"), load("config/experiments/e5.yaml"), load("config/app.yaml")


def test_e5_uses_the_chatbot_settings():
    assert (E5["query"], E5["gate"]) == (APP["query"], APP["gate"])
    assert E5["plan"] == [{"versions": [APP["chat"]["version"]], "models": [APP["chat"]["model"]]}]
    assert E5["format"] == APP["chat"]["format"]


def test_e5_uses_the_e3_questions_and_runs():
    assert (E5["question_set"], E5["questions"], E5["runs"]) == (E3["question_set"], E3["questions"], E3["runs"])