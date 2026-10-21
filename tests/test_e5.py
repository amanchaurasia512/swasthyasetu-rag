"""E5 = the chatbot (C-035): the final run must score exactly what users get, so config/experiments/e5.yaml and
config/app.yaml must agree on the front end, the gate and the model / prompt / format, and E5 must use E3's questions.
Run from the project root:   python -m pytest tests/test_e5.py -q        (no models, no API calls; seconds)"""
from src.config import load

E3, E5, APP = load("config/experiments/e3.yaml"), load("config/experiments/e5.yaml"), load("config/app.yaml")
E6 = load("config/experiments/e6.yaml")


def test_e5_uses_the_chatbot_settings():
    assert (E5["query"], E5["gate"]) == (APP["query"], APP["gate"])
    assert E5["plan"] == [{"versions": [APP["chat"]["version"]], "models": [APP["chat"]["model"]]}]
    assert E5["format"] == APP["chat"]["format"]


def test_e5_uses_the_e3_questions_and_runs():
    assert (E5["question_set"], E5["questions"], E5["runs"]) == (E3["question_set"], E3["questions"], E3["runs"])

def test_e6_differs_from_e5_only_in_its_models():
    """C-036: the model comparison keeps every E5 setting; E5's model is one of the eight."""
    same = lambda config: {key: value for key, value in config.items() if key not in ("experiment", "purpose", "plan")}
    assert same(E6) == same(E5)
    assert [group["versions"] for group in E6["plan"]] == [["v3b_med"]] and len(E6["plan"][0]["models"]) == 8
    assert E5["plan"][0]["models"][0] in E6["plan"][0]["models"]