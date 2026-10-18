"""Shared fixtures, built once per test session: the retriever and gate, and every question of the document
prepared once (stages 1-5). tests/test_retrieval.py and tests/test_pipeline.py both read them, so the 173
questions go through retrieval one time instead of two (local models, no API calls)."""
import pytest
from src import pipeline
from src.config import load


@pytest.fixture(scope="session")
def config():
    return load()


@pytest.fixture(scope="session")
def gate(config):
    return pipeline.open_gate(config)


@pytest.fixture(scope="session")
def prepared(config, gate):
    """{qid: record} for every question, as pipeline.prepare returns it."""
    return {question["qid"]: pipeline.prepare(question, gate) for question in pipeline.load_questions(config)}