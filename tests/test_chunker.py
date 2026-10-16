"""Chunker regression (Phase 2, step 4a): src/chunker.py must rebuild chunks.jsonl exactly from sections + letters.
Run from the project root:   python -m pytest tests/test_chunker.py -q        (seconds, no models)"""
from collections import Counter

import pytest

from src import chunker
from src.paths import CHUNKS, LETTERS_JSONL, PAGES_FINAL, SECTIONS


@pytest.fixture(scope="module")
def built():
    return chunker.build_chunks(chunker.read_jsonl(SECTIONS), chunker.read_jsonl(LETTERS_JSONL),
                                chunker.read_jsonl(PAGES_FINAL))


def test_counts_by_type(built):
    assert len(built) == 427
    assert Counter(chunk["doc_type"] for chunk in built) == {"section": 314, "table": 43, "letter": 68, "letter_stub": 2}


def test_identical_to_saved_chunks(built):
    saved = chunker.read_jsonl(CHUNKS)
    different = [new["chunk_id"] for new, old in zip(built, saved) if new != old]
    assert len(built) == len(saved) and not different, f"{len(different)} chunks differ, e.g. {different[:5]}"