"""Blocks + chunker regression (Phase 2, step 4b): sections + letters -> blocks -> chunks must give chunks.jsonl exactly.
Run from the project root:   python -m pytest tests/test_chunker.py -q        (seconds, no models)"""
from collections import Counter

import pytest

from src import blocks as blocks_module
from src import chunker
from src.extract import nhm
from src.paths import CHUNKS, LETTERS_JSONL, PAGES_FINAL, SECTIONS


@pytest.fixture(scope="module")
def pages():
    return chunker.read_jsonl(PAGES_FINAL)


@pytest.fixture(scope="module")
def blocks(pages):
    return nhm.build_blocks(chunker.read_jsonl(SECTIONS), chunker.read_jsonl(LETTERS_JSONL), pages)


@pytest.fixture(scope="module")
def built(blocks, pages):
    return chunker.build_chunks(blocks, pages)


def test_blocks_hold_all_invariants(blocks, pages):
    assert blocks_module.validate(blocks, pages) == []
    assert Counter(block["unit"] for block in blocks) == {"section": 105, "glossary": 1, "letter": 21}


def test_counts_by_type(built):
    assert len(built) == 427
    assert Counter(chunk["doc_type"] for chunk in built) == {"section": 314, "table": 43, "letter": 68, "letter_stub": 2}


def test_identical_to_saved_chunks(built):
    saved = chunker.read_jsonl(CHUNKS)
    different = [new["chunk_id"] for new, old in zip(built, saved) if new != old]
    assert len(built) == len(saved) and not different, f"{len(different)} chunks differ, e.g. {different[:5]}"


def test_identical_after_blocks_file_round_trip(blocks, pages, tmp_path):
    path = tmp_path / "blocks.jsonl"
    blocks_module.write_blocks(path, blocks)
    assert chunker.build_chunks(blocks_module.read_blocks(path), pages) == chunker.read_jsonl(CHUNKS)