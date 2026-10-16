"""Format f0p (Phase 4, fix 1, C-014): the f0 prompt with both page numbers in every source header.
f0 itself must stay unchanged (the other test files prove that); f0p must differ from f0 only in its headers.
Run from the project root:   python -m pytest tests/test_format_f0p.py -q    (local models, no API calls; ~6 min alone)"""
import re

from src import pipeline, prompts

F0P_HEADER = re.compile(r"\(printed page ([^;\n]*); PDF page ([\d, ]+)\)")
NHM09_FIRST = ("[1] Chapter 6: Appointment and Onboarding > 6.1 Joining Formalities > 6.1.1 Appointment / Offer Letter "
               "(printed page 17; PDF page 37)")


def to_llm(prepared):
    return [record for record in prepared.values() if record["action"] in pipeline.TO_LLM]


def test_f0p_is_f0_plus_pdf_pages(prepared):
    """Removing 'printed ' and '; PDF page ...' from every f0p header gives back the f0 prompt exactly."""
    records = to_llm(prepared)
    wrong = [record["qid"] for record in records
             if F0P_HEADER.sub(r"(page \1)", prompts.build(record, "v3b_med", "f0p")[1])
             != prompts.build(record, "v3b_med", "f0")[1]]
    assert len(records) == 104
    assert not wrong, f"{len(wrong)} differ, e.g. {wrong[:5]}"


def test_every_header_shows_the_hit_pdf_pages(prepared):
    wrong = []
    for record in to_llm(prepared):
        headers = F0P_HEADER.findall(prompts.build(record, "v3b_med", "f0p")[1])
        shown = [pdf for printed, pdf in headers]
        expected = [", ".join(str(page) for page in hit["pdf_pages"]) for hit in record["hits"]]
        if shown != expected:
            wrong.append(record["qid"])
    assert not wrong, f"{len(wrong)} records, e.g. {wrong[:5]}"


def test_nhm09_shows_pdf_page_37(prepared):
    assert NHM09_FIRST in prompts.build(prepared["NHM09"], "v3b_med", "f0p")[1]


def test_system_prompt_is_the_same_for_f0_and_f0p(prepared):
    record = prepared["NHM09"]
    assert prompts.build(record, "v3b_med", "f0p")[0] == prompts.build(record, "v3b_med", "f0")[0]