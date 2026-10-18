"""The document's own names (C-030, step 1): name-like words, trimming, counts, the index flag, and the real PDF list.
Run from the project root:   python -m pytest tests/test_names.py -q        (no models, no API calls; seconds)"""
from src import names
from src.config import load

TEXTS = ["The Rogi Kalyan Samiti meets every month in the district.",
         "Manoj Jhalani Additional Secretary, write to manoj.jhalani@nic.in",
         "The additional secretary of the health centre signed it.",
         "Part 3: Public Health Cadre in Tamil Nadu"]


def test_names_are_found_and_ordinary_words_trimmed():
    table = names.candidates(TEXTS, [])
    found = set(table.name)
    assert {"Rogi Kalyan Samiti", "Manoj Jhalani", "Tamil Nadu"} <= found
    assert not found & {"Additional Secretary", "The", "Part", "Public Health Cadre"}


def test_counts_and_the_index_flag():
    table = names.candidates(TEXTS + ["Rogi Kalyan Samiti again"], ["Tamil Nadu has a public health cadre."])
    rows = table.set_index("name")
    assert rows.loc["Rogi Kalyan Samiti", "count"] == 2
    assert bool(rows.loc["Tamil Nadu", "in_index"]) and not bool(rows.loc["Manoj Jhalani", "in_index"])
    assert list(table.columns) == names.COLUMNS


def test_real_pdf_list_shows_names_missing_from_the_index():
    table = names.candidates(*names.pdf_texts(load())).set_index("name")
    assert bool(table.loc["Madhya Pradesh", "in_index"]) and bool(table.loc["Rogi Kalyan Samiti", "in_index"])
    assert not bool(table.loc["Nirman Bhavan", "in_index"]) and "Nirman Bhawan" in table.index   # both spellings