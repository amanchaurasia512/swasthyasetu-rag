"""The document's own names - people, places, schemes (C-030, step 1: the candidate list).

A name is taken only from this PDF, never from the test questions. A word is name-like when it is capitalised, is
not an English dictionary word, and is never written in lower case anywhere in the PDF ("Jhalani", "Kalyan"; not
"Secretary", which the PDF also writes as "secretary"). A candidate is a run of capitalised
words with at least one name-like word, trimmed of the words the PDF also writes in lower case
("Manoj Jhalani Additional Secretary" -> "Manoj Jhalani"; "Tamil Nadu" stays whole).
The list is noisy on purpose: the author reviews it and saves the rows to keep (config names.reviewed).

    python -m src.names candidates        writes config names.candidates (no models, no API calls)
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

import pandas as pd
from spellchecker import SpellChecker

from src.config import load, path_of

RUN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}\b")     # capitalised words in a row (ALL-CAPS left out)
ADDRESS = re.compile(r"\S+@\S+|https?://\S+|www\.\S+")              # e-mails and links: manoj.jhalani@nic.in is not prose
COLUMNS = ["name", "count", "in_index", "example", "kind", "keep"]


def pdf_texts(config: dict) -> tuple[list[str], list[str]]:
    """(every page line, every chunk text). Page lines include pages kept out of the index (letters, front matter)."""
    with open(path_of(config["files"]["pages"]), encoding="utf-8") as file:
        lines = [line["text"] for page in map(json.loads, file) for line in page["lines"]]
    with open(path_of(config["files"]["chunks"]), encoding="utf-8") as file:
        chunks = [chunk["text"] for chunk in map(json.loads, file)]
    return lines, chunks


def lower_words(texts: list[str]) -> set[str]:
    """Every word the PDF writes in lower case in its prose (e-mail addresses and links left out)."""
    return {word for text in texts for word in re.findall(r"\b[a-z]+\b", ADDRESS.sub(" ", text))}


def trimmed(span: str, name_like, lower: set[str]) -> str:
    """The span without the words the PDF also writes in lower case at either end ("Additional Secretary"), or ""
    when no name-like word is in it. Dictionary words that the PDF only ever capitalises stay ("Tamil Nadu")."""
    words = span.split()
    if not any(name_like(word) for word in words):
        return ""
    kept = [index for index, word in enumerate(words) if word.lower() not in lower]
    return " ".join(words[kept[0]:kept[-1] + 1])


def candidates(texts: list[str], index_texts: list[str], english: SpellChecker | None = None) -> pd.DataFrame:
    """One row per candidate name: how often it appears, whether the index (chunks) holds it, one example line."""
    english = english or SpellChecker()
    lower = lower_words(texts + index_texts)
    name_like = lambda word: len(word) >= 3 and word.lower() not in lower and not english.known([word.lower()])
    counts, examples = Counter(), {}
    for text in texts + index_texts:
        for span in RUN.findall(text):
            name = trimmed(span, name_like, lower)
            if name:
                counts[name] += 1
                examples.setdefault(name, " ".join(text.split())[:120])
    index = " ".join(index_texts).lower()
    rows = [{"name": name, "count": count, "in_index": name.lower() in index, "example": examples[name],
             "kind": "", "keep": ""} for name, count in counts.most_common()]
    return pd.DataFrame(rows, columns=COLUMNS)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="The document's own names")
    parser.add_argument("action", choices=["candidates"])
    args = parser.parse_args(argv)
    config = load()
    lines, chunks = pdf_texts(config)
    table = candidates(lines, chunks)
    out = path_of(config["names"]["candidates"])
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_excel(out, index=False)
    print(f"{len(table)} candidate names ({int((~table.in_index).sum())} not in the index) -> {out}\n"
          f"Review: fill 'keep' (yes) and 'kind' (person / place / scheme / other), "
          f"then save the reviewed copy as {config['names']['reviewed']}")


if __name__ == "__main__":
    main()