"""Query processing: fix spelling mistakes using the guideline's OWN vocabulary.

Only spelling changes. The question stays a natural sentence (no stop-word removal, no stemming),
because vector search, the reranker and the LLM all work best on natural sentences.
"""
from __future__ import annotations

import difflib
import re
from collections import Counter

from spellchecker import SpellChecker   

MIN_LEN = 4      # shorter words are never changed (a, of, is, ANM ...)
CUTOFF  = 0.8    # how similar a correction must be (0-1); higher = fewer, safer changes


class SpellFixer:
    def __init__(self, chunks: list[dict]):
        self.vocab = Counter(w for c in chunks for w in re.findall(r"[a-z]+", c["embed_text"].lower()))
        self.words = list(self.vocab)
        self.english = SpellChecker()
        self._cache: dict[str, str | None] = {}

    def fix_word(self, word: str, first: bool = False) -> str:
        low = word.lower()
        if (len(low) < MIN_LEN or low in self.vocab or word.isupper()
                or (word[0].isupper() and not first)      # names mid-sentence: Kerala, France, Delhi
                or self.english.known([low])):            # a correct English word: leave it
            return word
        if low not in self._cache:
            best = difflib.get_close_matches(low, self.words, n=1, cutoff=CUTOFF)
            self._cache[low] = best[0] if best else None
        new = self._cache[low]
        if new is None:
            return word
        return new.capitalize() if word[0].isupper() else new

    def fix(self, question: str) -> tuple[str, dict]:
        changes, first = {}, True

        def repl(m):
            nonlocal first
            w = m.group(0)
            new = self.fix_word(w, first)
            first = False
            if new != w:
                changes[w] = new
            return new

        return re.sub(r"[A-Za-z]+", repl, question), changes    