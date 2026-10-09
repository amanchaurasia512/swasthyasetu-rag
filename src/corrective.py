"""SwasthyaSetu corrective layer (CRAG-style): grade the retrieved evidence, then choose an action.

The same logic as notebook 10 (cells 10.5-10.6), moved here so the benchmark (notebook 11) and the chat
(src/chat.py) use ONE copy of it.

    question -> spelling fix -> retrieval (top 5) -> evidence grade -> action
                                                     I  irrelevant            -> REJECT  (fixed "Not found", no LLM)
                                                     M  name/year not in PDF   -> SCOPED  (fixed template, no LLM)
                                                     P  partial               -> RE-RETRIEVE, then answer with a note
                                                     S  supported             -> answer

Usage
-----
    from src.retriever import Retriever
    from src.query import SpellFixer
    from src.corrective import Corrective
    R = Retriever()
    CR = Corrective(R, SpellFixer(R.chunks))
    prep = CR.prepare("What is the probation period?")
    prep["action"], prep["final"], prep["prompt"]     # prompt is "" when no LLM call is needed
"""
from __future__ import annotations

import re

import src.answer as answer_mod
from src.retriever import STOP

LOWER, UPPER = -5.0, 0.0          # best rerank score < LOWER: I | LOWER..UPPER: weak evidence (P) | >= UPPER: ok
NUMBER = r"(?<![a-z0-9])\d+(?:-\d+)?(?![a-z0-9])"
KEEP_STRIPS = 2                   # refine: keep the 2 best sentences of a chunk (+ their neighbours)
POOL = 10                         # re-retrieve: chunks taken per query variant
YEAR = re.compile(r"(19|20)\d\d(-\d\d)?")   # a year / 2016-17 the PDF never mentions -> M; other numbers are user input
ALIASES = {"MP": "Madhya Pradesh", "UP": "Uttar Pradesh", "HP": "Himachal Pradesh", "AP": "Andhra Pradesh",
           "TN": "Tamil Nadu", "WB": "West Bengal"}   # state short forms -> the names the PDF uses


def expand_aliases(text: str) -> str:
    """'In MP's appraisal' -> 'In Madhya Pradesh's appraisal' (whole upper-case words only)."""
    return re.sub(r"\b(" + "|".join(ALIASES) + r")\b", lambda found: ALIASES[found.group(1)], text)


def norm(text: str) -> str:
    """Lower case, one kind of dash and quote, single spaces (used for every text comparison)."""
    text = str(text).lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[‐-―−]", "-", text)
    return re.sub(r"\s+", " ", text).strip()


class Corrective:
    def __init__(self, retriever, speller, refine: bool = True):
        self.R = retriever
        self.SP = speller
        self.refine_on = refine
        self.corpus_stems = {s for c in retriever.chunks for s in self.stems(c["embed_text"])}
        self.pdf_numbers = {n for c in retriever.chunks for n in re.findall(r"\d+(?:-\d+)?", norm(c["text"]))}

    # ------------------------------------------------------------------ evidence grade (S / P / I / M)
    def stems(self, text: str) -> set[str]:
        return set(self.R.tokenize(norm(text)))

    def strong_parts(self, question: str):
        """(hard, soft, numbers): acronyms / non-English names are hard, capitalised English words are soft."""
        hard, soft = [], []
        for sentence in re.split(r"(?<=[.?!])\s+", question.strip()):
            for i, word in enumerate(re.findall(r"[A-Za-z][A-Za-z0-9]*", sentence)):
                acronym = word.isupper() and len(word) > 1
                if not acronym and (i == 0 or not word[0].isupper()):
                    continue
                is_english = bool(self.SP.english.known([word.lower()]))
                (hard if acronym or not is_english else soft).append(word)
        return hard, soft, re.findall(NUMBER, norm(question))

    def signals(self, fixed: str, hits: list[dict]) -> dict:
        evidence = " ".join(h["path"] + " " + h["text"] for h in hits)
        evidence_stems, evidence_numbers = self.stems(evidence), set(re.findall(NUMBER, norm(evidence)))
        hard, soft, numbers = self.strong_parts(fixed)

        def word_status(word):
            word_stems = self.stems(word)
            if not word_stems:
                return "ok"
            if not word_stems <= self.corpus_stems:
                return "out"
            return "ok" if word_stems <= evidence_stems else "missing"

        def number_status(number):
            if number in evidence_numbers:
                return "ok"
            if number in self.pdf_numbers:
                return "missing"
            return "out" if YEAR.fullmatch(number) else "ok"       # "200 facilities" is the user's input, not a PDF fact

        status = {w: word_status(w) for w in hard + soft}
        status.update({n: number_status(n) for n in numbers})
        return {"best": round(max((h["score"] for h in hits), default=-99.0), 2),
                "hard_out": [p for p in hard + numbers if status[p] == "out"],
                "soft_out": [p for p in soft if status[p] == "out"],
                "missing": [p for p in status if status[p] == "missing"]}

    @staticmethod
    def grade(sig: dict) -> str:
        if sig["best"] < LOWER:
            return "I"
        if sig["hard_out"]:
            return "M"
        if sig["soft_out"] or sig["missing"] or sig["best"] < UPPER:
            return "P"
        return "S"

    # ------------------------------------------------------------------ actions
    def re_retrieve(self, fixed: str, sig: dict, k: int = 5) -> list[dict]:
        keywords = " ".join(w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", fixed) if w.lower() not in STOP)
        queries = [fixed, keywords] + ([" ".join(sig["missing"]) + " " + keywords] if sig["missing"] else [])
        pool = list({h["chunk_id"]: h for q in queries for h in self.R.search(q, k=POOL)}.values())
        scores = self.R.reranker.rerank(fixed, [self.R.chunks[self.R.row_of[h["chunk_id"]]]["embed_text"] for h in pool])
        for hit, score in zip(pool, scores):
            hit["score"] = float(score)
        return sorted(pool, key=lambda h: -h["score"])[:k]

    def refine(self, hits: list[dict], question: str) -> list[dict]:
        for hit in hits:
            sentences = [s.strip() for s in re.split(r"(?<=[.;?])\s+|\n+", hit["text"]) if s.strip()]
            if hit["doc_type"] == "table" or len(sentences) <= 2 * KEEP_STRIPS + 1:
                continue
            scores = list(self.R.reranker.rerank(question, sentences))
            top = sorted(range(len(sentences)), key=lambda i: -scores[i])[:KEEP_STRIPS]
            chosen = sorted({j for i in top for j in (i - 1, i, i + 1) if 0 <= j < len(sentences)})
            hit["text"] = " ".join(("… " if n and j != chosen[n - 1] + 1 else "") + sentences[j] for n, j in enumerate(chosen))
        return hits

    @staticmethod
    def notes_for(sig: dict) -> list[str]:
        notes = []
        if sig["soft_out"]:
            notes.append(f"The sources do not mention {', '.join(sig['soft_out'])}. Say that the guideline does not cover it; "
                         f"answer only for NHM if the sources clearly cover that.")
        if sig["missing"]:
            notes.append(f"The sources may not contain: {', '.join(sig['missing'])}.")
        if sig["best"] < UPPER:
            notes.append(f"The evidence is weak. If the sources do not clearly answer the question, reply exactly: "
                         f"{answer_mod.NOT_FOUND}")
        return notes

    # ------------------------------------------------------------------ the whole layer for one question
    def prepare(self, question: str) -> dict:
        """Everything before the LLM call. 'prompt' is empty when the answer is fixed (reject / scoped)."""
        fixed, changes = self.SP.fix(question)
        fixed = expand_aliases(fixed)                 # MP -> Madhya Pradesh (shows in the 'fixed' column)
        hits = self.R.search(fixed, k=5)
        sig = self.signals(fixed, hits)
        first = final = self.grade(sig)
        if first == "P":
            hits = self.re_retrieve(fixed, sig)
            sig = self.signals(fixed, hits)
            final = self.grade(sig)
        base = {"fixed": fixed, "changes": changes, "first": first, "final": final, "best": sig["best"],
                "signals": sig, "hits": hits, "hit_ids": ";".join(h["chunk_id"] for h in hits)}
        if final == "I":
            return {**base, "action": "reject", "fixed_answer": answer_mod.NOT_FOUND, "prompt": "", "notes": ""}
        if final == "M":
            terms = ", ".join(sig["hard_out"])
            scoped = f"The guideline does not cover {terms}. It covers human resources under the National Health Mission (NHM) only."
            return {**base, "action": "scoped", "fixed_answer": scoped, "prompt": "", "notes": ""}
        notes = self.notes_for(sig) if final == "P" else []
        hits = answer_mod.add_context(hits, self.R)
        hits = self.refine(hits, fixed) if self.refine_on else hits
        prompt = ("Important: " + " ".join(notes) + "\n\n" if notes else "") + answer_mod.build_prompt(fixed, hits)
        return {**base, "hits": hits, "action": "answer_with_note" if notes else "answer", "fixed_answer": "",
                "prompt": prompt, "notes": " | ".join(notes)}
