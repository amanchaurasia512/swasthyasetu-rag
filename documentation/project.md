# SwasthyaSetu

**A question-answering chatbot over the NHM guideline *Human Resources for Health* (2022).** It answers in English or Hinglish, only from the guideline, with the page numbers of its sources, and it says so when the guideline does not cover a question.

Author: Aman Chaurasia · Built October 2026 · Full design and evaluation record: [`docs/restructure/swasthyasetu_restructure.pdf`](docs/restructure/swasthyasetu_restructure.pdf)

---

## Result

| Metric | E3 (before the level checks) | **E5 (the chatbot)** | Target |
|---|---|---|---|
| Correct on answerable questions | 79.9% | **85.3%** | ≥ 90% |
| Over-answer on unanswerable questions | 0.0% | **2.2%** (1 question) | ≤ 5% |
| Answer statements with a citation | 80.0% | 79.5% | — |
| Questions that needed a rewrite call | — | 28 of 175 (16%) | — |

Same 175 questions, model (Gemini flash-lite), prompt (`v3b_med`) and format (`f0p`) in both runs, 2 runs each: every difference comes from the decisions described below. E5 gave 242 answers, none cut off; the two runs agreed on every refusal.

![E3 vs E5: correct and over-answer](docs/images/results_e3_e5.png)

The one over-answer (C-062) is a question with a false premise ("introduced in 1990, right?"); the chatbot corrected it from the guideline (1974) and answered. The question is labelled unanswerable in the key, so it counts against the score; with the label corrected the over-answer rate is 0%.

![Where questions were lost, E3 vs E5](docs/images/lost_at_e3_e5.png)

---

## How it works

```
question
  → English check: every word English or in the PDF?   yes → spelling fixer (SpellFixer)
                                                        no  → Gemini rewrite into plain English (1 extra call)
  → hybrid search (BM25 + vectors) → rerank
  → gate: grade the evidence  S (good) / P (weak) / M (word not in the PDF) / I (irrelevant → fixed reply, no call)
  → prompt v3b_med + format f0p (sources with printed and PDF page numbers)
  → Gemini answers from the sources, with citations
```

- **Experiments** run through `src/run.py`: dry run by default, API calls only with `--allow-api-calls`, every run in its own folder with a manifest (commit, config, data fingerprints), evidence, traces, answers and scorecard.
- **Diagnosis** (`src/diagnose.py`) finds, for every question, the first stage that lost it: index, retrieval, gate, context, the model refusing, or the model answering incompletely.
- **The chatbot** (`src/chat.py`, `config/app.yaml`) takes exactly the path that is scored, and a test keeps `config/experiments/e5.yaml` and `config/app.yaml` in agreement.
- **The chat page** (`app/streamlit_app.py`, dark theme) shows the answer, its sources with PDF pages, and what the system did (grade, model calls, seconds).

---

## Problems faced, and how they were solved

| Problem | What showed it | What was done |
|---|---|---|
| Notebooks and runs scattered across VS Code, Colab and Kaggle; results lost | Phase 0 map of the old pipeline | Restructured into plain `src/` stages with config files, run manifests and a regression test that replays E2 exactly (C-001–C-013) |
| Users would jump to the wrong page: printed page numbers differ from PDF pages | NHM09 answer pointed to the wrong pages | Format `f0p`: every source shows both page numbers (C-014) |
| A score says how many answers were wrong, not where | E3 scorecard | Diagnosis per question, after RAGChecker and the Seven Failure Points of RAG (C-017) |
| Fixes kept breaking on the next new question | Hard-coded word lists and special cases | Every stage checked on its own, ON vs OFF, with the pass rule written down before the check (C-018–C-027) |
| The model refused answerable questions | E3: P-graded questions right only 16 of 34 times; the warning notes pushed it to refuse | P notes switched off; E5: 27 of 38, five refusals fixed (C-025, C-028) |
| Typos and Hinglish | Hold-out of the author's own typos: spelling fixer fixed 16 of 45 words, an LLM rewrite 34; Hinglish questions passed the gate 7 vs 19 of 20 | LLM rewrite as the front end (C-029) |
| A rewrite call on every question doubles the cost | Cost per question | English check: rewrite only when a word is neither English nor in the PDF; 46/46 Hinglish and 9/9 rewrite-helped questions caught, about 3% unnecessary calls on clean English (C-032) |
| A cache of rewrites looked cheap | Exact repeats of typo/Hinglish questions are rare; a cache of known questions hides real waiting time and failures | Live rewrite, no cache; a question log to measure repeats first (C-029, C-033) |
| A test set written by the same hand that builds the system flatters it | On typos written by Claude the spelling fixer passed the gate for 20 of 20 questions; on the author's own typos it fixed only 16 of 45 words | Human-written hold-out, opened once |
| Names typed differently from the PDF | "Bharati" became "bharat"; letter senders not in the index | Name list built from the PDF (85 candidates, 24 not in the index); matcher dropped, stated as a limitation (C-030) |
| The chatbot answered with a prompt that was never scored | Old `chat.py` | Chat on the scored path, settings from `config/app.yaml` (C-033) |
| `python -m src.run` silently did nothing | The `__main__` lines were lost by copy-paste twice while every test passed | A test for the entry point; it caught a third loss (C-031) |

---

## Run it

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```
API keys go in `.env` (never committed). Then:

| What | Command |
|---|---|
| Tests (no API calls, ~12 min) | `python -m pytest tests -q` |
| Chat page | `streamlit run app/streamlit_app.py` → http://localhost:8501 |
| Experiment dry run | `python -m src.run --config config/experiments/e5.yaml` |
| Experiment, real | `python -m src.run --config config/experiments/e5.yaml --allow-api-calls` |
| Diagnosis of a run | `python -m src.diagnose e5-20261018-2210` |

---

## Limitations

- Below the 90% target: 10 answers are incomplete although the facts were in the prompt, some scored down by answer keys that also list section references.
- Names typed differently from the PDF ("Jhalni", "Bhawan") are not corrected.
- Real-word mistakes ("preference" for "performance") are fixed neither by the spelling fixer nor by the rewrite.
- Not searchable yet: the salary table T76-1-01, the front matter, letter sender names.
- Questions with unknown words cost two calls and need the Gemini API; British spellings ("computerised") trigger a rewrite that was not needed.
- The gate cut-off, the English check and the prompt were tuned on this PDF and these question sets; 175 questions means one question moves a score by about 0.6 points; the scorer is lexical.

## Future work

- A check after the answer: is every sentence supported, is the answer complete (MiniCheck, Self-RAG, ARES-style judge).
- A small trained rewriter distilled from the Gemini rewrites, run locally (no rewrite call), then other Indian languages.
- A cache only if the question log shows enough repeats; a fresh test set from real users' questions.
- Index fixes (salary table, front matter, letter senders); a FastAPI endpoint for other front ends; a greeting reply.
- A second PDF, to prove the pipeline is document-independent.

## Credits

Methods taken from CRAG (the gate), STC (table chunking), Trust-Score (refusal and citation measures), RAGChecker and the Seven Failure Points (diagnosis), QE-RAG and Flipkart's query rewriting (front end), RRF, BM25, BGE, MS MARCO, ChromaDB. Answers by Google Gemini. Claude (Anthropic) helped with the design, code and documentation. The full list, with what was taken and where it lives in the code, is in the restructure document.
