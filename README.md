# SwasthyaSetu

Question answering over the NHM guideline "Human Resources for Health for NHM (2022)": retrieval (BM25 + vectors,
reranked), a corrective gate that refuses what the guideline does not cover, and an LLM that answers with citations.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

API keys go in `.env` (never committed). `data/` and `runs/` are not in git; back them up separately.

## Tests (no API calls)

```
python -m pytest -q
```

About 9 minutes; local models only.

## Running an experiment

Every run gets its own folder `runs/<experiment>-<date>-<time>/` with `manifest.json`, `evidence.jsonl`,
`traces.jsonl`, `answers.csv` and `run.log`.

```
python -m src.run --config config/experiments/e3.yaml                  # dry run: shows the plan, calls nothing
python -m src.run --resume <run_id>                                    # what is still missing, calls nothing
python -m src.run --resume <run_id> --allow-api-calls                  # answer only the missing ones
```

Rules:
- API calls happen only with `--allow-api-calls` typed on the command line; no file can switch them on.
- Commit before a real run, so the manifest records `git_dirty: false`.
- A resumed run refuses to continue if the settings or a question file changed; start a new run instead.

## Where things live

- `config/` - settings: `base.yaml` (any document), `docs/` (one file per PDF), `experiments/` (one file per run plan)
- `src/` - the pipeline as plain functions; `src/extract/` holds the code particular to one PDF
- `tests/` - regression tests that prove each stage still gives the frozen E2 results
- `notebooks/` - exploration and history; experiments no longer run from notebooks
- `docs/restructure/` - the restructuring report (LaTeX)