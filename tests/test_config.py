"""Config = code (Phase 2, step 5a): every value in config/base.yaml and config/docs/nhm_hrh_2022.yaml must equal
the value the code uses today. Modules keep their own constants until after the regression test (rule R10).
Run from the project root:   python -m pytest tests/test_config.py -q        (no models, no API calls)"""
import inspect

import pytest

from src import answer, chunker, corrective, evaluate, llm, paths, prompts, query, retriever, vectordb
from src.config import load, path_of
from src.extract import nhm

CONFIG = load()

CONSTANTS = [
    ("normalize.spell_min_len", query.MIN_LEN),
    ("normalize.spell_cutoff", query.CUTOFF),
    ("normalize.typo_min_len", corrective.TYPO_MIN_LEN),
    ("retrieval.embed_model", vectordb.EMBED_MODEL),
    ("retrieval.rerank_model", retriever.RERANK_MODEL),
    ("retrieval.hybrid_pool", retriever.HYBRID_POOL),
    ("retrieval.rrf_c", retriever.RRF_C),
    ("retrieval.rerank_pool", retriever.RERANK_POOL),
    ("retrieval.keep_within", retriever.KEEP_WITHIN),
    ("retrieval.top_k", answer.TOP_K),
    ("gate.lower", corrective.LOWER),
    ("gate.lower", answer.MIN_RERANK_SCORE),
    ("gate.upper", corrective.UPPER),
    ("gate.re_retrieve_pool", corrective.POOL),
    ("evidence.context_top", answer.CONTEXT_TOP),
    ("evidence.context_tail", answer.CONTEXT_TAIL),
    ("evidence.keep_strips", corrective.KEEP_STRIPS),
    ("chunker.target", chunker.TARGET),
    ("chunker.keep_whole", chunker.KEEP_WHOLE),
    ("chunker.hard_max", chunker.HARD_MAX),
    ("chunker.min_chunk", chunker.MIN_CHUNK),
    ("chunker.overlap", chunker.OVERLAP),
    ("chunker.small", chunker.SMALL),
    ("chunker.table_budget", chunker.TABLE_BUDGET),
    ("llm.temperature", llm.OPTIONS["temperature"]),
    ("llm.api_max_tokens", llm.API_MAX_TOKENS),
    ("llm.ollama.num_ctx", llm.OPTIONS["num_ctx"]),
    ("llm.ollama.num_predict", llm.OPTIONS["num_predict"]),
    ("scoring.tau", evaluate.TAU),
    ("scoring.refuse_first_counts", evaluate.REFUSE_FIRST_COUNTS),
    ("doc_id", nhm.DOC_ID),
    ("extractor.letter_parent", nhm.LETTER_PARENT),
    ("not_found", answer.NOT_FOUND),
    ("not_found", evaluate.NOT_FOUND),
    ("aliases", corrective.ALIASES),
    ("index.collection", vectordb.COLLECTION),
]

PATHS = [
    ("pdf", paths.PDF_PATH),
    ("files.pages", paths.PAGES_FINAL),
    ("files.sections", paths.SECTIONS),
    ("files.letters", paths.LETTERS_JSONL),
    ("files.blocks", paths.BLOCKS),
    ("files.chunks", paths.CHUNKS),
    ("index.vectordb", vectordb.VECTOR_DB_DIR),
]

# values written inside a function body, not as a module constant: the text must still be in that function
LITERALS = [
    (corrective.Corrective.prepare, "self.R.search(fixed, k={retrieval.top_k})"),
    (corrective.Corrective.re_retrieve, "k: int = {retrieval.top_k}"),
    (corrective.Corrective.__init__, "refine: bool = {evidence.refine}"),
    (corrective.Corrective.__init__, 'm_action: str = "{gate.m_action}"'),
    (corrective.Corrective.__init__, "p_notes: bool = {gate.p_notes}"),
    (corrective.Corrective.__init__, "re_retrieve: bool = {gate.re_retrieve}"),
    (prompts.build, "notes: bool = {gate.p_notes}"),
    (llm._api, '"temperature": {llm.temperature}'),
    (llm._api, '"max_tokens": {llm.gemini.max_tokens}'),
    (llm._api, '"reasoning_effort": "{llm.gemini.reasoning_effort}"'),
    (llm._api, 'payload["reasoning_effort"] = "{llm.gpt_oss.reasoning_effort}"'),
]


def value_at(dotted: str):
    """'gate.lower' -> CONFIG['gate']['lower']"""
    value = CONFIG
    for key in dotted.split("."):
        value = value[key]
    return value


def filled(text: str) -> str:
    """'k={retrieval.top_k}' -> 'k=5' (Python spelling, so true -> True)."""
    for key in [part.split("}")[0] for part in text.split("{")[1:]]:
        text = text.replace("{" + key + "}", str(value_at(key)))
    return text


@pytest.mark.parametrize("dotted, code_value", CONSTANTS, ids=[dotted for dotted, code_value in CONSTANTS])
def test_constant_equals_code(dotted, code_value):
    assert value_at(dotted) == code_value


@pytest.mark.parametrize("dotted, code_path", PATHS, ids=[dotted for dotted, code_path in PATHS])
def test_path_equals_code(dotted, code_path):
    assert path_of(value_at(dotted)).resolve() == code_path.resolve()


@pytest.mark.parametrize("function, text", LITERALS, ids=[filled(text) for function, text in LITERALS])
def test_literal_still_in_code(function, text):
    assert filled(text) in inspect.getsource(function)


def test_no_secrets_and_no_api_switch_in_config():
    flat = str(CONFIG).lower()
    assert not any(word in flat for word in ("api_key", "token:", "allow_api_calls", "github_pat"))

@pytest.mark.parametrize("experiment", sorted(path.name for path in (paths.ROOT / "config" / "experiments").glob("*.yaml")))
def test_every_planned_model_has_a_pause(experiment):
    """C-036: the runner reads llm.pause[provider] for every model in the plan; a provider missing there stopped E6
    after its first model (KeyError 'sarvam'). Checked for every experiment file, before any run."""
    config = load(f"config/experiments/{experiment}")
    providers = {llm.provider_of(model)[0] for group in config["plan"] for model in group["models"]}
    assert providers <= set(config["llm"]["pause"]), providers - set(config["llm"]["pause"])