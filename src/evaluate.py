"""SwasthyaSetu answer evaluation: scores answers that already exist (no LLM calls here).

Algorithm 1  trust_score  - strategy of Trust-Score (Song et al., ICLR 2025):
             grounded refusals (refuse vs answer, both directions), answer correctness calibrated to the facts that
             were IN THE PROMPT, citation groundedness; plus answered ratio and one error type per answer.
Algorithm 2B rule_checks / rule_rates - strategy of ConInstruct (He et al., AAAI 2026):
             one satisfaction rate per prompt rule, so we see WHICH instruction the model breaks.
(Algorithm 2A, the conflict audit of the prompt text, is done by hand before writing a prompt version.)

Input of score_run(): one row per answer with columns
    qid, label ('answerable' / 'unanswerable'), question, expected, action, prompt, sources (list of source texts), answer
Only answers that reached the LLM belong here; questions decided by the gate are counted separately.
"""
from __future__ import annotations

import re

import pandas as pd

TAU = 0.5                                   # a fact set counts as covered when at least this share is found
NOT_FOUND = "Not found in the guideline."
REFUSAL_PHRASES = ("not found in the guideline", "does not cover", "do not cover", "does not mention", "do not mention",
                   "not covered", "no information", "not in the guideline", "not given", "not mentioned", "no mention",
                   "do not contain", "does not contain")
REFUSE_FIRST_COUNTS = False   # Option 1 (strict). True = Option 2: "Not covered for X; the guideline gives Y for Z" is a refusal
LONG_REQUEST = re.compile(r"\b(list|all|steps?|process|procedure|forms?|format|letters?|message|explain|describe|"
                          r"components?|responsibilit\w*|duties|contents?|roadmap|phases?)\b", re.I)
STOP_WORDS = set("""the a an and or of to in on for by with from as at is are be this that these those it its which what who
will can may should must also each every other than then into per not no any all their there here such has have had
was were been being if when where how does do done only more most some your you them they given give gives""".split())


# ---------------------------------------------------------------- facts (same rules as notebook 13.10 / 14.2)
def numbers(text) -> set[str]:
    joined = re.sub(r"(?<=\d)[\s\u00a0\u2009\u202f](?=\d{3}\b)", "", str(text))
    return {found.replace(",", "") for found in re.findall(r"\d[\d,]*(?:\.\d+)?", joined) if len(found.replace(",", "")) >= 2}


def key_words(text) -> set[str]:
    return {word[:6] for word in re.findall(r"[a-z]+", str(text).lower()) if len(word) >= 4 and word not in STOP_WORDS}


def facts(expected) -> list[str]:
    return [piece.strip() for piece in re.split(r";|:|\.\s|\s-\s|\(\d\)", str(expected)) if key_words(piece) or numbers(piece)]


def fact_found(fact, text_words: set[str], text_numbers: set[str]) -> bool:
    fact_words = key_words(fact)
    words_ok = len(fact_words & text_words) >= max(1, len(fact_words) / 2) if fact_words else True
    return words_ok and numbers(fact) <= text_numbers


def found_in(fact_list: list[str], text) -> list[str]:
    text_words, text_numbers = key_words(text), numbers(text)
    return [fact for fact in fact_list if fact_found(fact, text_words, text_numbers)]


# ---------------------------------------------------------------- answer parts
def first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?;])\s", text, maxsplit=1)[0]


def refused(answer, action: str = "answer") -> bool:
    """A gate reply; the exact NOT_FOUND text at the start or the end; a SHORT reply built around a refusal phrase;
    with REFUSE_FIRST_COUNTS also any reply whose first sentence is a refusal."""
    text = str(answer).strip().lower()
    short_refusal = len(text.split()) <= 25 and any(phrase in text for phrase in REFUSAL_PHRASES)
    opens_with_refusal = any(phrase in first_sentence(text) for phrase in REFUSAL_PHRASES)
    return (action in ("reject", "scoped") or text.startswith(NOT_FOUND.lower()) or text.endswith(NOT_FOUND.lower())
            or short_refusal or (REFUSE_FIRST_COUNTS and opens_with_refusal))


def answer_mode(question) -> str:
    """'long' when the question asks for a list / steps / form / letter / explanation, else 'short'."""
    return "long" if LONG_REQUEST.search(str(question)) else "short"



def bullet_count(answer) -> int:
    return len(re.findall(r"^\s*(?:[-*•]|\d+[.)]|\|)", str(answer), flags=re.M))


# ---------------------------------------------------------------- Algorithm 1, lines 1-12: one answer
def score_answer(row) -> dict:
    is_refused = refused(row.answer, row.action)
    available = found_in(facts(row.expected), row.prompt) if row.expected else []
    llm_label = "answerable" if available else "unanswerable"          # answerability judged on the PROMPT
    correctness = round(len(found_in(available, row.answer)) / len(available), 2) if available and not is_refused else None
    parts = [] if is_refused else statements(row.answer)
    supported = precise = cited_total = 0
    for text, cites in parts:
        cited = [row.sources[n - 1] for n in cites if 1 <= n <= len(row.sources)]
        supported += bool(cited) and bool(found_in([text], " ".join(cited)))
        precise += sum(bool(found_in([text], source)) for source in cited)
        cited_total += len(cited)
    cite_recall = supported / len(parts) if parts else None
    cite_precision = precise / cited_total if cited_total else (0.0 if parts else None)
    if llm_label == "answerable" and is_refused:
        error = "unwarranted refusal"
    elif llm_label == "unanswerable" and not is_refused:
        error = "over-responsiveness"
    elif correctness is not None and correctness < TAU:
        error = "inaccurate answer"
    elif cite_recall is not None and cite_recall < TAU:
        error = "improper citation"
    else:
        error = ""
    return {"refused": is_refused, "llm_label": llm_label, "facts_in_prompt": len(available), "correctness": correctness,
            "statements": len(parts), "cite_recall": cite_recall, "cite_precision": cite_precision, "error": error}


# ---------------------------------------------------------------- Algorithm 2B: one check per prompt rule
def rule_checks(row, measures: dict) -> dict:
    """True / False per rule; a rule that does not apply to this answer is left out (counts as not applicable)."""
    checks = {"refuse only when nothing": measures["refused"] == (measures["llm_label"] == "unanswerable")}
    if measures["refused"]:
        return checks
    parts, mode = statements(row.answer), answer_mode(row.question)
    checks["sources only (numbers)"] = numbers(row.answer) <= numbers(row.prompt) | numbers(row.question)
    checks["cite every statement"] = bool(parts) and all(any(1 <= n <= len(row.sources) for n in cites) for text, cites in parts)
    checks["length fits question"] = len(parts) <= 3 if mode == "short" else bullet_count(row.answer) >= 2
    if mode == "long" and measures["correctness"] is not None:
        checks["complete list"] = measures["correctness"] >= TAU
    return checks


def score_run(answers: pd.DataFrame) -> pd.DataFrame:
    """Per-answer measures (Algorithm 1) and rule checks (Algorithm 2B) added as columns."""
    rows = []
    for row in answers.itertuples():
        measures = score_answer(row)
        rows.append({**measures, **{f"rule: {name}": passed for name, passed in rule_checks(row, measures).items()}})
    return pd.concat([answers.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


# ---------------------------------------------------------------- Algorithm 1, lines 13-17: one run
def f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def flag_f1(predicted: pd.Series, actual: pd.Series) -> float:
    hits = (predicted & actual).sum()
    return f1(hits / max(predicted.sum(), 1), hits / max(actual.sum(), 1))


def trust_score(scored: pd.DataFrame) -> dict:
    answerable, answered = scored.llm_label.eq("answerable"), ~scored.refused.astype(bool)
    f1_gr = (flag_f1(answered, answerable) + flag_f1(~answered, ~answerable)) / 2
    good = scored[answered & answerable]
    f1_ac = f1(good.correctness.mean() if len(good) else 0.0, good.correctness.sum() / max(answerable.sum(), 1))
    cited = scored[answered & scored.statements.gt(0)]
    f1_gc = f1(cited.cite_precision.mean(), cited.cite_recall.mean()) if len(cited) else 0.0
    return {"trust": round((f1_gr + f1_ac + f1_gc) / 3, 3), "f1_refusal": round(f1_gr, 3), "f1_correct": round(f1_ac, 3),
            "f1_citation": round(f1_gc, 3), "answered_ratio": round(answered.mean(), 3),
            "over_answer": int((answered & ~answerable).sum()), "over_refusal": int((~answered & answerable).sum()),
            "n": len(scored)}


def rule_rates(scored: pd.DataFrame) -> pd.Series:
    """Algorithm 2B, line 9: share of applicable answers that follow each rule."""
    rule_columns = [column for column in scored.columns if column.startswith("rule: ")]
    return scored[rule_columns].astype(float).mean().round(3).rename(lambda column: column[6:]) 

CITE = re.compile(r"[\[【]\s*(\d+(?:\s*[,;–-]\s*\d+)*)\s*(?:†[^\]】]*)?[\]】]")


def citations(sentence) -> list[int]:
    """Source numbers in [4], [1, 3], [1-3], 【4】 or 【4†L2-L5】 citations."""
    cited = []
    for group in CITE.findall(str(sentence)):
        for piece in re.split(r"\s*[,;]\s*", group):
            ends = [int(number) for number in re.findall(r"\d+", piece)]
            cited += list(range(ends[0], ends[-1] + 1))
    return cited


def statements(answer) -> list[tuple[str, list[int]]]:
    """Answer -> (statement text, cited source numbers). Lines are split into sentences; a sentence without its
    own citation takes the citations of the next cited sentence on the same line ("… fact. … fact [2]")."""
    result = []
    for line in str(answer).split("\n"):
        pending = []
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line):
            text = CITE.sub("", sentence).strip(" -*•|#>")
            if len(key_words(text)) < 2 and not (key_words(text) and numbers(text)):
                continue
            cites = citations(sentence)
            pending.append(text)
            if cites:
                result += [(item, cites) for item in pending]
                pending = []
        result += [(item, []) for item in pending]
    return result

# ---------------------------------------------------------------- Phase 2, step 1b: scoring from a file (C-004)
SCORING_COLUMNS = ["model", "version", "format", "run", "qid", "answer", "source", "label",
                   "question", "action", "expected", "prompt", "sources"]


def load_scoring_input(path) -> pd.DataFrame:
    """Saved answers with their question data, sources and exact prompt (e.g. e2_scoring_input.jsonl)."""
    return pd.read_json(path, lines=True, dtype=False)


def select(rows: pd.DataFrame, pairs, qids, fmt: str = "f0", max_run: int = 2) -> pd.DataFrame:
    """Rows of the planned (model, version) pairs on the given questions, one format, runs 1..max_run."""
    planned = (rows.model + " | " + rows.version).isin({f"{model} | {version}" for model, version in pairs})
    return rows[planned & rows.qid.isin(set(qids)) & rows.format.eq(fmt) & rows.run.le(max_run)]


def scorecard(scored: pd.DataFrame) -> pd.Series:
    """The three numbers the goal depends on (notebook 15.15)."""
    refused = scored.refused.astype(bool)
    answerable, unanswerable = scored.llm_label.eq("answerable"), scored.llm_label.eq("unanswerable")
    correct = ~refused & scored.correctness.astype(float).fillna(0).ge(TAU)
    cited = scored.cite_recall.astype(float)[~refused].dropna()
    return pd.Series({"correct on answerable %": 100 * correct[answerable].mean(),
                      "over-answer on unanswerable %": 100 * (~refused)[unanswerable].mean(),
                      "statements cited %": 100 * cited.mean(),
                      "answers": len(scored)})


def scorecards(scored: pd.DataFrame) -> pd.DataFrame:
    """One scorecard row per model and prompt version."""
    return scored.groupby(["model", "version"]).apply(scorecard)


def trade_off(scored: pd.DataFrame, models, versions, retrieval_miss_qids=frozenset(), accepted_corrections=frozenset(),
              p_answerable: float = 0.7, k_values=(1, 2, 3), chosen_k: int = 3):
    """Expected cost per 100 questions: p * missed answers + (1 - p) * real over-answers * k (notebook 15, trade-off cell).
    Returns the table sorted by cost at chosen_k and the selected (model, version)."""
    cards = scorecards(scored)
    rows = scored[scored["model"].isin(models) & scored["version"].isin(versions)]
    unanswerable = rows[(rows["llm_label"] == "unanswerable") & ~rows["qid"].isin(retrieval_miss_qids)]
    real_over_answer = ~unanswerable["refused"] & ~unanswerable["qid"].isin(accepted_corrections)
    over_rate = real_over_answer.groupby([unanswerable["model"], unanswerable["version"]]).mean()
    table = cards.loc[over_rate.index, ["correct on answerable %", "statements cited %"]].copy()
    table["real over-answer %"] = (100 * over_rate).round(1)
    miss_cost = 100 * p_answerable * (1 - table["correct on answerable %"] / 100)
    over_cost = 100 * (1 - p_answerable) * over_rate
    for k in k_values:
        table[f"cost k={k}"] = (miss_cost + over_cost * k).round(1)
    table = table.sort_values([f"cost k={chosen_k}", "statements cited %"], ascending=[True, False])
    selected = table.index[0]
    table["beats selected if k <"] = ((miss_cost[selected] - miss_cost) / (over_cost - over_cost[selected])
                                      ).where(over_cost > over_cost[selected]).round(2)
    return table, selected