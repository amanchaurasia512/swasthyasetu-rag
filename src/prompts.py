"""SwasthyaSetu prompts: system prompt versions, weak-evidence notes and source formats (moved from notebook 15.3b).
build(record, version, fmt) -> (system, user prompt). A record is one frozen-evidence question:
it needs 'fixed' (question), 'hits' (sources), 'final' (gate grade) and 'signals'."""
from src import answer as answer_mod
from src.corrective import Corrective

FORMAT_RULES = """When you answer:
- Use ONLY these sources. No outside knowledge, no guessing.
- After each fact, and after each bullet, write the number of the source you read it in, e.g. [4].
- Tables: read row by row and keep each value with its row and column names. For "only" questions, choose rows where that column is Yes and the other columns are No.
- Length: for a single fact, answer in 1-3 sentences. When the question asks for a list, steps, a process, a form, a letter or "all" items, give every item from the sources as a bullet list."""

NHM = {"title": "Guidelines on Human Resources for Health for NHM (2022)", "word": "guideline", "outside": "NHM rules",
       "not_found": answer_mod.NOT_FOUND}


def system_v3b_med(doc: dict) -> str:
    """The v3b_med system prompt for one document (C-040): title, the word for it, what not to add and the not-found
    reply come from the document config. With NHM it is the E2 prompt character for character."""
    return f"""You answer questions about the "{doc['title']}".
First decide how to answer. Check these in order and use the FIRST that applies:
1. The question is about an organisation, place or topic the sources do not mention: reply only "The {doc['word']} does not cover <it>." Do not add {doc['outside']}.
2. The question states something as true that the sources contradict: correct it with the value the sources state.
3. The question asks about one state, year, cadre or letter, and the sources give it only for a different one: reply only "The {doc['word']} does not cover this for <the one asked>." Do not give the other one's value.
4. The sources answer only part of the question: answer that part, then say which part the {doc['word']} does not cover.
5. None of the sources contains any answer: reply exactly: {doc['not_found']}
6. Otherwise answer the question.
{FORMAT_RULES}"""


def system_v3b_strict(doc: dict) -> str:
    """The v3b_strict system prompt for one document (C-040); with NHM it is the E2 prompt character for character."""
    return f"""You answer questions about the "{doc['title']}".
First decide how to answer. Check these in order and use the FIRST that applies:
1. The question is about an organisation, place or topic the sources do not mention: reply exactly: {doc['not_found']}
2. The question states something as true that the sources contradict: correct it, but only if a source states the correct value directly; otherwise reply exactly: {doc['not_found']}
3. The question asks about one state, year, cadre or letter, and the sources give it only for a different one: reply exactly: {doc['not_found']}
4. The sources do not state the answer to EVERY part of the question: reply exactly: {doc['not_found']}
5. Otherwise answer the question.
{FORMAT_RULES}"""


SYSTEM_V3B_MED = system_v3b_med(NHM)
SYSTEM_V3B_STRICT = system_v3b_strict(NHM)

def notes_v3b_med(sig):
    """Weak-evidence notes that agree with SYSTEM_V3B_MED (refuse the out-of-scope part, never fill it with NHM rules)."""
    notes = []
    if sig["soft_out"]:
        names = ", ".join(sig["soft_out"])
        notes.append(f"The sources do not mention {names}. If the question is about {names}, reply only: The guideline does not cover {names}.")
    if sig["missing"]:
        notes.append(f"The sources may not contain: {', '.join(sig['missing'])}. If they do not, say the guideline does not cover it; do not use a value given for something else.")
    if sig["best"] < 0:
        notes.append("The evidence may be weak: answer only what the sources clearly state.")
    return notes


def notes_v3b_strict(sig):
    """Weak-evidence notes that agree with SYSTEM_V3B_STRICT (any doubt -> the exact refusal)."""
    notes = []
    if sig["soft_out"]:
        names = ", ".join(sig["soft_out"])
        notes.append(f"The sources do not mention {names}. If the question is about {names}, reply exactly: {answer_mod.NOT_FOUND}")
    if sig["missing"]:
        notes.append(f"The sources may not contain: {', '.join(sig['missing'])}. If they do not, reply exactly: {answer_mod.NOT_FOUND}")
    if sig["best"] < 0:
        notes.append(f"The evidence may be weak: if the sources do not clearly state the answer, reply exactly: {answer_mod.NOT_FOUND}")
    return notes


def pages_of(hit):
    return ", ".join(str(page) for page in hit["printed_pages"] if page is not None) or "?"


def format_source_label(question, hits):
    blocks = [f"Source [{n}]{' (table)' if hit['doc_type'] == 'table' else ''} - {hit['path']} - page {pages_of(hit)}:\n{hit['text']}"
              for n, hit in enumerate(hits, 1)]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\nAnswer:"


def format_markdown(question, hits):
    blocks = [f"### [{n}] {hit['path']} (page {pages_of(hit)}){' - table' if hit['doc_type'] == 'table' else ''}\n{hit['text']}"
              for n, hit in enumerate(hits, 1)]
    return f"## Question\n{question}\n\n## Sources\n" + "\n\n".join(blocks) + f"\n\n## Task\nAnswer the question: {question}"

def printed_and_pdf(hit):
    """'printed page 17, 18; PDF page 37, 38' - printed '?' when the page has no printed number."""
    return f"printed page {pages_of(hit)}; PDF page {', '.join(str(page) for page in hit['pdf_pages'])}"


def format_page_labels(question, hits):
    """f0p: f0 with both page numbers in each source header, so the model can tell printed pages from PDF pages
    (C-014, the NHM09 fix). Needs hits with pdf_pages: new runs only, not E2's frozen evidence."""
    blocks = [f"[{n}]{' (table)' if hit['doc_type'] == 'table' else ''} {hit['path']} ({printed_and_pdf(hit)})\n{hit['text']}"
              for n, hit in enumerate(hits, 1)]
    return (f"Question: {question}\n\nSources:\n\n" + "\n\n".join(blocks)
            + f"\n\nAnswer this question using the sources above: {question}\nAnswer:")

# Query rewriting before search (Level 1 decision, C-029): the question in plain English for searching the document.
REWRITES = {"r1": ("You rewrite a user's question so it can be used to search a document. Fix spelling mistakes. "
                   "Translate any Hindi or Hinglish words into English. Keep every name, number, date, abbreviation and "
                   "place exactly as written. Do not answer the question. Do not add, remove or guess any information. "
                   "Reply with the rewritten question only, on one line.")}

VERSIONS = {"v1": (answer_mod.SYSTEM_PROMPT, Corrective.notes_for),
            "v3b_med": (SYSTEM_V3B_MED, notes_v3b_med),
            "v3b_strict": (SYSTEM_V3B_STRICT, notes_v3b_strict)}
SYSTEMS = {"v3b_med": system_v3b_med, "v3b_strict": system_v3b_strict}     # C-040: versions that name the document

FORMATS = {"f0": answer_mod.build_prompt, "f1": format_source_label, "f2": format_markdown, "f0p": format_page_labels}


def build(record, version, fmt, notes: bool = True, doc: dict | None = None):
    """(system prompt, user prompt) for one question; notes are added only when the gate graded it P and notes is on
    (config gate.p_notes, C-028). doc: how the system prompt names the document (pipeline.document); None = NHM."""
    system, notes_fn = VERSIONS[version]
    system = SYSTEMS[version](doc) if doc and version in SYSTEMS else system
    lines = notes_fn(record["signals"]) if notes and record["final"] == "P" else []
    return system, ("Important: " + " ".join(lines) + "\n\n" if lines else "") + FORMATS[fmt](record["fixed"], record["hits"])

