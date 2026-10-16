"""NHM extractor, block side: repack sections.jsonl + letters.jsonl into blocks (design note 5.1, option A).
Everything particular to the NHM guideline lives here: the letter compendium's name, how a letter's path and
stub text are written, which sections count as a glossary. The chunker sees only the finished labels."""
DOC_ID = "nhm_hrh_2022"
LETTER_PARENT = "Annexure II: Compendium of Letters"
LINE_TYPES = {"L0": "heading", "L1": "heading", "L2": "heading", "L3": "heading", "body": "text",
              "table": "table", "caption": "caption", "diagram": "diagram"}


def section_block(sec, method, order):
    """method = {pdf_page: 'text' | 'ocr'}; a section is OCR text if any of its pages was read by OCR."""
    tables = {table["id"]: table for table in sec["tables"]}
    items = []
    for line in sec["lines"]:
        kind = LINE_TYPES.get(line["level"], "text")
        items.append({"type": kind, "depth": int(line["level"][1]) if kind == "heading" else None,
                      "text": line["text"], "pdf_page": line["pdf_page"], "top": line.get("top"),
                      "table": tables[line["text"][7:-1]] if kind == "table" else None})
    return {"doc_id": DOC_ID, "block_id": sec["section_id"], "order": order,
            "unit": "glossary" if sec["kind"] == "glossary" else "section",
            "title": sec["title"], "path": sec["path"], "level": sec["level"], "parent_id": sec["parent_id"],
            "pdf_pages": sec["pdf_pages"], "printed_pages": sec["printed_pages"],
            "source": "ocr" if any(method.get(page) == "ocr" for page in sec["pdf_pages"]) else "text",
            "searchable": True, "summary": None, "items": items, "footnotes": sec["footnotes"],
            "chunk_fields": {"section_id": sec["section_id"], "letter_no": None},
            "meta": {"kind": sec["kind"], "number": sec["number"], "label": sec["label"],
                     "printed_start": sec["printed_start"], "printed_end": sec["printed_end"]}}


def letter_block(letter, order):
    no = letter["letter_no"]
    readable = letter["status"] == "ok"
    return {"doc_id": DOC_ID, "block_id": f"L{no:02d}", "order": order, "unit": "letter",
            "title": letter["subject"],
            "path": [LETTER_PARENT, f"Letter {no} ({letter['date_text']}): {letter['subject']}"],
            "level": 1, "parent_id": None,
            "pdf_pages": letter["pdf_pages"], "printed_pages": letter["printed_pages"],
            "source": "ocr", "searchable": readable,
            "summary": None if readable else f"Letter {no} dated {letter['date_text']}: {letter['subject']}. {letter['notice']}",
            "items": [{"type": "paragraph", "depth": None, "text": part.strip(), "pdf_page": None, "top": None, "table": None}
                      for part in letter["body"].split("\n\n") if part.strip()],
            "footnotes": [], "chunk_fields": {"section_id": None, "letter_no": no},
            "meta": {"letter_no": no, "date": letter["date"], "date_text": letter["date_text"],
                     "subject": letter["subject"], "status": letter["status"], "notice": letter["notice"],
                     "page_status": letter["page_status"]}}


def build_blocks(sections, letters, pages):
    """All NHM blocks in reading order: the 106 sections, then the 21 letters."""
    method = {page["pdf_page"]: page["method"] for page in pages}
    blocks = [section_block(sec, method, order) for order, sec in enumerate(sections, 1)]
    return blocks + [letter_block(letter, order) for order, letter in enumerate(letters, len(blocks) + 1)]


if __name__ == "__main__":
    import json
    from src.blocks import validate, write_blocks
    from src.paths import BLOCKS, LETTERS_JSONL, PAGES_FINAL, SECTIONS

    def read_jsonl(path):
        with open(path, encoding="utf-8") as file:
            return [json.loads(line) for line in file]

    pages = read_jsonl(PAGES_FINAL)
    blocks = build_blocks(read_jsonl(SECTIONS), read_jsonl(LETTERS_JSONL), pages)
    problems = validate(blocks, pages)
    if problems:
        raise SystemExit(f"blocks NOT written, {len(problems)} problems: {problems[:10]}")
    write_blocks(BLOCKS, blocks)
    print(f"wrote {len(blocks)} blocks -> {BLOCKS.name} (all invariants hold)")