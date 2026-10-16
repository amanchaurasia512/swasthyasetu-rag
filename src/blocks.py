"""SwasthyaSetu blocks: the contract between an extractor and the generic chunker (design note 5.1).
Every extractor writes blocks; the chunker reads only blocks. validate() checks the invariants."""
import json

BLOCK_FIELDS = ("doc_id", "block_id", "order", "unit", "title", "path", "level", "parent_id", "pdf_pages",
                "printed_pages", "source", "searchable", "summary", "items", "footnotes", "chunk_fields", "meta")
UNITS = {"section", "glossary", "letter"}
ITEM_TYPES = {"heading", "text", "table", "caption", "diagram", "paragraph"}


def read_blocks(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def write_blocks(path, blocks):
    with open(path, "w", encoding="utf-8") as file:
        for block in blocks:
            file.write(json.dumps(block, ensure_ascii=False) + "\n")


def validate(blocks, pages):
    """Invariants of design note 5.1; returns a list of problems (empty = valid). pages = pages.jsonl records."""
    printed = {page["pdf_page"]: page["printed_page"] for page in pages}
    ids = [block["block_id"] for block in blocks]
    problems = []
    if len({block["doc_id"] for block in blocks}) != 1:
        problems.append("more than one doc_id in the file")
    if len(set(ids)) != len(ids):
        problems.append("block_id not unique")
    if [block["order"] for block in blocks] != list(range(1, len(blocks) + 1)):
        problems.append("order does not run 1..n")
    for block in blocks:
        name = block["block_id"]
        missing = [field for field in BLOCK_FIELDS if field not in block]
        if missing:
            problems.append(f"{name}: missing fields {missing}")
            continue
        if block["unit"] not in UNITS:
            problems.append(f"{name}: unknown unit {block['unit']!r}")
        if not block["pdf_pages"] or block["pdf_pages"] != sorted(block["pdf_pages"]):
            problems.append(f"{name}: pdf_pages empty or not ascending")
        if len(block["printed_pages"]) != len(block["pdf_pages"]):
            problems.append(f"{name}: printed_pages and pdf_pages differ in length")
        if any(printed.get(pdf) != label for pdf, label in zip(block["pdf_pages"], block["printed_pages"])):
            problems.append(f"{name}: printed label disagrees with pages.jsonl")
        if block["parent_id"] is not None and block["parent_id"] not in ids:
            problems.append(f"{name}: parent {block['parent_id']} does not exist")
        if not block["path"]:
            problems.append(f"{name}: empty path")
        if any(item["pdf_page"] is not None and item["pdf_page"] not in block["pdf_pages"] for item in block["items"]):
            problems.append(f"{name}: an item's page is outside the block's pages")
        if any(item["type"] not in ITEM_TYPES for item in block["items"]):
            problems.append(f"{name}: unknown item type")
        if not block["searchable"] and not block["summary"]:
            problems.append(f"{name}: not searchable but has no summary")
    return problems