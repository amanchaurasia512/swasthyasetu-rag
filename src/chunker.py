"""SwasthyaSetu chunker: blocks -> chunks. Generic: it reads only block fields (design note 5.1), never meta,
and knows nothing about any particular PDF. Logic moved from notebook 06: text packed by size with one-sentence
overlap, tables by STC (Guttal et al. 2026), and one stub chunk (the block's summary) for a block that is not searchable.
build_chunks(blocks, pages) -> chunk records."""
import json
import re
from collections import Counter

# ---- size settings (characters; ~4-5 chars per token) ----
TARGET     = 1000     # a split section aims for chunks about this size (~220 tokens)
KEEP_WHOLE = 1400     # a section up to this size stays ONE chunk, even above TARGET
HARD_MAX   = 1600     # no chunk may be bigger (a 512-token embedding model fits this)
MIN_CHUNK  = 300      # a sub-heading starts a new chunk only if the current one has at least this much
OVERLAP    = 1        # sentences repeated at the start of the next chunk of the same section
SMALL      = 200      # a chunk under this is merged into its neighbour (same section/letter) when it fits
TABLE_BUDGET = TARGET # B in characters (~250 tokens); embedder and reranker read only ~512 tokens

# dots that do NOT end a sentence: abbreviations and single initials (C.K., D.O.)
NO_END = re.compile(r"(\b(?:No|Nos|e\.g|i\.e|Rs|Dr|Mr|Ms|Mrs|Smt|Shri|Sh|etc|viz|vs|Govt|Dept|Fig|approx|Sl|para)\.|\b[A-Za-z]\.)$",
                    re.I)
LIST_START = re.compile(r"^(-\s|\(?[ivx]{1,4}\)\s|\(?[a-h]\)\s|[a-h]\.\s|\d{1,2}[.)]\s)")
LIST_MARK  = re.compile(r"^(-\s|\(?[a-z]{1,4}[.)]\s|e\.g\.,?\s)", re.I)              # a) · c. · iii. · e.g. · "- "
LABEL_BODY = re.compile(r"^((chapter|annexure)(\s*[-:]?\s*[0-9ivx]+)?|[ivx]+)\s*$", re.I)   # "Annexure" · "I" · "III"


def read_jsonl(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as file:
        for item in records:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def para_gap(blocks):
    """Paragraph gap learned from the data: the most common line step between body text lines, times 1.3."""
    gaps = Counter(round(b["top"] - a["top"]) for s in blocks for a, b in zip(s["items"], s["items"][1:])
                   if a["type"] == b["type"] == "text" and a["pdf_page"] == b["pdf_page"]
                   and a.get("top") is not None and b.get("top") is not None and b["top"] > a["top"])
    if not gaps:
        raise ValueError("no line positions ('top') in the blocks -> the extractor must keep each line's 'top'")
    return gaps.most_common(1)[0][0] * 1.3


# ---------------------------------------------------------------- text helpers (notebook 06, cell 3)
def join_lines(texts):
    """Visual lines -> running text. 'knowledge-' + 'based' -> 'knowledge-based'; otherwise a space."""
    out = ""
    for t in texts:
        t = t.strip()
        if not t:
            continue
        out = out + t if re.search(r"[A-Za-z]-$", out) else (out + " " + t).strip()
    return out


def split_sentences(text):
    """Split at '. ' / '? ' / '! ' before a capital or digit, then glue back pieces cut after an abbreviation."""
    parts = re.split(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])", text.strip())
    out = []
    for p in parts:
        if out and NO_END.search(out[-1]):
            out[-1] += " " + p
        else:
            out.append(p)
    return [s for s in out if s]


def hard_split(text, size):
    """Last resort for one sentence longer than HARD_MAX: cut between words."""
    words, out, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > size:
            out.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    return out + ([cur] if cur else [])


def glue_continuations(blocks):
    """A paragraph that starts lower-case (not a list item) right after a paragraph that does not end a
    sentence is the SAME sentence, broken by a page break or a big line gap -> join them back."""
    out = []
    for b in blocks:
        prev = out[-1] if out else None
        if (prev and b["kind"] == "para" and prev["kind"] == "para"
                and re.match(r"[a-z]", b["text"]) and not LIST_MARK.match(b["text"])
                and not re.search(r"[.?!]$", prev["text"])):
            out[-1] = {**prev, "text": join_lines([prev["text"], b["text"]]), "pages": prev["pages"] | b["pages"]}
        else:
            out.append(b)
    return out


# ---------------------------------------------------------------- block items -> text pieces (cell 7)
def line_blocks(items, gap):
    """Line items -> pieces in order: heading / para / list / table.
    A caption is glued to the piece after it. Diagram lines (PDF 32) become one list piece.
    gap = paragraph gap from para_gap()."""
    blocks, buf, buf_pages, prev = [], [], set(), None

    def flush():
        nonlocal buf, buf_pages
        if buf:
            blocks.append({"kind": "para", "text": join_lines(buf), "pages": set(buf_pages)})
        buf, buf_pages = [], set()

    for l in items:
        lv, t, n = l["type"], l["text"], l["pdf_page"]
        if lv == "heading" or LABEL_BODY.match(t.strip()):               # bare "Annexure" / "I" = part of a title
            flush(); blocks.append({"kind": "heading", "text": t.strip(), "pages": {n}})
        elif lv == "table":
            flush(); blocks.append({"kind": "table", "text": t.strip(), "pages": {n}})
        elif lv == "caption":
            flush(); blocks.append({"kind": "caption", "text": t.strip(), "pages": {n}})
        elif lv == "diagram":
            flush()
            if blocks and blocks[-1]["kind"] == "list" and blocks[-1].get("diagram"):
                blocks[-1]["text"] += "\n" + t.strip(); blocks[-1]["pages"].add(n)
            else:
                blocks.append({"kind": "list", "text": t.strip(), "pages": {n}, "diagram": True})
        else:                                                    # body (and anything else)
            new_para = (not buf or prev is None
                        or LIST_START.match(t.strip())
                        or (n != prev["pdf_page"] and re.search(r"[.:;?!]$", buf[-1]))
                        or (n == prev["pdf_page"] and l.get("top") is not None and prev.get("top") is not None
                            and not (0 < l["top"] - prev["top"] <= gap)))
            if new_para:
                flush()
            buf.append(t); buf_pages.add(n)
        prev = l
    flush()

    # glue each caption onto the block that follows it ("Exhibit 1: ..." + the 8 steps)
    out = []
    for b in blocks:
        if out and out[-1]["kind"] == "caption" and b["kind"] in ("para", "list"):
            cap = out.pop()
            b = {**b, "text": cap["text"] + "\n" + b["text"], "pages": cap["pages"] | b["pages"]}
        out.append(b)
    for b in out:                                                 # a caption with nothing after it -> plain para
        if b["kind"] == "caption":
            b["kind"] = "para"
    return glue_continuations(out)


def item_blocks(block, gap):
    """A block's items -> pieces: line items are joined into paragraphs; paragraph items are already paragraphs
    (their pages are the block's pages)."""
    if block["items"] and block["items"][0]["type"] == "paragraph":
        return glue_continuations([{"kind": "para", "text": item["text"], "pages": set(block["pdf_pages"])}
                                   for item in block["items"]])
    return line_blocks(block["items"], gap)


# ---------------------------------------------------------------- packing (cell 9)
def parts_size(parts):
    return sum(len(p) + 2 for p in parts) - 2 if parts else 0


def merge_small(chunks):
    """Tiny chunks (a leftover tail, 'Checked by / Verified by') join the chunk before them;
    a tiny FIRST chunk joins the one after it. Only when the result still fits HARD_MAX."""
    body = lambda c: c["parts"][1:] if c.get("overlap") else c["parts"]   # drop the repeated overlap sentence
    out = []
    for c in chunks:
        if out and parts_size(body(c)) < SMALL and parts_size(out[-1]["parts"]) + 2 + parts_size(body(c)) <= HARD_MAX:
            out[-1]["parts"] += body(c); out[-1]["pages"] |= c["pages"]
        else:
            out.append(c)
    if len(out) > 1 and parts_size(out[0]["parts"]) < SMALL and parts_size(out[0]["parts"]) + 2 + parts_size(body(out[1])) <= HARD_MAX:
        first, nxt = out.pop(0), out[0]
        out[0] = {**nxt, "parts": first["parts"] + body(nxt), "pages": first["pages"] | nxt["pages"],
                  "sub_heading": first["sub_heading"], "overlap": ""}
    return out


def pieces_of(block):
    """A block that fits TARGET stays whole; a longer one is cut into sentences (never inside one)."""
    if len(block["text"]) <= TARGET or block["kind"] == "list":
        return [block["text"]]
    out = []
    for s in split_sentences(block["text"]):
        out += hard_split(s, TARGET) if len(s) > HARD_MAX else [s]
    return out


def pack(blocks, keep_whole=True):
    """Blocks of ONE section/letter -> list of {"parts": [...], "pages": set, "sub_heading": str|None}."""
    body = [b for b in blocks if b["kind"] in ("para", "list")]
    if not body:
        return []                                                 # heading-only section -> no chunk
    if keep_whole and sum(len(b["text"]) for b in blocks if b["kind"] != "table") <= KEEP_WHOLE:
        keep = [b for b in blocks if b["kind"] != "table"]
        return [{"parts": [b["text"] for b in keep], "pages": set().union(*(b["pages"] for b in keep)),
                 "sub_heading": None, "overlap": ""}]

    chunks, heading, title_seen = [], None, False

    def start(overlap=""):
        return {"parts": [overlap] if overlap else [], "pages": set(), "sub_heading": heading,
                "overlap": overlap, "has_body": False}

    def size(c):
        return sum(len(p) + 2 for p in c["parts"])

    cur = start()
    for b in blocks:
        if b["kind"] == "table":
            continue                                               # tables get their own chunks (STC)
        if b["kind"] == "heading":
            is_title, title_seen = not title_seen, True            # first heading = the section's own title
            if not is_title:
                heading = b["text"]
                if cur["has_body"] and size(cur) >= MIN_CHUNK:     # sub-heading -> new chunk, no overlap
                    chunks.append(cur)
                    cur = start()
                elif not cur["has_body"]:
                    cur["sub_heading"] = heading
            cur["parts"].append(b["text"]); cur["pages"] |= b["pages"]
            continue
        title_seen = True
        for piece in pieces_of(b):
            if cur["has_body"] and size(cur) + len(piece) > TARGET:
                last = split_sentences(cur["parts"][-1])[-OVERLAP:] if OVERLAP else []
                ov = " ".join(last) if last and len(" ".join(last)) <= TARGET // 3 else ""
                chunks.append(cur)
                cur = start(ov if len(ov) + 2 + len(piece) <= HARD_MAX else "")   # no room -> skip the overlap
            cur["parts"].append(piece); cur["pages"] |= b["pages"]; cur["has_body"] = True
    if cur["has_body"]:
        chunks.append(cur)
    elif chunks and cur["parts"]:                                  # only a trailing heading left -> previous chunk
        chunks[-1]["parts"] += cur["parts"]; chunks[-1]["pages"] |= cur["pages"]
    return merge_small(chunks)


# ---------------------------------------------------------------- tables: STC, Guttal et al. 2026 (cell 12)
def table_rows(t):
    return [str(row).strip() for row in (t.get("texts") or []) if str(row).strip()]


def short_keys(header):
    """Line 1 (H), made short: drop the header level shared by every multi-level column and any (...) note."""
    levels = [[part.strip() for part in str(name).split(" - ")] for name in header]
    multi = [parts for parts in levels if len(parts) > 1]
    shared = set.intersection(*(set(parts) for parts in multi)) if len(multi) > 1 else set()
    clean = lambda part: re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", "", part)).strip()
    return {str(name): " - ".join(clean(part) for part in parts if part not in shared) or clean(parts[-1])
            for name, parts in zip(header, levels)}


def key_values(row, header):
    """Line 4: one row -> [(column, value)]; a cell belongs to the column whose header it starts with."""
    names = sorted({str(name) for name in header if name}, key=len, reverse=True)
    pairs = []
    for cell in row.split(" | "):
        name = next((name for name in names if cell.startswith(name + ":")), None)
        pairs.append((name, cell[len(name) + 1:].strip()) if name else (None, cell.strip()))
    return pairs


def render(pairs, short):
    return " | ".join(f"{short.get(name, name)}: {value}" if name else value for name, value in pairs)


def split_on_key_values(pairs, short):
    """Line 8: a row over budget -> pieces cut only between pairs; every piece repeats the row label (first pair)."""
    label, pieces, current = pairs[0], [], [pairs[0]]
    for pair in pairs[1:]:
        if len(current) > 1 and len(render(current + [pair], short)) > TABLE_BUDGET:
            pieces.append(current)
            current = [label]
        current.append(pair)
    return pieces + [current]


def table_groups(tables, captions):
    """Line 12 (context group): one table; its continuation on the next page with the same title + header joins it."""
    groups = []
    for t in tables:
        title = captions.get(t["id"]) or t.get("title") or f"Table {t['id']}"
        last = groups[-1] if groups else None
        if last and (last["title"], last["header"]) == (title, t.get("header") or []) and t["pdf_page"] == last["pages"][-1] + 1:
            last["tables"].append(t)
            last["pages"].append(t["pdf_page"])
        else:
            groups.append({"id": t["id"], "title": title, "header": t.get("header") or [], "kind": t.get("kind"),
                           "tables": [t], "pages": [t["pdf_page"]]})
    return groups


def table_leaves(group):
    """Lines 3-10: every row -> one key-value leaf (text, page), or several leaves if it is over budget."""
    short, leaves, keyed = short_keys(group["header"]), [], False
    for t in group["tables"]:
        for row in table_rows(t):
            pairs = [(None, row)] if group["kind"] == "2_col_list" else key_values(row, group["header"])
            keyed = keyed or any(name for name, value in pairs)
            pieces = [pairs] if len(render(pairs, short)) <= TABLE_BUDGET else split_on_key_values(pairs, short)
            leaves += [(render(piece, short), t["pdf_page"]) for piece in pieces]
    return leaves, keyed


def stc_chunks(group):
    """Lines 11-15: greedily batch the leaves of one table into chunks within the budget -> [(text, pages)]."""
    leaves, keyed = table_leaves(group)
    header_line = " | ".join(str(name) for name in group["header"] if name and not re.fullmatch(r"col\d+", str(name)))
    top = group["title"] + (f"\n{header_line}" if header_line and not keyed and group["kind"] != "2_col_list" else "")
    chunks, rows, pages = [], [], set()
    for text, page in leaves:
        if rows and len(top) + sum(len(row) + 1 for row in rows) + len(text) + 1 > TABLE_BUDGET:
            chunks.append(("\n".join([top] + rows), pages))
            rows, pages = [], set()
        rows.append(text)
        pages.add(page)
    return chunks + [("\n".join([top] + rows), pages or set(group["pages"]))] if rows or not chunks else chunks


# ---------------------------------------------------------------- chunk records (cell 14)
def record(chunk_id, doc_type, path, sub_heading, pages, text, source, printed, **extra):
    """printed = {pdf_page: printed_page} from pages.jsonl."""
    pages = sorted(pages)
    prefix = " > ".join(path) + (f" > {sub_heading}" if sub_heading else "")
    return {"chunk_id": chunk_id, "doc_type": doc_type, "section_id": extra.get("section_id"),
            "letter_no": extra.get("letter_no"), "path": path, "sub_heading": sub_heading,
            "pdf_pages": pages, "printed_pages": [printed.get(n) for n in pages], "source": source,
            "text": text, "embed_text": f"{prefix}\n\n{text}", "n_chars": len(text),
            "footnotes": extra.get("footnotes", [])}


def table_captions(block, gap):
    """Caption printed above a table ("Exhibit 2: ...") -> used as its title."""
    return {re.match(r"^\[table ([^\]]+)\]$", b["text"])[1]: b["caption"]
            for b in line_blocks(block["items"], gap) if b["kind"] == "table" and b.get("caption")}


def block_chunks(block, gap, printed):
    """One block -> its chunks: text chunks, then table chunks; a block that is not searchable -> one stub chunk.
    chunk ids start with the block_id; chunk_fields are copied onto every chunk unread."""
    fields = block["chunk_fields"]
    if not block["searchable"]:
        return [record(f"{block['block_id']}-01", f"{block['unit']}_stub", block["path"], None, block["pdf_pages"],
                       block["summary"], block["source"], printed, **fields)]
    out = []
    for k, c in enumerate(pack(item_blocks(block, gap)), 1):
        notes = [f for f in block["footnotes"] if f["pdf_page"] in c["pages"]]
        out.append(record(f"{block['block_id']}-{k:02d}", block["unit"], block["path"], c["sub_heading"], c["pages"],
                          "\n\n".join(c["parts"]), block["source"], printed, footnotes=notes, **fields))
    tables = [item["table"] for item in block["items"] if item["type"] == "table"]
    for group in table_groups(tables, table_captions(block, gap)):
        for k, (text, pages) in enumerate(stc_chunks(group), 1):
            out.append(record(f"T{group['id']}-{k:02d}", "table", block["path"], group["title"], pages,
                              text, "text", printed, **fields))
    return out


def build_chunks(blocks, pages):
    """All chunks, block by block in reading order."""
    printed = {p["pdf_page"]: p["printed_page"] for p in pages}
    gap = para_gap(blocks)
    return [c for block in blocks for c in block_chunks(block, gap, printed)]


if __name__ == "__main__":
    from src.blocks import read_blocks
    from src.paths import BLOCKS, CHUNKS, PAGES_FINAL
    chunks = build_chunks(read_blocks(BLOCKS), read_jsonl(PAGES_FINAL))
    print(f"{len(chunks)} chunks:", dict(Counter(c["doc_type"] for c in chunks)),
          "| same as chunks.jsonl:", chunks == read_jsonl(CHUNKS))