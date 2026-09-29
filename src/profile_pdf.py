import re
from collections import Counter
import pdfplumber
import pandas as pd
from pathlib import Path

def clutter_inventory(pdf_path, page_numbers, footer_top=780, footnote_zone=650,
                      superscript_max=7.5, footnote_size=8.5, repeat_min_pages=5):
    """Find and count every kind of clutter on the given pages. Nothing is removed."""
    inventory = {}                       # clutter_type -> {"count", "pages", "example"}
    line_seen_on = {}                    # line text    -> set of pages (for repeated lines)

    # ---------- the helper ----------
    def add(clutter_type, page_no, example_text):
        # create the entry the first time we see this clutter type
        if clutter_type not in inventory:
            inventory[clutter_type] = {"count": 0, "pages": set(), "example": None}
        inventory[clutter_type]["count"] += 1
        inventory[clutter_type]["pages"].add(page_no)
        if inventory[clutter_type]["example"] is None:
            inventory[clutter_type]["example"] = (page_no, example_text)

    with pdfplumber.open(pdf_path) as pdf:
        for page_no in page_numbers:
            page = pdf.pages[page_no - 1]

            # 1. side label: rotated characters, read top to bottom
            rotated = sorted((c for c in page.chars if not c["upright"]), key=lambda c: c["top"])
            side_label = "".join(c["text"] for c in rotated)
            if side_label:
                add("side label", page_no, side_label)

            # 2. page number: words below the footer line
            footer = " ".join(w["text"] for w in page.extract_words() if w["top"] > footer_top)
            if footer:
                add("page number", page_no, footer)

            # 3 & 4. character-level clutter
            for char in page.chars:
                if char["upright"] and char["size"] < superscript_max and char["top"] <= footer_top:
                    if char["text"].isdigit():
                        add("footnote marker", page_no, char["text"])     # health¹
                    else:
                        add("ordinal suffix", page_no, char["text"])      # 19ᵗʰ
                if "Wingdings" in char["fontname"]:
                    add("bullet glyph", page_no, char["text"])

            # 5 & 6. line-level clutter
            for line in page.extract_text_lines(return_chars=True):
                upright = [c for c in line["chars"] if c["upright"]]
                if not upright:
                    continue
                size = Counter(round(c["size"], 1) for c in upright).most_common(1)[0][0]
                text = line["text"].strip()

                if size <= footnote_size and footnote_zone < line["top"] <= footer_top:
                    add("footnote text", page_no, text[:60])
                if re.search(r"_{3,}|…{2,}|\.{4,}", text):
                    add("fill-in line", page_no, text[:60])
                if line["top"] <= footer_top:
                    line_seen_on.setdefault(text, set()).add(page_no)

    # 7. repeated lines: same text on many pages (running headers / footers)
    for text, pages in line_seen_on.items():
        if len(pages) >= repeat_min_pages:
            add("repeated line", min(pages), f"{text[:50]} (on {len(pages)} pages)")

    rows = [{"clutter_type": k, "count": v["count"], "n_pages": len(v["pages"]),
             "example": v["example"]} for k, v in inventory.items()]
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)

import pdfplumber

def pages_with_real_text(pdf_path, min_chars: int = 200, max_image_share: float = 0.5):
    """
    Return the PDF page numbers (1-based) that contain real, selectable text.

    A page counts if:
      - it has at least `min_chars` upright characters (the rotated side label is ignored), and
      - images cover less than `max_image_share` of the page (0.5 = half the page).
    """
    result = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            upright_chars = [c for c in page.chars if c["upright"]]
            n_chars = len(upright_chars)

            page_area = page.width * page.height
            image_area = sum(img["width"] * img["height"] for img in page.images)
            image_share = image_area / page_area

            if n_chars >= min_chars and image_share < max_image_share:
                result.append(page_no)
    return result

if __name__ == "__main__":
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    PDF_PATH = "../data/raw/Final Guideline on Human Resources for Health for NHM.pdf"

    real_pages = pages_with_real_text(PDF_PATH)
    clutter = clutter_inventory(PDF_PATH, real_pages)

    assert len(real_pages) == 118, f"expected 118 real pages, got {len(real_pages)}"
    print(f"✅ real pages: {len(real_pages)}")
    print(clutter[["clutter_type", "count", "n_pages"]])