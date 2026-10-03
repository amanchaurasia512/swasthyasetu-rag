
"""Every file location in the project, in one place."""
from pathlib import Path

# This file lives in D:\SwasthyaSetu\src\paths.py → two levels up = project root
ROOT = Path(__file__).resolve().parent.parent

# ---------- folders ----------
DATA_RAW  = ROOT / "data" / "raw"
PROFILE   = ROOT / "data" / "profile"
INTERIM   = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
CONFIG    = ROOT / "config"
QUALITY   = ROOT / "quality"
DEC_PATH  = CONFIG / "manual_decisions.json"
EVAL_DIR   = ROOT / "data" / "eval"
# ---------- inputs ----------
PDF_PATH     = DATA_RAW / "Final Guideline on Human Resources for Health for NHM.pdf"

PAGE_MAP_CSV = PROFILE / "page_map.csv"
TABLE_INV    = PROFILE / "table_inventory.csv"
TOC_CSV       = PROFILE / "toc.csv"      #created at step 15
PAGES_OCR    = INTERIM / "pages_ocr.jsonl"  
PAGE_TYPES_CSV = PROFILE / "classify_pdf_pages.csv"   # notebook 01 writes (after the rename)


# ---------- pipeline files: who writes → who reads ----------
PAGES_RAW    = INTERIM / "pages_raw.jsonl"      # step 12 writes → step 13 reads
PAGES_CLEAN  = INTERIM / "pages_clean.jsonl"    # step 13 writes → step 14 reads
PAGES_TABLES = INTERIM / "pages_tables.jsonl"   # step 14 writes → step 15 reads
TABLES_DIR   = INTERIM / "tables"               # step 14 writes one CSV per table
OCR_LINES    = INTERIM / "ocr_lines.jsonl"      #created at step 15 
LETTERS_JSONL = INTERIM / "letters.jsonl"       #created at step 15
PAGES_FINAL    = INTERIM / "pages.jsonl"        # step 18 writes → step 20 reads
SECTIONS     = INTERIM / "sections.jsonl"       #created at step 20
CHUNKS       = INTERIM / "chunks.jsonl"          #create at chunking step


#-------vecror databse-----------------
VECTOR_DB  = ROOT / "data" / "vectordb"   

#---------Model----------------------
MODELS_DIR    = ROOT / "models"