
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
# ---------- inputs ----------
PDF_PATH     = DATA_RAW / "Final Guideline on Human Resources for Health for NHM.pdf"
DEC_PATH     = CONFIG / "manual_decisions.json"
PAGE_MAP_CSV = PROFILE / "page_map.csv"
TABLE_INV    = PROFILE / "table_inventory.csv"

# ---------- pipeline files: who writes → who reads ----------
PAGES_RAW    = INTERIM / "pages_raw.jsonl"      # step 12 writes → step 13 reads
PAGES_CLEAN  = INTERIM / "pages_clean.jsonl"    # step 13 writes → step 14 reads
PAGES_TABLES = INTERIM / "pages_tables.jsonl"   # step 14 writes → step 15 reads
TABLES_DIR   = INTERIM / "tables"               # step 14 writes one CSV per table