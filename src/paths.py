"""Every file location in the project, in one place."""
from pathlib import Path

# This file lives in  D:\SwasthyaSetu\src\paths.py
# .parent = src\   → .parent.parent = D:\SwasthyaSetu  (the project root)
ROOT = Path(__file__).resolve().parent.parent

DATA_RAW  = ROOT / "data" / "raw"
PROFILE   = ROOT / "data" / "profile"
PROCESSED = ROOT / "data" / "processed"
CONFIG    = ROOT / "config"

PDF_PATH  = DATA_RAW / "Final Guideline on Human Resources for Health for NHM.pdf"
DEC_PATH  = CONFIG / "manual_decisions.json"