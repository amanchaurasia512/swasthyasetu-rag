"""Text helpers shared by all notebooks."""
import re

CHAR_MAP = {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-",
            "√": "Yes", "\u00a0": " ", "\u200b": ""}

def normalize_chars(text):
    if not text:
        return text
    text = re.sub(r"^\s*•\s*", "- ", text)       # bullet at line start
    text = text.replace("•", "-")
    for old, new in CHAR_MAP.items():
        text = text.replace(old, new)
    return re.sub(r"\s{2,}", " ", text).strip()