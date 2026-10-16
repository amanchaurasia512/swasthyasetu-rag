"""Settings from YAML: base -> document -> experiment -> command line (later wins).

Usage
-----
    from src.config import load
    config = load()                                        # base + the NHM document
    config = load("config/experiments/e3.yaml")            # + one experiment (its 'docs' picks the document)
    config["gate"]["lower"], config["files"]["chunks"]

API keys never live here (they stay in .env), and API calls are never switched on by a file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from src.paths import CONFIG, ROOT

DEFAULT_DOC = "nhm_hrh_2022"


def read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def merge(base: dict, override: dict) -> dict:
    """A copy of base with override on top; nested dicts are merged key by key, anything else is replaced."""
    merged = dict(base)
    for key, value in override.items():
        both_dicts = isinstance(value, dict) and isinstance(merged.get(key), dict)
        merged[key] = merge(merged[key], value) if both_dicts else value
    return merged


def load(experiment: str | Path | None = None, overrides: dict | None = None) -> dict:
    """The merged settings for one run."""
    run = read_yaml(ROOT / experiment) if experiment else {}
    doc_id = (run.get("docs") or [DEFAULT_DOC])[0]
    config = merge(read_yaml(CONFIG / "base.yaml"), read_yaml(CONFIG / "docs" / f"{doc_id}.yaml"))
    return merge(merge(config, run), overrides or {})


def config_hash(config: dict) -> str:
    """Same hash = same settings (key order does not matter)."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def path_of(relative: str) -> Path:
    """A path written in the config (relative to the project root) -> an absolute path."""
    return ROOT / relative