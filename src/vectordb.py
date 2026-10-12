"""SwasthyaSetu vector database (ChromaDB): check it, build it, open it.

The database holds one bge-small vector per chunk of data/interim/chunks.jsonl. Each record also stores a
fingerprint of the chunk's embed_text, so a changed chunk is noticed even when its chunk_id stays the same.

From a terminal (project root):
    python -m src.vectordb              # status: is the database in step with chunks.jsonl?
    python -m src.vectordb --rebuild    # embed every chunk again

In code:
    from src.vectordb import load_chunks, status, open_collection
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from src.paths import CHUNKS

ROOT          = CHUNKS.parents[2]
VECTOR_DB_DIR = ROOT / "data" / "vectordb"
MODELS_DIR    = ROOT / "models"
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"
COLLECTION    = "hrh_chunks_bge_small"
BATCH         = 64          # chunks embedded and added per step


def load_chunks(path: Path = CHUNKS) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def unit_of(chunk: dict) -> str:
    """The section a chunk belongs to ('S017'), or 'L05' for letter 5."""
    if chunk.get("letter_no") is not None:
        return f"L{int(chunk['letter_no']):02d}"
    return chunk["section_id"]


def fingerprint(chunk: dict) -> str:
    return hashlib.md5(chunk["embed_text"].encode("utf-8")).hexdigest()


def metadata(chunk: dict) -> dict:
    return {"unit": unit_of(chunk), "doc_type": chunk["doc_type"],
            "pdf_pages": ",".join(map(str, chunk["pdf_pages"])),
            "printed_pages": ",".join(str(p) for p in chunk["printed_pages"] if p is not None),
            "path": " > ".join(chunk["path"]), "fingerprint": fingerprint(chunk)}


def load_embedder(models_dir: Path = MODELS_DIR):
    from fastembed import TextEmbedding
    return TextEmbedding(EMBED_MODEL, cache_dir=str(models_dir))


def _client(db_dir: Path):
    import chromadb
    return chromadb.PersistentClient(path=str(db_dir))


def status(chunks: list[dict], db_dir: Path = VECTOR_DB_DIR, collection: str = COLLECTION) -> dict:
    """Compare the database with the chunks: what is missing, extra or changed."""
    client = _client(db_dir)
    if collection not in [c.name for c in client.list_collections()]:
        stored = {}
    else:
        saved = client.get_collection(collection).get(include=["metadatas"])
        stored = {cid: (meta or {}).get("fingerprint") for cid, meta in zip(saved["ids"], saved["metadatas"])}
    wanted = {c["chunk_id"]: fingerprint(c) for c in chunks}
    report = {"chunks": len(wanted), "in_db": len(stored),
              "missing": sorted(set(wanted) - set(stored)),
              "extra": sorted(set(stored) - set(wanted)),
              "changed": sorted(cid for cid in set(wanted) & set(stored) if stored[cid] != wanted[cid])}
    report["up_to_date"] = not (report["missing"] or report["extra"] or report["changed"])
    return report


def build(chunks: list[dict], embedder, db_dir: Path = VECTOR_DB_DIR, collection: str = COLLECTION):
    """Delete the collection and embed every chunk again. Returns the new collection."""
    client = _client(db_dir)
    if collection in [c.name for c in client.list_collections()]:
        client.delete_collection(collection)
    col = client.create_collection(name=collection, embedding_function=None, metadata={"hnsw:space": "cosine"})
    start = time.time()
    for first in range(0, len(chunks), BATCH):
        batch = chunks[first:first + BATCH]
        vectors = list(embedder.passage_embed([c["embed_text"] for c in batch]))
        col.add(ids=[c["chunk_id"] for c in batch], embeddings=[list(map(float, v)) for v in vectors],
                documents=[c["text"] for c in batch], metadatas=[metadata(c) for c in batch])
        print(f"  embedded {min(first + BATCH, len(chunks))}/{len(chunks)}", end="\r")
    print(f"\nvector DB built: {col.count()} chunks in {time.time() - start:.0f} s → {db_dir}")
    if col.count() != len(chunks):
        raise RuntimeError(f"vector DB has {col.count()} records, chunks.jsonl has {len(chunks)}")
    return col


def open_collection(chunks: list[dict], get_embedder, db_dir: Path = VECTOR_DB_DIR,
                    collection: str = COLLECTION, rebuild: bool = False):
    """Open the collection; build it first if asked or if it is not in step with the chunks.
    get_embedder is a function, so the model is loaded only when a build is needed."""
    report = status(chunks, db_dir, collection)
    if rebuild or not report["up_to_date"]:
        reason = "asked" if rebuild else (f"{len(report['missing'])} missing, {len(report['extra'])} extra, "
                                          f"{len(report['changed'])} changed")
        print(f"rebuilding vector DB ({reason})")
        return build(chunks, get_embedder(), db_dir, collection)
    return _client(db_dir).get_collection(collection)


def self_check(col, chunks: list[dict], embedder, n: int = 5) -> bool:
    """Each of n chunks, searched with its own text, must come back as the top hit."""
    picks = chunks[:: max(len(chunks) // n, 1)][:n]
    vectors = list(embedder.query_embed([c["embed_text"] for c in picks]))
    found = col.query(query_embeddings=[list(map(float, v)) for v in vectors], n_results=1)["ids"]
    misses = [c["chunk_id"] for c, ids in zip(picks, found) if ids[0] != c["chunk_id"]]
    print(f"self-check: {n - len(misses)}/{n} chunks find themselves" + (f" · misses {misses}" if misses else ""))
    return not misses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check or rebuild the SwasthyaSetu vector database")
    parser.add_argument("--rebuild", action="store_true", help="embed every chunk again")
    args = parser.parse_args()
    chunks = load_chunks()
    report = status(chunks)
    print({key: (value if not isinstance(value, list) else len(value)) for key, value in report.items()})
    if args.rebuild or not report["up_to_date"]:
        embedder = load_embedder()
        col = open_collection(chunks, lambda: embedder, rebuild=args.rebuild)
        self_check(col, chunks, embedder)