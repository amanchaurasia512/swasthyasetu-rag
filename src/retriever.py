"""SwasthyaSetu retriever: BM25 + vector search (ChromaDB) -> hybrid (RRF) -> rerank.

This is the search that won the step 23 evaluation (notebook 08): hit@5 0.984 on 185 questions.

Usage
-----
    from src.retriever import Retriever
    r = Retriever()                                   # loads chunks, BM25, ChromaDB
    for hit in r.search("Can a staff nurse take maternity leave?", k=5):
        print(hit["rank"], hit["chunk_id"], hit["path"], hit["pdf_pages"])

From a terminal:
    python -m src.retriever "Can a staff nurse take maternity leave?"
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

from src.paths import CHUNKS

# --- where things live (data/interim/chunks.jsonl -> project root is 2 folders up) ---
ROOT          = CHUNKS.parents[2]
VECTOR_DB_DIR = ROOT / "data" / "vectordb"
MODELS_DIR    = ROOT / "models"

# --- the settings that were tested in notebook 08 ---
EMBED_MODEL  = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
COLLECTION   = "hrh_chunks_bge_small"
HYBRID_POOL  = 50      # top-N taken from BM25 and from vector search before fusing
RRF_C        = 60      # reciprocal rank fusion: score = sum of 1 / (RRF_C + rank)
RERANK_POOL  = 20      # the reranker re-reads the hybrid's top-N

STOP = set("""a an the of to in on for and or is are was were be been by with as at from that this these those it its
what which who whom how why when where does do did should shall can could will would may might must about into than
there their they them his her he she we our you your i my me not no any all""".split())


def load_chunks(path: Path = CHUNKS) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def unit_of(chunk: dict) -> str:
    """The section a chunk belongs to ('S017'), or 'L05' for letter 5."""
    if chunk.get("letter_no") is not None:
        return f"L{int(chunk['letter_no']):02d}"
    return chunk["section_id"]


class Retriever:
    METHODS = ("bm25", "vector", "hybrid", "rerank")

    def __init__(self, chunks_path: Path = CHUNKS, db_dir: Path = VECTOR_DB_DIR,
                 models_dir: Path = MODELS_DIR, collection: str = COLLECTION):
        from rank_bm25 import BM25Okapi
        import snowballstemmer

        self.chunks      = load_chunks(chunks_path)
        self.row_of      = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}
        self.models_dir  = Path(models_dir)
        self.db_dir      = Path(db_dir)
        self.collection_name = collection
        self._stem       = snowballstemmer.stemmer("english").stemWord
        self._embedder   = None        # models are loaded on first use (lazy)
        self._reranker   = None

        self.bm25 = BM25Okapi([self.tokenize(c["embed_text"]) or ["_empty_"] for c in self.chunks])
        self.collection = self._open_collection()

    # ------------------------------------------------------------------ text → tokens (for BM25)
    def tokenize(self, text: str) -> list[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        return [self._stem(w) for w in words if w not in STOP]

    # ------------------------------------------------------------------ models, loaded only when needed
    @property
    def embedder(self):
        if self._embedder is None:
            from fastembed import TextEmbedding
            self._embedder = TextEmbedding(EMBED_MODEL, cache_dir=str(self.models_dir))
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            self._reranker = TextCrossEncoder(RERANK_MODEL, cache_dir=str(self.models_dir))
        return self._reranker

    # ------------------------------------------------------------------ vector database
    @staticmethod
    def _metadata(c: dict) -> dict:
        return {"unit": unit_of(c), "doc_type": c["doc_type"],
                "pdf_pages": ",".join(map(str, c["pdf_pages"])),
                "printed_pages": ",".join(str(p) for p in c["printed_pages"] if p is not None),
                "path": " > ".join(c["path"])}

    def _open_collection(self, rebuild: bool = False):
        """Open the ChromaDB collection; (re)build it if it is missing or does not match chunks.jsonl."""
        import chromadb
        client = chromadb.PersistentClient(path=str(self.db_dir))
        col = client.get_or_create_collection(name=self.collection_name, embedding_function=None,
                                              metadata={"hnsw:space": "cosine"})
        ids = [c["chunk_id"] for c in self.chunks]
        if rebuild or set(col.get(include=[])["ids"]) != set(ids):
            client.delete_collection(self.collection_name)
            col = client.create_collection(name=self.collection_name, embedding_function=None,
                                           metadata={"hnsw:space": "cosine"})
            vectors = np.array(list(self.embedder.passage_embed([c["embed_text"] for c in self.chunks])))
            col.add(ids=ids, embeddings=vectors.tolist(),
                    documents=[c["text"] for c in self.chunks],
                    metadatas=[self._metadata(c) for c in self.chunks])
        return col

    def rebuild_vector_db(self):
        """Call after chunks.jsonl changes (new chunking or new embedding model)."""
        self.collection = self._open_collection(rebuild=True)

    # ------------------------------------------------------------------ the four searches → [(row, score)]
    def _bm25(self, question: str, k: int):
        scores = self.bm25.get_scores(self.tokenize(question))
        top = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in top]

    def _vector(self, question: str, k: int):
        q = np.array(list(self.embedder.query_embed(question)))[0]
        res = self.collection.query(query_embeddings=[q.tolist()], n_results=k)
        return [(self.row_of[cid], 1 - dist) for cid, dist in zip(res["ids"][0], res["distances"][0])]

    def _hybrid(self, question: str, k: int):
        fused = {}
        for search in (self._bm25, self._vector):
            for rank, (i, _) in enumerate(search(question, HYBRID_POOL), 1):
                fused[i] = fused.get(i, 0) + 1 / (RRF_C + rank)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [(i, fused[i]) for i in top]

    def _rerank(self, question: str, k: int):
        cands  = [i for i, _ in self._hybrid(question, RERANK_POOL)]
        scores = np.array(list(self.reranker.rerank(question, [self.chunks[i]["embed_text"] for i in cands])))
        order  = np.argsort(-scores, kind="stable")[:k]
        return [(cands[j], float(scores[j])) for j in order]

    # ------------------------------------------------------------------ public
    def ranked(self, question: str, k: int = 5, method: str = "rerank"):
        """[(row number in self.chunks, score), ...] best first. Used by the evaluation notebook."""
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}, got {method!r}")
        return getattr(self, "_" + method)(question, k)

    def search(self, question: str, k: int = 5, method: str = "rerank") -> list[dict]:
        """Top-k chunks with everything needed to answer and cite."""
        hits = []
        for rank, (i, score) in enumerate(self.ranked(question, k, method), 1):
            c = self.chunks[i]
            hits.append({"rank": rank, "score": round(score, 4), "chunk_id": c["chunk_id"],
                         "unit": unit_of(c), "doc_type": c["doc_type"], "path": " > ".join(c["path"]),
                         "pdf_pages": c["pdf_pages"], "printed_pages": c["printed_pages"],
                         "text": c["text"]})
        return hits


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What should the State do if it has no NHM leave policy?"
    r = Retriever()
    print(question)
    for h in r.search(question, k=5):
        print(f"{h['rank']}. {h['chunk_id']:10} p.{h['printed_pages']}  {h['path'][:70]}")