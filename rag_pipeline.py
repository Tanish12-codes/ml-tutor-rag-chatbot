"""
rag_pipeline.py
Sentence-based chunking + FAISS retrieval.
Chunks stored as dicts: {"text": str, "source": str, "page": int}
"""

import os
import pickle
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME  = "all-MiniLM-L6-v2"
INDEX_PATH  = "faiss_index.bin"
CHUNKS_PATH = "chunks.pkl"


# ── Sentence splitting ────────────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """
    Split on sentence-ending punctuation followed by whitespace.
    Avoids splitting on abbreviations like "e.g." or "Fig. 3".
    """
    # Protect known abbreviations from being split
    text = re.sub(r"\b(e\.g|i\.e|vs|Fig|fig|Eq|eq|et al|approx|Prof|Dr)\.", r"\1<DOT>", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    # Restore protected dots
    return [s.replace("<DOT>", ".") for s in sentences]


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 5,   # sentences per chunk  (was 2 — root cause of 25k)
    overlap: int = 1,      # sentence overlap between chunks
) -> list[str]:
    """
    Groups sentences into chunks of ~5 sentences each.
    chunk_size=5, overlap=1 → step=4 → ~4x fewer chunks than chunk_size=2,overlap=1.
    Returns plain strings; ingest.py wraps them in metadata dicts.
    """
    sentences = [s.strip() for s in split_sentences(text) if len(s.strip()) > 15]
    if not sentences:
        return []

    step   = max(1, chunk_size - overlap)
    chunks = []

    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i : i + chunk_size])
        # Keep chunks that are substantive but not too long
        if 80 < len(chunk) < 1200:
            chunks.append(chunk)

    return chunks


# ── RAG Pipeline ──────────────────────────────────────────────────────────────

class RAGPipeline:
    def __init__(self):
        self.model  = SentenceTransformer(MODEL_NAME)
        self.index  = None
        self.chunks : list[dict] = []  # {"text", "source", "page"}

    def build_index_from_chunks(self, chunks: list[dict]):
        if not chunks:
            raise ValueError("No chunks provided.")

        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        print("Embedding chunks...")
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=64,
        )

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(np.array(embeddings, dtype="float32"))

        faiss.write_index(self.index, INDEX_PATH)
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(self.chunks, f)

        print(f"✅ Index built — {len(chunks)} chunks")

    def load_index(self):
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError("Run ingest.py first.")
        self.index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)

    def retrieve(self, query: str, top_k: int = 4, score_threshold: float = 0.20) -> list[dict]:
        if self.index is None:
            self.load_index()
        q_vec = self.model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(np.array(q_vec, dtype="float32"), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:                          # FAISS padding — no real result
                continue
            if idx >= len(self.chunks):            # stale index / corrupt state
                continue
            if float(score) < score_threshold:     # below caller-defined floor
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = round(float(score), 4)
            results.append(chunk)
        # FAISS IndexFlatIP already returns in score order, but sort explicitly
        # so callers never depend on FAISS internals staying stable.
        results.sort(key=lambda c: c["score"], reverse=True)
        return results

# ── Hybrid RAG Pipeline ───────────────────────────────────────────────────────
# Extends RAGPipeline with BM25 + cross-encoder reranking.
# The base class (and all existing code that uses it) is UNCHANGED.
#
# Retrieval stages:
#   1. FAISS semantic search    → top faiss_k candidates (dense)
#   2. BM25 keyword search      → top bm25_k candidates  (sparse)
#   3. Union + deduplicate      → by (source, page, text[:80]) fingerprint
#   4. Cross-encoder reranking  → score every candidate against query
#   5. Return top max_chunks    → highest reranker scores, hard-capped

import re as _re

try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder as _CrossEncoder
    _RERANKER_AVAILABLE = True
except ImportError:
    _RERANKER_AVAILABLE = False

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_CHUNKS     = 5   # hard cap — prevents context overload


def _bm25_tokenize(text: str) -> list[str]:
    """Lowercase + alphanumeric tokens only. Fast, deterministic."""
    return _re.findall(r"[a-z0-9]+", text.lower())


class HybridRAGPipeline(RAGPipeline):
    """
    Drop-in upgrade over RAGPipeline.
    Uses the same FAISS index + chunks file — no re-ingestion needed.

    Extra kwargs in __init__:
        faiss_k  : candidates to pull from FAISS (default 10)
        bm25_k   : candidates to pull from BM25  (default 10)
        max_chunks: final chunks returned after reranking (default 5)
        use_reranker: whether to run cross-encoder (default True)
    """

    def __init__(
        self,
        faiss_k:      int  = 10,
        bm25_k:       int  = 10,
        max_chunks:   int  = MAX_CHUNKS,
        use_reranker: bool = True,
    ):
        super().__init__()          # loads SentenceTransformer(MODEL_NAME)
        self.faiss_k      = faiss_k
        self.bm25_k       = bm25_k
        self.max_chunks   = max_chunks
        self.use_reranker = use_reranker and _RERANKER_AVAILABLE

        self._bm25:     object | None = None
        self._reranker: object | None = None

    # ── BM25 index build ─────────────────────────────────────────────────────

    def _build_bm25(self) -> None:
        """Build BM25 index lazily from self.chunks. Called once after load."""
        if not _BM25_AVAILABLE:
            return
        tokenized = [_bm25_tokenize(c["text"]) for c in self.chunks]
        self._bm25 = _BM25Okapi(tokenized)

    def load_index(self) -> None:
        """Load FAISS index then build BM25 in-memory."""
        super().load_index()
        self._build_bm25()

    def build_index_from_chunks(self, chunks: list[dict]) -> None:
        """Build FAISS index then build BM25 in-memory."""
        super().build_index_from_chunks(chunks)
        self._build_bm25()

    # ── Reranker lazy-load ────────────────────────────────────────────────────

    def _get_reranker(self):
        if self._reranker is None and self.use_reranker:
            print(f"Loading reranker ({RERANKER_MODEL})...")
            self._reranker = _CrossEncoder(RERANKER_MODEL)
        return self._reranker

    # ── BM25 retrieval ────────────────────────────────────────────────────────

    def _bm25_retrieve(self, query: str, top_k: int) -> list[dict]:
        if self._bm25 is None:
            return []
        tokens  = _bm25_tokenize(query)
        scores  = self._bm25.get_scores(tokens)
        top_idx = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_idx:
            if scores[idx] <= 0:        # BM25 score=0 → no keyword overlap
                continue
            chunk = dict(self.chunks[idx])
            chunk["bm25_score"] = round(float(scores[idx]), 4)
            results.append(chunk)
        return results

    # ── Deduplication ─────────────────────────────────────────────────────────

    @staticmethod
    def _dedup(candidates: list[dict]) -> list[dict]:
        """
        Deduplicate by (source, page, first-80-chars).
        Keeps the first occurrence (FAISS results come first → higher priority).
        """
        seen: set = set()
        out:  list = []
        for c in candidates:
            key = (c.get("source"), c.get("page"), c.get("text", "")[:80])
            if key not in seen:
                seen.add(key)
                out.append(c)
        return out

    # ── Main retrieve (replaces parent) ──────────────────────────────────────

    def retrieve(
        self,
        query:           str,
        top_k:           int   = 4,      # kept for API compatibility; ignored internally
        score_threshold: float = 0.20,
    ) -> list[dict]:
        """
        Hybrid retrieval: FAISS (dense) + BM25 (sparse) → cross-encoder rerank.

        The top_k parameter is preserved for API compatibility with existing callers
        (app.py, rag_eval.py). Internally, faiss_k and bm25_k control candidate
        breadth, and max_chunks caps the final output.
        """
        if self.index is None:
            self.load_index()

        # ── Stage 1: FAISS dense retrieval ───────────────────────────────
        q_vec = self.model.encode([query], normalize_embeddings=True)
        scores_arr, indices_arr = self.index.search(
            np.array(q_vec, dtype="float32"), self.faiss_k
        )
        faiss_hits: list[dict] = []
        for score, idx in zip(scores_arr[0], indices_arr[0]):
            if idx == -1 or idx >= len(self.chunks):
                continue
            if float(score) < score_threshold:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"]      = round(float(score), 4)
            chunk["bm25_score"] = 0.0
            faiss_hits.append(chunk)

        # ── Stage 2: BM25 sparse retrieval ───────────────────────────────
        bm25_hits = self._bm25_retrieve(query, self.bm25_k)

        # ── Stage 3: Union + deduplicate ─────────────────────────────────
        # FAISS first → in case of duplicate, FAISS score is kept
        candidates = self._dedup(faiss_hits + bm25_hits)

        if not candidates:
            return []

        # ── Stage 4: Cross-encoder reranking ─────────────────────────────
        reranker = self._get_reranker()
        if reranker is not None and len(candidates) > 1:
            pairs  = [(query, c["text"]) for c in candidates]
            re_scores = reranker.predict(pairs)    # numpy array

            for chunk, re_score in zip(candidates, re_scores):
                chunk["rerank_score"] = round(float(re_score), 4)

            candidates.sort(key=lambda c: c.get("rerank_score", 0.0), reverse=True)
        else:
            # No reranker — fall back to FAISS score ordering
            candidates.sort(key=lambda c: c.get("score", 0.0), reverse=True)

        # ── Stage 5: Hard cap ─────────────────────────────────────────────
        return candidates[: self.max_chunks]