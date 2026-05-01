"""
rag_pipeline.py — Adaptive Hybrid RAG Pipeline
Sentence-based chunking + FAISS retrieval + conditional reranking.

Optimizations in this version:
  • Fast-mode defaults: faiss_k=5, bm25_k=3, max_chunks=3, use_reranker=False
  • Conditional reranking: only triggers when confidence is low
  • Hybrid scoring: 0.7 * rerank_score + 0.3 * faiss_score
  • Chunk trimming: caps context at ~600 chars for LLM efficiency
  • Definition-signal boosting in scoring
  • Penalty for overly long chunks (>900 chars)

Chunks stored as dicts: {"text": str, "source": str, "page": int}
"""

import os
import pickle
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME  = "BAAI/bge-small-en-v1.5"  # Better ML-domain semantics than all-MiniLM-L6-v2
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


# ── Chunk trimming ────────────────────────────────────────────────────────────

def trim_chunk(text: str, max_chars: int = 600) -> str:
    """
    Trim a chunk to max_chars, breaking at sentence boundary.
    Removes redundant whitespace and noisy fragments.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    # Try to break at sentence boundary
    truncated = text[:max_chars]
    last_period = truncated.rfind(". ")
    if last_period > max_chars * 0.5:
        return truncated[:last_period + 1].strip()
    # Fallback: break at word boundary
    last_space = truncated.rfind(" ")
    if last_space > 0:
        return truncated[:last_space].strip() + "…"
    return truncated.strip() + "…"


# ── Definition-signal scoring boost ──────────────────────────────────────────

_DEF_SIGNALS = {
    " is ", " is defined ", " is a ", " is an ",
    "defined as", "refers to", "algorithm is", "method is",
    "technique is", "can be defined", "we define",
}

def _compute_chunk_boost(chunk: dict) -> float:
    """
    Boost useful chunks, penalize overly long ones.
    Returns a score adjustment (can be negative).
    """
    text = chunk.get("text", "").lower()
    # Positive boost for definitional language
    bonus = sum(0.05 for sig in _DEF_SIGNALS if sig in text)
    bonus = min(bonus, 0.20)
    # Penalty for excessively long chunks (>900 chars)
    if len(text) > 900:
        bonus -= 0.10
    return bonus


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

    def retrieve(self, query: str, top_k: int = 4, score_threshold: float = 0.30) -> list[dict]:
        if self.index is None:
            self.load_index()
        q = _preprocess_retrieval_query(query)
        q_vec = self.model.encode([q], normalize_embeddings=True)
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
# Extends RAGPipeline with BM25 + conditional cross-encoder reranking.
#
# Retrieval stages:
#   1. FAISS semantic search    → top faiss_k candidates (dense)
#   2. BM25 keyword search      → top bm25_k candidates  (sparse)
#   3. Union + deduplicate      → by (source, page, text[:80]) fingerprint
#   4. Confidence check         → decide if reranking is needed
#   5. Cross-encoder reranking  → only when confidence is low
#   6. Hybrid scoring           → 0.7 * rerank + 0.3 * faiss (when reranked)
#   7. Return top max_chunks    → highest scores, hard-capped + trimmed

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
MAX_CHUNKS     = 3   # hard cap — reduced from 5 for faster LLM processing


def _bm25_tokenize(text: str) -> list[str]:
    """Lowercase BM25 tokens while preserving hyphenated ML terms."""
    return _re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


def _normalize_scores(values: list[float]) -> list[float]:
    """Min-max normalize scores to [0, 1] for comparable hybrid scoring."""
    if not values:
        return []
    min_s = min(values)
    max_s = max(values)
    rng = max_s - min_s
    if len(values) == 1:
        return [1.0]
    if rng < 1e-8:
        return [1.0] * len(values)
    return [(v - min_s) / rng for v in values]


def _preprocess_retrieval_query(query: str) -> str:
    """Normalize noisy phrasing before dense+sparse retrieval."""
    q = query or ""
    contractions = {
        r"\bwhat's\b": "what is",
        r"\bit's\b": "it is",
        r"\bcan't\b": "cannot",
        r"\bdon't\b": "do not",
    }
    for pattern, replacement in contractions.items():
        q = re.sub(pattern, replacement, q, flags=re.IGNORECASE)
    q = re.sub(r"\blike (i'?m|i am) \d+\b", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\b(idk|lol|tbh|ngl)\b", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\s+", " ", q).strip()
    return q or query


def needs_rerank(chunks: list[dict]) -> bool:
    """
    Decide if reranking is needed using combined confidence signals.
    Handles both weak retrieval (low absolute score) and
    ambiguous retrieval (small gap between top-2 candidates).
    """
    if len(chunks) < 2:
        return True
    # Weak retrieval: top score too low
    if chunks[0].get("score", 0.0) < 0.65:
        return True
    # Ambiguous retrieval: top-2 scores too close
    score_diff = chunks[0].get("score", 0.0) - chunks[1].get("score", 0.0)
    return score_diff < 0.05


class HybridRAGPipeline(RAGPipeline):
    """
    Adaptive Hybrid RAG Pipeline — drop-in upgrade over RAGPipeline.
    Uses the same FAISS index + chunks file — no re-ingestion needed.

    Key optimization: conditional reranking.
    - Fast mode (default): FAISS + BM25 only → ~1-2s retrieval
    - Rerank mode (auto):  adds CrossEncoder when confidence is low → ~3-4s

    Extra kwargs in __init__:
        faiss_k  : candidates to pull from FAISS (default 5)
        bm25_k   : candidates to pull from BM25  (default 3)
        max_chunks: final chunks returned after scoring (default 3)
        use_reranker: allow reranking when needed (default True)
    """

    def __init__(
        self,
        faiss_k:      int  = 5,    # reduced from 10 for speed
        bm25_k:       int  = 3,    # reduced from 10 for speed
        max_chunks:   int  = MAX_CHUNKS,
        use_reranker: bool = True,  # allows reranking, doesn't force it
    ):
        super().__init__()          # loads SentenceTransformer(MODEL_NAME)
        self.faiss_k      = faiss_k
        self.bm25_k       = bm25_k
        self.max_chunks   = max_chunks
        self.use_reranker = use_reranker and _RERANKER_AVAILABLE
        self.min_final_score = 0.30
        self.min_semantic_score = 0.30

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
        score_threshold: float = 0.30,
        level:           str   = "",     # "Beginner" | "Intermediate" | "Advanced"
    ) -> list[dict]:
        """
        Adaptive Hybrid Retrieval:
          FAISS (dense) + BM25 (sparse) → confidence check → conditional rerank.

        Fast path (~1-2s): Skip reranker when top FAISS score is confident.
        Full path (~3-4s): Apply reranker only when retrieval confidence is low.

        The top_k parameter is preserved for API compatibility with existing callers
        (app.py, rag_eval.py). Internally, faiss_k and bm25_k control candidate
        breadth, and max_chunks caps the final output.

        Level-aware adaptation:
          Beginner:     faiss_k=4, bm25_k=2, rerank only if needed
          Intermediate: uses instance defaults
          Advanced:     faiss_k=6, bm25_k=4, rerank when sparse or ambiguous
        """
        # ── Level-aware adaptation (local vars — never mutate self) ───
        faiss_k = self.faiss_k
        bm25_k  = self.bm25_k

        if level == "Advanced":
            faiss_k = 6
            bm25_k  = 4
        elif level == "Beginner":
            faiss_k = 4
            bm25_k  = 2

        if self.index is None:
            self.load_index()
        q = _preprocess_retrieval_query(query)

        # ── Stage 1: FAISS dense retrieval ───────────────────────────────
        q_vec = self.model.encode([q], normalize_embeddings=True)
        scores_arr, indices_arr = self.index.search(
            np.array(q_vec, dtype="float32"), faiss_k
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
        bm25_hits = self._bm25_retrieve(q, bm25_k)

        # ── Stage 3: Union + deduplicate ─────────────────────────────────
        # FAISS first → in case of duplicate, FAISS score is kept
        candidates = self._dedup(faiss_hits + bm25_hits)

        if not candidates:
            return []

        # ── Stage 4: Confidence check — conditional reranking ────────────
        # Advanced: rerank only when retrieval is sparse or ambiguous
        # Other levels: rerank only when confidence is low
        if level == "Advanced":
            force_rerank = len(candidates) < 3 or needs_rerank(candidates)
        else:
            force_rerank = False

        should_rerank = self.use_reranker and (force_rerank or needs_rerank(candidates))

        if should_rerank:
            reranker = self._get_reranker()
            if reranker is not None and len(candidates) > 1:
                # Narrow to top 5 before reranking (cost control)
                candidates = candidates[:5]
                pairs  = [(q, c["text"]) for c in candidates]
                re_scores = reranker.predict(pairs)    # numpy array

                for chunk, re_score in zip(candidates, re_scores):
                    chunk["rerank_score"] = round(float(re_score), 4)

                # ── Stage 5: Normalized hybrid scoring ───────────────────
                # Normalize both score types to [0,1] before combining,
                # since FAISS cosine-sim and cross-encoder logits differ.
                faiss_vals  = [c.get("score", 0.0) for c in candidates]
                rerank_vals = [c.get("rerank_score", 0.0) for c in candidates]
                norm_faiss  = _normalize_scores(faiss_vals)
                norm_rerank = _normalize_scores(rerank_vals)

                for chunk, nf, nr in zip(candidates, norm_faiss, norm_rerank):
                    chunk["final_score"] = round(0.7 * nr + 0.3 * nf, 4)

                candidates.sort(
                    key=lambda c: c.get("final_score", 0.0), reverse=True
                )
        else:
            # Fast path: apply definition-signal boost + length penalty
            for chunk in candidates:
                base = chunk.get("score", 0.0)
                boost = _compute_chunk_boost(chunk)
                chunk["final_score"] = round(base + boost, 4)

            candidates.sort(
                key=lambda c: c.get("final_score", 0.0), reverse=True
            )

        # ── Stage 5.5: Out-of-scope guard ────────────────────────────────
        # Reject weak/BM25-only retrieval so app can abstain cleanly.
        top_final = float(candidates[0].get("final_score", 0.0))
        top_semantic = max(float(c.get("score", 0.0) or 0.0) for c in candidates)
        if top_final < self.min_final_score or top_semantic < self.min_semantic_score:
            return []

        # ── Stage 6: Hard cap + trim chunks ──────────────────────────────
        final = candidates[:self.max_chunks]

        # Keep full ingest chunk size (ingest enforces 1200-char ceiling).
        for chunk in final:
            chunk["text"] = trim_chunk(chunk["text"], max_chars=1200)

        return final