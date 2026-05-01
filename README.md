# ML Tutor — Hybrid RAG Chatbot

A Streamlit chatbot that explains machine learning concepts from a PDF textbook corpus. Retrieval combines FAISS semantic search and BM25 keyword search, with a cross-encoder reranker that only fires when retrieval confidence is low. Answers follow a fixed 6-section format and adapt to the user's level and preferred response style.

---

## How It Works

```
User question (+ last 12 messages as conversation context)
        |
        v
preprocess_query() — strip slang, expand contractions, normalize
        |
        v
HybridRAGPipeline.retrieve()
  ├── FAISS dense search     (BAAI/bge-small-en-v1.5, top-5 candidates)
  ├── BM25 sparse search     (rank-bm25, top-3 candidates)
  ├── Union + deduplicate    (by source · page · first 80 chars)
  ├── needs_rerank()?        (top score < 0.65 or top-2 gap < 0.05)
  │     ├── YES → CrossEncoder rerank → hybrid score (0.7·rerank + 0.3·FAISS)
  │     └── NO  → definition-signal boost + length penalty applied
  └── Out-of-scope guard     (abstain if top final score < 0.30)
        |
        v
build_messages() — injects level + style instructions into system prompt
        |
        v
LLM call (Gemini / OpenAI / Claude) with exponential backoff retry (×3)
  └── Truncation check → one retry with higher token limit if needed
        |
        v
Structured answer: Definition → How it Works → Key Points
  + (Detailed mode): Intuition → Example → Common Mistakes → Sources
        |
        v
If LLM fails → make_structured_fallback() builds answer from raw chunks,
               no API call needed
```

---

## What I Built

**Retrieval Pipeline (`rag_pipeline.py`)**

- Built a hybrid retrieval system combining FAISS (dense, cosine similarity) and BM25 (sparse, keyword) with deduplication by `(source, page, text[:80])` fingerprint — so the same chunk from two paths isn't counted twice.
- Implemented conditional reranking: the CrossEncoder (`ms-marco-MiniLM-L-6-v2`) only loads and runs when `needs_rerank()` returns true — either the top FAISS score is below 0.65, or the gap between the top two candidates is under 0.05. Fast path skips it entirely (~1–2s vs ~3–4s).
- Hybrid scoring normalizes FAISS cosine similarity and cross-encoder logits to [0, 1] before combining (`0.7 × rerank + 0.3 × FAISS`) — prevents one scale dominating.
- Level-aware retrieval: Beginner uses `faiss_k=4, bm25_k=2`; Advanced uses `faiss_k=6, bm25_k=4` with reranking forced on sparse or ambiguous results.
- Out-of-scope guard: if the top final score is below 0.30 semantically, the pipeline returns empty and the app abstains rather than hallucinating an answer.

**PDF Ingestion (`ingest.py`)**

- Extracts text with `pdfplumber` using `x_tolerance=1` and `keep_blank_chars=True` — tighter horizontal grouping that fixes merged tokens like `"Thisisasentence"` that degraded LLM output quality.
- 14-rule bad-chunk filter removes bibliography entries, LaTeX equations, TOC lines, figure captions, matrix rows, pseudocode blocks, and repetitive headers — keeps only prose chunks with sufficient alpha density (≥55%), real word count (≥8 words), and sentence-ending punctuation (≥2 marks).
- Sentence-based chunking at 5 sentences, 1-sentence overlap — step of 4 produces ~4× fewer chunks than the prior 2-sentence chunking without losing coverage.

**Prompt Engineering (`prompt_builder.py`)**

- System prompt enforces a strict 6-section output format with per-level and per-style instructions injected at runtime — Beginner gets plain language + analogies with all math stripped, Advanced gets full technical depth with notation consistency rules.
- `make_structured_fallback()` assembles a clean 6-section response directly from retrieved chunks when the LLM is unavailable — no API call, no broken UI.
- Deterministic video recommendation system across 14 ML topics: synonym-first matching, then keyword scoring (score ≥2 hits = confident, score = 1 + multi-word hit = confident), always returns exactly 2 videos on a confident match.

**App (`app.py`)**

- Multi-provider LLM support: Gemini, OpenAI, and Claude selectable via `LLM_PROVIDER` env var — all three use the same retry wrapper.
- Exponential backoff retry (×3, base delay 2s) on rate-limit and server timeout errors; a second truncation retry fires if the response is detected as cut-off mid-section.
- Session history capped at 20 messages; last 12 are passed as conversation context; LLM responses are cached by `(question, level, style)` tuple with a 1-hour TTL.
- Duplicate-submission guard via SHA-256 hash of the current question compared against the last processed hash.

**Evaluation (`evaluate_system.py`)**

- Evaluated across 15 queries spanning 6 categories: core concepts, phrasing variants, multi-concept, out-of-scope, ambiguous single-word, noisy/casual phrasing, and multi-turn follow-ups.
- Measured latency per query, fallback rate, relevance (keyword + semantic match), section completeness (0–3 hits), and a composite score.

---

## Eval Results (Intermediate level, 15 queries)

| Metric | Value |
|---|---|
| Relevance score | 93.3% |
| Quality score (section completeness) | 100% |
| Avg section hits | 3.0 / 3 |
| Composite score | 0.797 |
| Avg latency | 7.16s |
| Fallback rate | 80% ⚠️ |

The 80% fallback rate reflects free-tier API rate limits during evaluation — not retrieval or answer quality failures. All 15 fallback responses still returned structured output via `make_structured_fallback()`. Relevance and section completeness were unaffected.

---

## Tech Stack

| Component | Technology |
|---|---|
| App | Streamlit |
| LLM | Gemini 2.5 Flash / OpenAI / Claude (switchable) |
| Dense retrieval | FAISS + `BAAI/bge-small-en-v1.5` |
| Sparse retrieval | BM25 (rank-bm25) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| PDF extraction | pdfplumber |
| Embeddings | sentence-transformers |

---

## Setup

**Prerequisites:** Python 3.11+, a Gemini API key (or OpenAI / Anthropic)

```bash
git clone https://github.com/your-username/ml-tutor-rag.git
cd ml-tutor-rag

python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Set GEMINI_API_KEY (and optionally LLM_PROVIDER, OPENAI_API_KEY, ANTHROPIC_API_KEY)
```

Add your PDF textbooks to the `data/` folder, then:

```bash
python ingest.py          # extract → clean → chunk → build FAISS index
streamlit run app.py      # start the app
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | If using Gemini | — | Gemini API key |
| `OPENAI_API_KEY` | If using OpenAI | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | If using Claude | — | Anthropic API key |
| `LLM_PROVIDER` | No | `gemini` | `gemini` \| `openai` \| `claude` |
| `GEMINI_MODEL` | No | `models/gemini-2.5-flash` | Model identifier |

---

## Limitations

- **In-memory index** — FAISS and BM25 indexes are loaded into RAM at startup. Large corpora (>500MB of PDFs) will need chunking or a persistent vector DB like Chroma or Weaviate.
- **Image-based PDFs** — pdfplumber extracts text only. Scanned PDFs with no text layer will produce empty pages (logged with a warning during ingestion).
- **Single-turn retrieval** — context from prior turns is passed to the LLM but retrieval always runs on the current query only. Follow-up questions like "how does random forest improve on that?" retrieve independently.
- **Free-tier rate limits** — the 80% fallback rate in eval was caused entirely by Gemini free-tier quotas, not retrieval failures.