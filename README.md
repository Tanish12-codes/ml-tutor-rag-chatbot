# ML Tutor — Hybrid RAG Chatbot

An ML tutor chatbot that explains machine learning concepts using a Hybrid Retrieval-Augmented Generation (RAG) pipeline combining semantic search, keyword search, and reranking to produce structured, context-grounded answers.

---

## Features

* Hybrid retrieval: FAISS (semantic) + BM25 (keyword)
* Cross-encoder reranking for improved relevance
* Gemini 2.5 Flash for answer generation
* Fallback system for reliable responses during API failure
* Structured outputs: Definition, Intuition, Steps, Key Points, Example, Mistakes
* Caching and retry logic for stability

---

## Architecture

User Query
→ Hybrid Retrieval (FAISS + BM25)
→ Deduplication
→ Reranking (Cross-Encoder)
→ Top-K Chunks
→ LLM (Gemini)
→ Structured Answer

Fallback is used when LLM fails.

---

## Evaluation

* Relevance Score: ~80%
* Fallback Rate: ~60% (API-limited scenario)
* Avg Latency: ~13.7s (cold start)

---

## Setup

```bash
git clone https://github.com/your-username/ml-tutor-rag-chatbot.git
cd ml-tutor-rag-chatbot
pip install -r requirements.txt
```

Create `.env`:

```env
GEMINI_API_KEY=your_api_key
```

---

## Run

```bash
python ingest.py
streamlit run app.py
```

---

## Structure

```
app.py
rag_pipeline.py
prompt_builder.py
ingest.py
evaluate_system.py
requirements.txt
```

---

## Notes

* Designed for reliability under API constraints
* Retrieval quality improved using hybrid search and reranking
* Evaluation included to measure latency, fallback, and relevance
