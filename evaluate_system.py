import time
import json

from rag_pipeline import HybridRAGPipeline
from app import _cached_llm_call
from prompt_builder import build_messages, make_structured_fallback

# ----------------------------
# CONFIG
# ----------------------------
TEST_QUERIES = [
    "What is gradient descent?",
    "Explain backpropagation step by step",
    "What is overfitting?",
    "Explain SVM intuitively",
    "What is bias vs variance?",
    "What is regularization?",
    "Explain decision trees",
    "What is random forest?",
    "Explain KNN algorithm",
    "What is logistic regression?"
]

KEYWORDS = {
    "gradient descent": ["gradient", "optimization"],
    "backpropagation": ["gradient", "neural"],
    "overfitting": ["train", "generalization"],
    "svm": ["margin", "hyperplane"],
    "bias vs variance": ["bias", "variance"],
    "regularization": ["penalty", "overfitting"],
    "decision trees": ["split", "tree"],
    "random forest": ["ensemble", "trees"],
    "knn": ["neighbors", "distance"],
    "logistic regression": ["probability", "classification"],
}

# ----------------------------
# INIT
# ----------------------------
rag = HybridRAGPipeline(faiss_k=12, bm25_k=12, max_chunks=5)
rag.load_index()

results = []

total_latency = 0
fallback_count = 0
relevance_hits = 0

# ----------------------------
# RUN EVALUATION
# ----------------------------
for query in TEST_QUERIES:
    start = time.time()

    # STEP 1: RETRIEVE
    chunks = rag.retrieve(query, top_k=5, score_threshold=0.0)

    if not chunks:
        response = "No context found"
        used_fallback = True
    else:
        # STEP 2: PROMPT
        messages = build_messages(
            question=query,
            chunks=chunks,
            level="Intermediate",
            style="Detailed",
        )

        prompt = messages[0]["content"] + "\n\n" + messages[1]["content"]

        # STEP 3: LLM
        try:
            response = _cached_llm_call(prompt)

            if not response.strip():
                raise ValueError("Empty LLM response")

            used_fallback = False

        except Exception:
            # STEP 4: FALLBACK
            response = make_structured_fallback(chunks, query)
            used_fallback = True

    latency = time.time() - start
    total_latency += latency

    if used_fallback:
        fallback_count += 1

    # relevance check
    matched_keywords = []
    for key in KEYWORDS:
        if key in query.lower():
            matched_keywords = KEYWORDS[key]
            break

    hit = any(k in response.lower() for k in matched_keywords)
    if hit:
        relevance_hits += 1

    results.append({
        "query": query,
        "latency": round(latency, 2),
        "fallback": used_fallback,
        "relevant": hit
    })

    print(f"✔ {query}")
    print(f"   ⏱ {latency:.2f}s | fallback={used_fallback} | relevant={hit}\n")

# ----------------------------
# METRICS
# ----------------------------
avg_latency = total_latency / len(TEST_QUERIES)
fallback_rate = fallback_count / len(TEST_QUERIES)
relevance_score = relevance_hits / len(TEST_QUERIES)

summary = {
    "avg_latency": round(avg_latency, 2),
    "fallback_rate": round(fallback_rate, 2),
    "relevance_score": round(relevance_score, 2),
    "total_queries": len(TEST_QUERIES)
}

# ----------------------------
# SAVE
# ----------------------------
with open("eval_results.json", "w") as f:
    json.dump({
        "summary": summary,
        "details": results
    }, f, indent=2)

print("\n📊 FINAL METRICS")
print(summary)