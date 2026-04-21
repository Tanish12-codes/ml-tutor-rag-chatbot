"""
rag_eval.py — RAG ML Tutor Evaluation Pipeline
================================================
Tests retrieval quality + answer quality without any heavy frameworks.

Run:
    python rag_eval.py                     # uses mock answers (no Gemini needed)
    python rag_eval.py --live              # calls Gemini API for real answers
    python rag_eval.py --live --top-k 6   # tune retrieval depth

Output:
    eval_results.json   — full per-question breakdown
    eval_summary.txt    — human-readable summary
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# ── Load RAG pipeline ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from rag_pipeline import RAGPipeline, HybridRAGPipeline
from prompt_builder import build_messages, llm_response_is_adequate

# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION DATASET
#  Format:
#    question       : the query
#    expected_keywords : words that MUST appear in a correct answer
#    forbidden_keywords: words that signal hallucination if present
#    topic          : topic tag (for grouping results)
# ══════════════════════════════════════════════════════════════════════════════

EVAL_DATASET = [
    {
        "id": 1,
        "question": "What is gradient descent and how does it minimize loss?",
        "expected_keywords": ["gradient", "loss", "learning rate", "minimize", "update"],
        "forbidden_keywords": ["quantum", "blockchain", "unrelated"],
        "topic": "optimization",
    },
    {
        "id": 2,
        "question": "Explain backpropagation and the chain rule.",
        "expected_keywords": ["chain rule", "gradient", "derivative", "weight", "error"],
        "forbidden_keywords": [],
        "topic": "neural_networks",
    },
    {
        "id": 3,
        "question": "What is overfitting and how does regularization help?",
        "expected_keywords": ["overfitting", "regularization", "training", "generalization"],
        "forbidden_keywords": [],
        "topic": "regularization",
    },
    {
        "id": 4,
        "question": "What is a support vector machine?",
        "expected_keywords": ["support vector", "margin", "hyperplane", "kernel"],
        "forbidden_keywords": [],
        "topic": "svm",
    },
    {
        "id": 5,
        "question": "Explain the bias-variance tradeoff.",
        "expected_keywords": ["bias", "variance", "tradeoff", "complexity"],
        "forbidden_keywords": [],
        "topic": "bias_variance",
    },
    {
        "id": 6,
        "question": "How does the k-nearest neighbors algorithm work?",
        "expected_keywords": ["distance", "neighbor", "classification", "k"],
        "forbidden_keywords": [],
        "topic": "knn",
    },
    {
        "id": 7,
        "question": "What is cross-entropy loss?",
        "expected_keywords": ["cross", "entropy", "probability", "loss"],
        "forbidden_keywords": [],
        "topic": "loss_functions",
    },
    {
        "id": 8,
        "question": "Explain forward propagation in a neural network.",
        "expected_keywords": ["input", "weight", "activation", "output", "layer"],
        "forbidden_keywords": [],
        "topic": "neural_networks",
    },
    {
        "id": 9,
        "question": "What is a decision tree and how does it split nodes?",
        "expected_keywords": ["split", "entropy", "information", "gain", "tree"],
        "forbidden_keywords": [],
        "topic": "trees",
    },
    {
        "id": 10,
        "question": "What is reinforcement learning?",
        "expected_keywords": ["reward", "agent", "environment", "policy", "action"],
        "forbidden_keywords": [],
        "topic": "reinforcement_learning",
    },
    {
        "id": 11,
        "question": "What is the vanishing gradient problem?",
        "expected_keywords": ["gradient", "vanish", "deep", "layer"],
        "forbidden_keywords": [],
        "topic": "neural_networks",
    },
    {
        "id": 12,
        "question": "How does dropout work as a regularization technique?",
        "expected_keywords": ["dropout", "neuron", "randomly", "training"],
        "forbidden_keywords": [],
        "topic": "regularization",
    },
    {
        "id": 13,
        "question": "What is the difference between supervised and unsupervised learning?",
        "expected_keywords": ["label", "supervised", "unsupervised", "cluster"],
        "forbidden_keywords": [],
        "topic": "fundamentals",
    },
    {
        "id": 14,
        "question": "What is a kernel function in SVM?",
        "expected_keywords": ["kernel", "feature", "space", "nonlinear"],
        "forbidden_keywords": [],
        "topic": "svm",
    },
    {
        "id": 15,
        "question": "Explain the softmax activation function.",
        "expected_keywords": ["softmax", "probability", "output", "class"],
        "forbidden_keywords": [],
        "topic": "neural_networks",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
#  SCORING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def score_retrieval(chunks: list[dict], expected_keywords: list[str]) -> dict:
    """
    Check how many expected keywords appear in the retrieved chunks.
    Score = matched / total expected keywords (0.0 – 1.0).
    """
    if not chunks:
        return {"score": 0.0, "matched": [], "missed": expected_keywords, "chunks_returned": 0}

    combined_text = " ".join(c.get("text", "").lower() for c in chunks)
    matched = [kw for kw in expected_keywords if kw.lower() in combined_text]
    missed  = [kw for kw in expected_keywords if kw.lower() not in combined_text]
    score   = round(len(matched) / max(len(expected_keywords), 1), 2)

    top_scores = [c.get("score", 0.0) for c in chunks]

    return {
        "score":           score,
        "matched":         matched,
        "missed":          missed,
        "chunks_returned": len(chunks),
        "top_faiss_score": round(max(top_scores), 4) if top_scores else 0.0,
        "avg_faiss_score": round(sum(top_scores) / len(top_scores), 4) if top_scores else 0.0,
    }


def score_answer(answer: str, expected_keywords: list[str], forbidden_keywords: list[str]) -> dict:
    """
    Keyword-based answer scoring.

    answer_score      : fraction of expected keywords found in answer
    hallucination_flag: True if any forbidden keyword appears
    fallback_flag     : True if model admitted it couldn't answer
    adequate          : True if answer passes llm_response_is_adequate()
    """
    answer_lower = answer.lower()

    matched   = [kw for kw in expected_keywords   if kw.lower() in answer_lower]
    forbidden = [kw for kw in forbidden_keywords  if kw.lower() in answer_lower]
    missed    = [kw for kw in expected_keywords   if kw.lower() not in answer_lower]

    fallback_phrases = [
        "don't have enough information",
        "not enough information",
        "no relevant context",
        "does not fully cover",
    ]
    fallback_flag = any(p in answer_lower for p in fallback_phrases)

    return {
        "answer_score":       round(len(matched) / max(len(expected_keywords), 1), 2),
        "matched_keywords":   matched,
        "missed_keywords":    missed,
        "hallucination_flag": len(forbidden) > 0,
        "forbidden_found":    forbidden,
        "fallback_flag":      fallback_flag,
        "adequate":           llm_response_is_adequate(answer),
        "answer_length":      len(answer),
    }


def composite_score(retrieval_score: float, answer_score: float,
                    hallucination: bool, fallback: bool) -> float:
    """
    Weighted composite: retrieval 40% + answer 60%.
    Penalise hallucination (-0.3) and unanswered questions (-0.2).
    """
    base = 0.4 * retrieval_score + 0.6 * answer_score
    if hallucination:
        base -= 0.30
    if fallback:
        base -= 0.20
    return round(max(0.0, min(1.0, base)), 3)


# ══════════════════════════════════════════════════════════════════════════════
#  GEMINI ANSWER GENERATION  (only used with --live flag)
# ══════════════════════════════════════════════════════════════════════════════

def get_gemini_answer(question: str, chunks: list[dict]) -> str:
    try:
        import google.genai as genai
        from google.genai import types as genai_types

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "[SKIP] GEMINI_API_KEY not set."

        client = genai.Client(api_key=api_key)
        messages = build_messages(question, chunks, level="Intermediate", style="Detailed")
        full_prompt = messages[0]["content"] + "\n\n" + messages[1]["content"]

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=full_prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1024,
            ),
        )
        return response.text or "[EMPTY RESPONSE]"

    except Exception as e:
        return f"[ERROR] {e}"


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN EVAL RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_eval(top_k: int = 4, live: bool = False, hybrid: bool = False) -> None:
    print("\n" + "═" * 70)
    print("  ML TUTOR — RAG EVALUATION PIPELINE")
    print(f"  Mode: {'LIVE (Gemini API)' if live else 'RETRIEVAL-ONLY (no LLM)'}  |  top_k={top_k}  |  retriever={'Hybrid' if hybrid else 'FAISS-only'}")
    print("═" * 70)

    # Load FAISS index
    print("\nLoading FAISS index...")
    rag = HybridRAGPipeline(use_reranker=hybrid) if hybrid else RAGPipeline()
    try:
        rag.load_index()
    except FileNotFoundError:
        print("❌  No FAISS index found. Run ingest.py first.")
        sys.exit(1)
    retriever_name = "HybridRAGPipeline (BM25 + reranker)" if hybrid else "RAGPipeline (FAISS only)"
    print(f"✅  Index loaded — {len(rag.chunks):,} chunks  [{retriever_name}]\n")

    results      = []
    total_comp   = 0.0
    halluc_count = 0
    fallback_count = 0

    SEP = "─" * 70

    for item in EVAL_DATASET:
        qid      = item["id"]
        question = item["question"]
        topic    = item["topic"]

        print(f"{SEP}")
        print(f"  Q{qid:02d} [{topic}]  {question}")

        # ── Retrieval ──────────────────────────────────────────────────
        chunks = rag.retrieve(question, top_k=top_k, score_threshold=0.20)
        ret    = score_retrieval(chunks, item["expected_keywords"])

        print(f"       Retrieval  score={ret['score']:.2f}  "
              f"chunks={ret['chunks_returned']}  "
              f"faiss_top={ret['top_faiss_score']}  "
              f"matched={ret['matched']}  missed={ret['missed']}")

        # ── Answer ─────────────────────────────────────────────────────
        if live:
            print("       Calling Gemini...", end=" ", flush=True)
            t0     = time.time()
            answer = get_gemini_answer(question, chunks)
            elapsed = round(time.time() - t0, 1)
            print(f"done ({elapsed}s)")
        else:
            # Mock: treat retrieved chunk text as the "answer"
            answer = " ".join(c.get("text", "") for c in chunks)

        ans = score_answer(answer, item["expected_keywords"], item["forbidden_keywords"])
        comp = composite_score(
            ret["score"], ans["answer_score"],
            ans["hallucination_flag"], ans["fallback_flag"]
        )
        total_comp   += comp
        if ans["hallucination_flag"]: halluc_count  += 1
        if ans["fallback_flag"]:      fallback_count += 1

        status = "✅" if comp >= 0.6 else ("⚠️ " if comp >= 0.4 else "❌")
        print(f"       Answer     score={ans['answer_score']:.2f}  "
              f"composite={comp:.3f} {status}  "
              f"halluc={ans['hallucination_flag']}  "
              f"fallback={ans['fallback_flag']}")

        results.append({
            "id":           qid,
            "topic":        topic,
            "question":     question,
            "retrieval":    ret,
            "answer":       ans,
            "composite":    comp,
            "answer_text":  answer[:300] if live else "[mock]",
        })

    # ── Summary ────────────────────────────────────────────────────────
    n            = len(EVAL_DATASET)
    avg_comp     = round(total_comp / n, 3)
    pass_count   = sum(1 for r in results if r["composite"] >= 0.6)
    avg_ret      = round(sum(r["retrieval"]["score"] for r in results) / n, 3)
    avg_ans      = round(sum(r["answer"]["answer_score"] for r in results) / n, 3)

    print(f"\n{'═' * 70}")
    print(f"  SUMMARY")
    print(f"{'═' * 70}")
    print(f"  Questions evaluated : {n}")
    print(f"  Passed (≥ 0.60)     : {pass_count}/{n}")
    print(f"  Avg composite score : {avg_comp}")
    print(f"  Avg retrieval score : {avg_ret}")
    print(f"  Avg answer score    : {avg_ans}")
    print(f"  Hallucinations      : {halluc_count}")
    print(f"  Fallback triggers   : {fallback_count}")
    print(f"{'═' * 70}\n")

    # ── Per-topic breakdown ────────────────────────────────────────────
    from collections import defaultdict
    by_topic: dict = defaultdict(list)
    for r in results:
        by_topic[r["topic"]].append(r["composite"])

    print("  Per-topic avg composite:")
    for topic, scores in sorted(by_topic.items()):
        avg = round(sum(scores) / len(scores), 3)
        bar = "█" * int(avg * 20)
        print(f"    {topic:<25} {avg:.3f}  {bar}")
    print()

    # ── Save outputs ───────────────────────────────────────────────────
    with open("eval_results.json", "w") as f:
        json.dump({"summary": {
            "n": n, "passed": pass_count, "avg_composite": avg_comp,
            "avg_retrieval": avg_ret, "avg_answer": avg_ans,
            "hallucinations": halluc_count, "fallbacks": fallback_count,
        }, "results": results}, f, indent=2)

    summary_lines = [
        "ML TUTOR — EVAL SUMMARY",
        f"Mode: {'LIVE' if live else 'RETRIEVAL-ONLY'}  top_k={top_k}",
        f"Questions : {n}",
        f"Passed    : {pass_count}/{n}",
        f"Composite : {avg_comp}",
        f"Retrieval : {avg_ret}",
        f"Answer    : {avg_ans}",
        f"Hallucin. : {halluc_count}",
        f"Fallbacks : {fallback_count}",
        "",
        "Per-topic:",
    ] + [f"  {t:<25} {round(sum(s)/len(s),3)}" for t, s in sorted(by_topic.items())]

    with open("eval_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))

    print("  Saved: eval_results.json  eval_summary.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Eval Pipeline")
    parser.add_argument("--live",   action="store_true", help="Call Gemini API for real answers")
    parser.add_argument("--top-k",  type=int, default=4,  help="Chunks to retrieve per question")
    parser.add_argument("--hybrid", action="store_true", help="Use HybridRAGPipeline (BM25 + reranker)")
    args = parser.parse_args()
    run_eval(top_k=args.top_k, live=args.live, hybrid=args.hybrid)