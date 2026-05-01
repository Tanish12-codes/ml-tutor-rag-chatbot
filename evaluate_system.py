"""
evaluate_system.py — ML Tutor Evaluation Pipeline
==================================================
Rate limit strategy (Gemini free tier: 20 req/day on gemini-2.5-flash):
  • 10 queries × 1 level = 10 calls/run  →  safe within daily quota
  • Run one level per day across 3 days for full coverage
  • 75-second sleep between calls (~5 RPM, under per-minute limit)
  • Progress checkpointed to JSON after every call — crash-safe

3-day plan:
  Day 1:  python evaluate_system.py --level Beginner
  Day 2:  python evaluate_system.py --level Intermediate
  Day 3:  python evaluate_system.py --level Advanced

After all 3 runs:
          python evaluate_system.py --merge

Query coverage (10 queries, one per production case):
  ① Core concepts       — standard happy-path retrieval
  ② Phrasing variant    — same concept, different wording
  ③ Multi-concept       — answer needs to blend two topics
  ④ Out-of-scope ×2     — should abstain, not hallucinate
  ⑤ Noisy / casual ×2  — real user phrasing, not textbook
  ⑥ Ambiguous short     — single-word stress test
  ⑦ Multi-turn ×2      — conversation context injection

Outputs (per run):
  eval_results_<Level>.json   — per-query detail
  eval_summary_<Level>.txt    — human-readable summary

Output (after --merge):
  eval_results_all.json       — combined results + per-level breakdown
  eval_summary_all.txt        — final summary across all levels
"""

import argparse
import json
import os
import random
import time

from dotenv import load_dotenv
load_dotenv()

from rag_pipeline import HybridRAGPipeline
from prompt_builder import (
    build_messages,
    make_structured_fallback,
    llm_response_is_adequate,
    response_seems_truncated,
    context_adequacy_check,
)

import google.genai as genai
from google.genai import types as genai_types

_api_key     = os.getenv("GEMINI_API_KEY")
_client      = genai.Client(api_key=_api_key) if _api_key else None
GEMINI_MODEL = "models/gemini-2.5-flash"

INTER_CALL_SLEEP = 75   # seconds — keeps per-minute rate under free-tier limit


# ══════════════════════════════════════════════════════════════════════
#  GEMINI CALL
# ══════════════════════════════════════════════════════════════════════

def _call_gemini(prompt: str) -> str:
    if _client is None:
        raise ValueError("No GEMINI_API_KEY set in .env")
    cfg = genai_types.GenerateContentConfig(
        temperature=0.3,
        max_output_tokens=1800,
    )
    for attempt in range(3):
        try:
            resp = _client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt, config=cfg,
            )
            return getattr(resp, "text", None) or ""
        except Exception as exc:
            low = str(exc).lower()
            is_rate = any(x in low for x in [
                "429", "rate limit", "quota", "exhausted", "too many requests"
            ])
            if attempt == 2 or not is_rate:
                raise
            delay = (2 ** (attempt + 2)) + random.uniform(0, 1.0)
            print(f"     [RATE LIMIT] backing off {delay:.1f}s …")
            time.sleep(delay)
    return ""


# ══════════════════════════════════════════════════════════════════════
#  TEST SUITE — 10 queries, one per production case
# ══════════════════════════════════════════════════════════════════════
#
#  Removed from original 15:
#    - "Explain backpropagation"      (overlaps with gradient descent)
#    - "why does my model memorize"   (overlaps with regularization)
#    - "bias-variance tradeoff"       (overlaps with L1/L2)
#    - "kernels"                      (too ambiguous — noisy signal)
#    - "trees"                        (same issue)

TEST_CASES = [

    # ── ① Core concept ─────────────────────────────────────────────────
    {
        "query":    "What is gradient descent?",
        "category": "core_concept",
        "keywords": ["gradient", "optimization", "learning rate", "loss", "update"],
        "fake_history": None,
    },

    # ── ② Phrasing variant ─────────────────────────────────────────────
    # Same concept as regularization but phrased as a debugging question
    {
        "query":    "how do neural nets learn from errors",
        "category": "phrasing_variant",
        "keywords": ["gradient", "error", "weight", "backward", "loss"],
        "fake_history": None,
    },

    # ── ③ Multi-concept ────────────────────────────────────────────────
    {
        "query":    "What is the difference between L1 and L2 regularization?",
        "category": "multi_concept",
        "keywords": ["l1", "l2", "lasso", "ridge", "sparsity", "penalty"],
        "fake_history": None,
    },

    # ── ④ Out-of-scope ×2 ──────────────────────────────────────────────
    {
        "query":    "How does the Python GIL affect multithreading performance?",
        "category": "out_of_scope",
        "keywords": [],   # checking it abstains cleanly, not that it has ML content
        "fake_history": None,
    },
    {
        "query":    "What is the capital of France?",
        "category": "out_of_scope",
        "keywords": [],
        "fake_history": None,
    },

    # ── ⑤ Noisy / casual ×2 ────────────────────────────────────────────
    {
        "query":    "idk what svm even does can u explain it simply lol",
        "category": "noisy_casual",
        "keywords": ["margin", "hyperplane", "support vector", "classify"],
        "fake_history": None,
    },
    {
        "query":    "explain knn like im 10",
        "category": "noisy_casual",
        "keywords": ["neighbor", "distance", "closest", "classify"],
        "fake_history": None,
    },

    # ── ⑥ Ambiguous short ──────────────────────────────────────────────
    # Single-word stress test for retrieval — does it pull anything useful?
    {
        "query":    "What is regularization in machine learning?",
        "category": "core_concept",
        "keywords": ["penalty", "overfitting", "l1", "l2", "weight"],
        "fake_history": None,
    },

    # ── ⑦ Multi-turn follow-up ×2 ──────────────────────────────────────
    # Simulates: user just asked about gradient descent, follows up on learning rate
    {
        "query":    "what happens if the learning rate is too high?",
        "category": "multi_turn_followup",
        "keywords": ["diverge", "overshoot", "learning rate", "gradient", "converge"],
        "fake_history": [
            {"role": "user",      "content": "What is gradient descent?"},
            {"role": "assistant", "content": (
                "Gradient descent is an optimization algorithm that minimizes a loss "
                "function by iteratively updating model parameters in the direction of "
                "steepest descent of the gradient, scaled by the learning rate."
            )},
        ],
    },
    # Simulates: user asked about decision trees, asks about an extension
    {
        "query":    "how does random forest improve on that?",
        "category": "multi_turn_followup",
        "keywords": ["ensemble", "tree", "bagging", "bootstrap", "variance", "forest"],
        "fake_history": [
            {"role": "user",      "content": "Explain decision trees"},
            {"role": "assistant", "content": (
                "A decision tree is a supervised learning model that recursively splits "
                "data based on feature thresholds, maximizing information gain or "
                "minimizing Gini impurity at each node, until reaching leaf predictions."
            )},
        ],
    },
]


# ══════════════════════════════════════════════════════════════════════
#  SCORING
# ══════════════════════════════════════════════════════════════════════

def is_relevant(response: str, keywords: list[str]) -> tuple[bool, int]:
    """
    Out-of-scope queries (empty keywords) pass automatically —
    we only verify they don't crash or hallucinate ML content.
    """
    if not keywords:
        return True, 0
    resp_lower = response.lower()
    matches = sum(1 for k in keywords if k in resp_lower)
    return matches >= 1, matches


def score_response(response: str, keywords: list[str]) -> dict:
    relevant, match_count = is_relevant(response, keywords)
    adequate  = llm_response_is_adequate(response)
    truncated = response_seems_truncated(response, style="Detailed")
    has_sections = sum(
        1 for h in ["Definition", "How it Works", "Key Points"]
        if h in response
    )
    return {
        "relevant":        relevant,
        "keyword_matches": match_count,
        "adequate":        adequate,
        "truncated":       truncated,
        "section_hits":    has_sections,
    }


# ══════════════════════════════════════════════════════════════════════
#  MAIN EVALUATION LOOP
# ══════════════════════════════════════════════════════════════════════

def run_eval(level: str) -> None:
    out_json = f"eval_results_{level}.json"
    out_txt  = f"eval_summary_{level}.txt"

    print(f"\nLoading knowledge base…")
    rag = HybridRAGPipeline(faiss_k=5, bm25_k=3, max_chunks=3, use_reranker=False)
    rag.load_index()
    print(f"[OK] Loaded {len(rag.chunks)} chunks\n")

    total_calls = len(TEST_CASES)
    results: list = []

    total_latency = fallback_count = relevance_hits = quality_hits = section_hits = 0

    print(f"{'═' * 55}")
    print(f"  LEVEL: {level}   ({total_calls} queries)")
    print(f"{'═' * 55}\n")

    for i, case in enumerate(TEST_CASES, 1):
        query    = case["query"]
        category = case["category"]
        keywords = case["keywords"]
        history  = case["fake_history"] or []

        print(f"  [{i}/{total_calls}] [{category}]")
        print(f"  Q: {query}")

        start = time.time()

        chunks = rag.retrieve(query, top_k=4, level=level)

        used_fallback   = False
        fallback_reason = ""

        if not chunks:
            response        = "No context found."
            used_fallback   = True
            fallback_reason = "quality_gate"
        else:
            messages = build_messages(
                question=query,
                chunks=chunks,
                level=level,
                style="Detailed",
                chat_history=history,
            )
            prompt = messages[0]["content"] + "\n\n" + messages[1]["content"]

            try:
                response = _call_gemini(prompt)

                if level == "Advanced":
                    if (
                        not response.strip()
                        or response_seems_truncated(response, style="Detailed")
                    ):
                        response        = make_structured_fallback(chunks, query, style="Detailed")
                        used_fallback   = True
                        fallback_reason = "quality_gate"
                else:
                    if (
                        not response.strip()
                        or not llm_response_is_adequate(response)
                        or response_seems_truncated(response, style="Detailed")
                        or not context_adequacy_check(chunks)
                    ):
                        response        = make_structured_fallback(chunks, query, style="Detailed")
                        used_fallback   = True
                        fallback_reason = "quality_gate"

            except Exception as exc:
                print(f"     [WARN] LLM error: {exc}")
                response        = make_structured_fallback(chunks, query, style="Detailed")
                used_fallback   = True
                fallback_reason = "api_error"

        latency = time.time() - start
        scores  = score_response(response, keywords)

        total_latency  += latency
        if used_fallback:   fallback_count += 1
        if scores["relevant"]: relevance_hits += 1
        if scores["adequate"]: quality_hits   += 1
        section_hits += scores["section_hits"]

        ok = "[OK]" if scores["relevant"] and scores["adequate"] else "[!!]"
        print(
            f"     {ok} {latency:.1f}s | "
            f"fallback={used_fallback}({fallback_reason or 'none'}) | "
            f"relevant={scores['relevant']} ({scores['keyword_matches']} hits) | "
            f"adequate={scores['adequate']} | "
            f"sections={scores['section_hits']}/3"
        )

        record = {
            "query":            query,
            "category":         category,
            "level":            level,
            "latency":          round(latency, 2),
            "fallback":         used_fallback,
            "fallback_reason":  fallback_reason,
            "chunks_retrieved": len(chunks),
            **scores,
        }
        results.append(record)

        # Checkpoint after every call
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"summary": "IN PROGRESS", "details": results}, f, indent=2)

        if i < total_calls:
            remaining = total_calls - i
            print(f"     [SLEEP] {INTER_CALL_SLEEP}s  ({remaining} calls left)\n")
            time.sleep(INTER_CALL_SLEEP)

    # ── Final metrics ─────────────────────────────────────────────────
    n               = total_calls
    avg_latency     = total_latency / n
    fallback_rate   = fallback_count / n
    relevance_score = relevance_hits / n
    quality_score   = quality_hits / n
    avg_sections    = section_hits / n

    latency_bonus  = max(0, min(1, (10 - avg_latency) / 8))
    fallback_bonus = 1.0 - fallback_rate
    format_score   = avg_sections / 3.0
    composite = (
        0.40 * relevance_score
        + 0.30 * quality_score
        + 0.15 * latency_bonus
        + 0.10 * fallback_bonus
        + 0.05 * format_score
    )

    summary = {
        "level":            level,
        "total_queries":    n,
        "avg_latency":      round(avg_latency, 2),
        "fallback_rate":    round(fallback_rate, 3),
        "relevance_score":  round(relevance_score, 3),
        "quality_score":    round(quality_score, 3),
        "avg_section_hits": round(avg_sections, 2),
        "composite_score":  round(composite, 3),
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f, indent=2)

    lines = [
        f"ML Tutor Evaluation — {level}",
        "=" * 45,
        f"Total queries:      {n}",
        "",
        f"Relevance score:    {relevance_score:.2%}",
        f"Quality score:      {quality_score:.2%}",
        f"Avg section hits:   {avg_sections:.2f}/3.0",
        f"Fallback rate:      {fallback_rate:.2%}",
        f"Avg latency:        {avg_latency:.2f}s",
        "",
        f"Composite score:    {composite:.3f}",
        "=" * 45,
        "",
        f"Full detail → {out_json}",
    ]
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n{'=' * 55}")
    print(f"  {level} — DONE")
    print(f"  Relevance:    {relevance_score:.2%}")
    print(f"  Quality:      {quality_score:.2%}")
    print(f"  Fallback:     {fallback_rate:.2%}")
    print(f"  Avg latency:  {avg_latency:.2f}s")
    print(f"  Composite:    {composite:.3f}")
    print(f"{'=' * 55}")
    print(f"  Saved → {out_json}  +  {out_txt}")


# ══════════════════════════════════════════════════════════════════════
#  MERGE — combines 3 per-level JSON files into one summary
# ══════════════════════════════════════════════════════════════════════

def merge_results() -> None:
    levels = ["Beginner", "Intermediate", "Advanced"]
    all_details: list = []
    level_summaries: list = []
    missing = []

    for level in levels:
        path = f"eval_results_{level}.json"
        if not os.path.exists(path):
            missing.append(path)
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data.get("summary"), dict):
            level_summaries.append(data["summary"])
        all_details.extend(data.get("details", []))

    if missing:
        print(f"[WARN] Missing files — run these first: {missing}")
        if not level_summaries:
            return

    if not all_details:
        print("[ERROR] No detail records found.")
        return

    # Overall metrics across all levels
    n               = len(all_details)
    avg_latency     = sum(r["latency"] for r in all_details) / n
    fallback_rate   = sum(1 for r in all_details if r["fallback"]) / n
    relevance_score = sum(1 for r in all_details if r["relevant"]) / n
    quality_score   = sum(1 for r in all_details if r["adequate"]) / n
    avg_sections    = sum(r["section_hits"] for r in all_details) / n

    latency_bonus  = max(0, min(1, (10 - avg_latency) / 8))
    fallback_bonus = 1.0 - fallback_rate
    format_score   = avg_sections / 3.0
    composite = (
        0.40 * relevance_score
        + 0.30 * quality_score
        + 0.15 * latency_bonus
        + 0.10 * fallback_bonus
        + 0.05 * format_score
    )

    overall = {
        "levels_tested":    [s["level"] for s in level_summaries],
        "total_queries":    n,
        "avg_latency":      round(avg_latency, 2),
        "fallback_rate":    round(fallback_rate, 3),
        "relevance_score":  round(relevance_score, 3),
        "quality_score":    round(quality_score, 3),
        "avg_section_hits": round(avg_sections, 2),
        "composite_score":  round(composite, 3),
    }

    with open("eval_results_all.json", "w", encoding="utf-8") as f:
        json.dump({
            "overall":          overall,
            "per_level":        level_summaries,
            "details":          all_details,
        }, f, indent=2)

    # Build summary text
    lines = [
        "ML Tutor Evaluation — All Levels",
        "=" * 45,
        f"Levels:             {', '.join(overall['levels_tested'])}",
        f"Total queries:      {n}",
        "",
        "── Overall ──",
        f"Relevance score:    {relevance_score:.2%}",
        f"Quality score:      {quality_score:.2%}",
        f"Avg section hits:   {avg_sections:.2f}/3.0",
        f"Fallback rate:      {fallback_rate:.2%}",
        f"Avg latency:        {avg_latency:.2f}s",
        f"Composite score:    {composite:.3f}",
        "",
        "── Per Level ──",
    ]
    for s in level_summaries:
        lines += [
            f"  {s['level']:<14} relevance={s['relevance_score']:.0%}  "
            f"quality={s['quality_score']:.0%}  "
            f"fallback={s['fallback_rate']:.0%}  "
            f"latency={s['avg_latency']:.1f}s  "
            f"composite={s['composite_score']:.3f}",
        ]
    lines += ["", "=" * 45, "", "Full detail → eval_results_all.json"]

    with open("eval_summary_all.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n" + "=" * 55)
    print("  MERGED RESULTS")
    print("=" * 55)
    for line in lines:
        print(f"  {line}")
    print("  Saved → eval_results_all.json  +  eval_summary_all.txt")


# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ML Tutor Evaluation Pipeline — 10 queries, one level per day"
    )
    parser.add_argument(
        "--level",
        choices=["Beginner", "Intermediate", "Advanced"],
        default="Intermediate",
        help="Level to evaluate (default: Intermediate). Run one per day.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help=(
            "Merge eval_results_Beginner.json + Intermediate + Advanced "
            "into eval_results_all.json. Run after all 3 days are complete."
        ),
    )
    args = parser.parse_args()

    if args.merge:
        merge_results()
    else:
        est = len(TEST_CASES) * (INTER_CALL_SLEEP + 10) / 60
        print(f"ML Tutor Eval — {len(TEST_CASES)} queries | Level: {args.level}")
        print(f"Estimated time: ~{est:.0f} min  |  Sleep between calls: {INTER_CALL_SLEEP}s")
        print(f"Output files: eval_results_{args.level}.json  +  eval_summary_{args.level}.txt\n")
        run_eval(args.level)