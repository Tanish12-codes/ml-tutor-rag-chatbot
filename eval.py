"""
eval.py — Video recommendation system test suite
Run: python eval.py
"""
import sys, re
sys.path.insert(0, '.')
exec(open('prompt_builder.py').read())

SEP = "─" * 72

TEST_CASES = [
    # (query, expect_concept, note)
    ("what is gradient descent",               True,  "core ML topic"),
    ("explain overfitting in neural networks",  True,  "multi-topic: regularization + neural_networks"),
    ("knn algorithm",                           True,  "short 2-word query"),
    ("what is loss in football",                False, "FP guard: loss in non-ML context"),
    ("forward propagation in training",         True,  "training_process coverage"),
    ("reinforcement learning basics",           True,  "short RL query"),
    ("bias vs variance tradeoff",               True,  "vs-separated phrasing"),
    ("how to cook rice",                        False, "completely off-topic"),
    ("bias vs accuracy metrics",               False,  "FP guard: bias vs non-variance"),
    ("precision vs recall tradeoff",           False,  "FP guard: tradeoff non-bias"),
    ("explain backpropagation and chain rule",  True,  "backprop keywords"),
    ("decision tree and information gain",      True,  "trees_ensembles"),
    ("transformer attention mechanism",         True,  "transformers topic"),
    ("what is svm support vector machine",      True,  "SVM"),
    ("explain cross entropy loss function",     True,  "loss_functions"),
    ("what is network latency",                False,  "FP guard: network non-ML"),
    ("what is an agent in software",           False,  "FP guard: agent non-RL"),
    ("what does batch mean in cooking",        False,  "FP guard: batch non-ML"),
]

print(SEP)
print(f"  {'#':<3} {'STATUS':<8} {'EXPECT':<10} {'GOT':<10} QUERY")
print(SEP)

passed = 0
for i, (query, expect_concept, note) in enumerate(TEST_CASES, 1):
    result      = recommend_videos(query)
    got_concept = "concept" in result
    ok          = got_concept == expect_concept
    if ok: passed += 1

    exp    = "concept" if expect_concept else "fallback"
    got    = "concept" if got_concept    else "fallback"
    detail = " | ".join(f"{v['channel']}: {v['title'][:22]}" for v in result.get("concept", []))
    if not got_concept:
        detail = f"fallback → {result.get('fallback', {}).get('channel', '?')}"

    print(f"  {i:<3} {'✅ PASS' if ok else '❌ FAIL':<8} {exp:<10} {got:<10} {query}")
    print(f"       {note}")
    print(f"       {detail}\n")

print(SEP)
print(f"  FINAL: {passed}/{len(TEST_CASES)}", "✅ ALL PASS" if passed == len(TEST_CASES) else "❌ FAILURES FOUND")
print(SEP)
