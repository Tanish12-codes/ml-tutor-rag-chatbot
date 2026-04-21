"""
prompt_builder.py — ML Tutor
Production-grade prompt builder + deterministic video recommendation system.

Changes in this version:
  • 6-section output format: Definition → Intuition → How it Works →
    Key Points → Example → Common Mistakes
  • Per-level and per-style instructions injected at runtime
  • _make_structured_fallback() produces clean structured output from raw chunks
    without any LLM call (used when API is unavailable)
  • TOPIC_KEYWORDS covers all 14 required topics
  • Video system unchanged — 18/18 eval passing
"""

import re
import textwrap

# ══════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT — 6-section structured output
# ══════════════════════════════════════════════════════════════════════

SYSTEM_TEMPLATE = """You are a precise, expert machine learning tutor.

═══ ABSOLUTE RULES ═══
1. Use ONLY the provided context. Never use outside knowledge.
2. Never guess. Never fill gaps. Never invent equations or facts.
3. If context is missing or partial, say so explicitly.
4. Every claim must be traceable to a chunk in the context.
5. No preamble ("Great question!", "Sure!"). No sign-off.

═══ USER PROFILE ═══
Level : {level}
Style : {style}

{level_instruction}
{style_instruction}

═══ REQUIRED OUTPUT FORMAT — FOLLOW EXACTLY ═══
Produce ALL six sections below. Do not skip any. Do not add extras.

**📖 Definition**
(1–2 sentences. Precise, textbook-quality definition of the concept.)

**💡 Intuition**
(2–3 sentences. Simple, beginner-friendly analogy or mental model.
 No equations. Explain WHY it exists / what problem it solves.)

**⚙️ How it Works**
1. (Step one — numbered list, 3–5 steps)
2. (Step two)
3. (Step three)
(Add steps 4–5 only for Detailed style or Advanced level.)

**🔑 Key Points**
- (Bullet 1 — short and precise)
- (Bullet 2)
- (Bullet 3)
{extra_bullets}

**🧪 Example**
(1 concrete, simple example. For Beginner: everyday analogy.
 For Intermediate/Advanced: a brief numerical or code example if context supports it.)

**⚠️ Common Mistakes**
- (Mistake 1)
- (Mistake 2)
(Add a 3rd mistake for Detailed style.)

**📄 Sources**
- See [source filename], p.[page]

═══ FALLBACK RULES ═══
If NO relevant context:
  Respond ONLY with: "I don't have enough information in the provided context to answer this."

If PARTIAL context:
  Fill what you can, then end with:
  "Note: the context does not fully cover this topic."
"""

# ── Per-level and per-style instructions ─────────────────────────────

_LEVEL_NOTES = {
    "Beginner": (
        "Level instruction: Use plain, everyday language. "
        "Avoid jargon. Lead with a real-world analogy before any technical detail. "
        "Skip heavy math even if it appears in context."
    ),
    "Intermediate": (
        "Level instruction: Balance intuition and technical precision. "
        "Include key formulas if they appear verbatim in the context. "
        "Briefly explain what each formula means."
    ),
    "Advanced": (
        "Level instruction: Use full technical depth. "
        "Include equations, derivations, and assumptions exactly as in context. "
        "No hand-holding — assume strong ML background."
    ),
}

_STYLE_NOTES = {
    "Quick": (
        "Style instruction: Be concise. "
        "Definition: 1 sentence. Intuition: 1–2 sentences. "
        "How it Works: exactly 3 steps. Key Points: exactly 3 bullets. "
        "Example: 1 sentence. Common Mistakes: 2 bullets."
    ),
    "Detailed": (
        "Style instruction: Be thorough. "
        "Definition: 1–2 sentences. Intuition: 2–3 sentences. "
        "How it Works: 4–5 steps. Key Points: 4–5 bullets. "
        "Example: 2–3 sentences with specifics. Common Mistakes: 3 bullets."
    ),
}

_EXTRA_BULLETS = {
    "Quick":    "",                          # 3 bullets only
    "Detailed": "- (Bullet 4)\n- (Bullet 5)",
}


USER_TEMPLATE = """Context:
{context}

Question:
{question}
"""


# ══════════════════════════════════════════════════════════════════════
#  BUILD MESSAGES
# ══════════════════════════════════════════════════════════════════════

def build_messages(
    question: str,
    chunks:   list,
    level:    str = "Intermediate",
    style:    str = "Detailed",
) -> list[dict]:
    """
    Returns [system_msg, user_msg] dicts.
    Gemini flat call: concatenate messages[0]["content"] + "\n\n" + messages[1]["content"].
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    if level not in {"Beginner", "Intermediate", "Advanced"}:
        raise ValueError(f"Invalid level: {level!r}")
    if style not in {"Quick", "Detailed"}:
        raise ValueError(f"Invalid style: {style!r}")

    system_content = SYSTEM_TEMPLATE.format(
        level=level,
        style=style,
        level_instruction=_LEVEL_NOTES[level],
        style_instruction=_STYLE_NOTES[style],
        extra_bullets=_EXTRA_BULLETS[style],
    )

    if chunks:
        parts = []
        for i, c in enumerate(chunks, 1):
            text   = c.get("text", "").strip()
            source = c.get("source", "unknown")
            page   = c.get("page", "?")
            score  = c.get("rerank_score", c.get("score", ""))
            stag   = f"  [relevance: {score:.3f}]" if isinstance(score, float) else ""
            parts.append(f"[Chunk {i} | {source}, p.{page}{stag}]\n{text}")
        context_block = "\n\n---\n\n".join(parts)
    else:
        context_block = "No relevant context was retrieved."

    return [
        {"role": "system", "content": system_content},
        {"role": "user",
         "content": USER_TEMPLATE.format(
             context=context_block, question=question
         )},
    ]


# ══════════════════════════════════════════════════════════════════════
#  QUALITY CHECKS
# ══════════════════════════════════════════════════════════════════════

def context_adequacy_check(chunks: list) -> bool:
    """False only when retrieval clearly failed (all scores below 0.20)."""
    if not chunks:
        return False
    scores = [c.get("rerank_score", c.get("score", 1.0)) for c in chunks]
    return max(scores) >= 0.20


def llm_response_is_adequate(text: str) -> bool:
    """True when the LLM produced a substantive answer."""
    if not text or len(text.strip()) < 80:
        return False
    phrases = [
        "not enough information",
        "does not fully cover",
        "insufficient",
        "no relevant context",
    ]
    return not any(p in text.lower() for p in phrases)


def response_seems_truncated(text: str) -> bool:
    """
    Heuristic: response is likely cut off if it's missing ≥ 3 of the 6
    expected section headers, OR ends mid-sentence without punctuation.
    """
    expected_headers = [
        "📖 Definition", "💡 Intuition", "⚙️ How it Works",
        "🔑 Key Points", "🧪 Example", "⚠️ Common Mistakes",
    ]
    missing = sum(1 for h in expected_headers if h not in text)
    ends_abruptly = text.strip() and text.strip()[-1] not in ".!?\"'"
    return missing >= 3 or (missing >= 1 and ends_abruptly)


# ══════════════════════════════════════════════════════════════════════
#  STRUCTURED FALLBACK  (no LLM required)
#  Called when API is unavailable. Converts raw chunk text into the
#  6-section format using heuristic text analysis.
# ══════════════════════════════════════════════════════════════════════

def make_structured_fallback(chunks: list, question: str) -> str:
    """
    Build a clean 6-section answer from retrieved chunks WITHOUT any LLM.
    Never dumps raw text. Always produces readable, structured output.

    Strategy:
      Definition    ← best sentence containing "is", "defined as", "refers to"
      Intuition     ← 2 sentences that follow the definition or use plain language
      How it Works  ← remaining sentences split into numbered steps (max 4)
      Key Points    ← signal-word sentences from all chunks (deduplicated)
      Example       ← any sentence containing "example", "e.g.", "for instance"
      Common Mistakes ← warning-style sentences; generic fallback if none found
      Sources       ← from chunk metadata
    """
    if not chunks:
        return (
            "**📖 Definition**\nNo relevant content found in the textbooks.\n\n"
            "**💡 Intuition**\nTry rephrasing your question or use a more specific term.\n\n"
            "**⚙️ How it Works**\n1. Check that the relevant PDF is in `data/`.\n"
            "2. Rebuild the index: `python ingest.py --dir data`.\n\n"
            "**🔑 Key Points**\n- No textbook passage matched this query.\n\n"
            "**🧪 Example**\nNot available.\n\n"
            "**⚠️ Common Mistakes**\n- Make sure your PDFs cover this topic.\n\n"
            "**📄 Sources**\n- No sources available."
        )

    # ── Gather + clean sentences from top 3 chunks ────────────────────
    all_text  = " ".join(c.get("text", "") for c in chunks[:3])
    raw_sents = re.split(r'(?<=[.!?])\s+', all_text)
    sents     = [s.strip() for s in raw_sents if len(s.strip()) > 25]

    # ── Definition: best definitional sentence ────────────────────────
    _def_signals = [" is ", " is a ", " is an ", " are ", "defined as",
                    "refers to", "can be defined", "we define", "known as"]
    def_sent = next(
        (s for s in sents if any(sig in s.lower() for sig in _def_signals)),
        sents[0] if sents else "Definition not found in available context."
    )
    # Keep definition concise — cap at 200 chars
    if len(def_sent) > 200:
        def_sent = def_sent[:200].rsplit(" ", 1)[0] + "…"

    used: set = {def_sent}

    # ── Intuition: next 2 sentences not already used ──────────────────
    intuition_sents = [s for s in sents if s not in used][:2]
    intuition = " ".join(intuition_sents) if intuition_sents else (
        "Think of it as an iterative process that improves a solution step by step."
    )
    used.update(intuition_sents)

    # ── How it Works: remaining → numbered steps ──────────────────────
    step_sents = [s for s in sents if s not in used and len(s) > 30][:4]
    if step_sents:
        steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(step_sents))
    else:
        steps = "1. Refer to the source reference below for a detailed step-by-step explanation."

    # ── Key Points: signal-word sentences, deduplicated ───────────────
    kp_signals = {"key", "important", "note", "must", "always", "never",
                  "typically", "generally", "commonly", "often", "crucial",
                  "main", "primary", "fundamental", "essential"}
    kp_sents = [s for s in sents
                if any(w in s.lower() for w in kp_signals) and s not in used][:3]
    if not kp_sents:
        kp_sents = [s for s in sents if s not in used][:3]
    bullets = "\n".join(
        f"- {s[:110]}{'…' if len(s) > 110 else ''}" for s in kp_sents
    ) or "- See source reference for detailed key points."

    # ── Example: first example-style sentence ─────────────────────────
    ex_signals = ["example", "e.g.", "for instance", "such as",
                  "consider", "suppose", "imagine", "think of"]
    ex_sents = [s for s in sents
                if any(w in s.lower() for w in ex_signals) and s not in used]
    example = ex_sents[0] if ex_sents else (
        "For a worked example, refer to the source textbook at the page listed below."
    )

    # ── Common Mistakes: warning-style sentences ──────────────────────
    warn_signals = ["mistake", "error", "avoid", "pitfall", "common",
                    "wrong", "incorrect", "not ", "don't", "cannot",
                    "careful", "beware", "issue", "problem"]
    warn_sents = [s for s in sents
                  if any(w in s.lower() for w in warn_signals) and s not in used][:2]
    if warn_sents:
        mistakes = "\n".join(
            f"- {s[:110]}{'…' if len(s) > 110 else ''}" for s in warn_sents
        )
    else:
        # Topic-aware generic fallbacks
        q_lower = question.lower()
        if any(w in q_lower for w in ["gradient", "descent", "learning rate"]):
            mistakes = (
                "- Setting the learning rate too high causes divergence; too low causes slow convergence.\n"
                "- Not normalizing input features leads to uneven gradient updates."
            )
        elif any(w in q_lower for w in ["overfit", "regulariz", "dropout"]):
            mistakes = (
                "- Tuning regularization on the test set leads to data leakage.\n"
                "- Applying too much regularization causes underfitting."
            )
        else:
            mistakes = (
                "- Always validate on held-out data to detect overfitting.\n"
                "- Ensure data is preprocessed consistently between train and test sets."
            )

    # ── Sources ───────────────────────────────────────────────────────
    seen_src: set = set()
    source_lines  = []
    for c in chunks:
        key = (c.get("source", "?"), c.get("page", "?"))
        if key not in seen_src:
            seen_src.add(key)
            source_lines.append(f"- See {key[0]}, p.{key[1]}")
    sources = "\n".join(source_lines) or "- Source metadata unavailable."

    return (
        f"**📖 Definition**\n{def_sent}\n\n"
        f"**💡 Intuition**\n{intuition}\n\n"
        f"**⚙️ How it Works**\n{steps}\n\n"
        f"**🔑 Key Points**\n{bullets}\n\n"
        f"**🧪 Example**\n{example}\n\n"
        f"**⚠️ Common Mistakes**\n{mistakes}\n\n"
        f"**📄 Sources**\n{sources}\n\n"
        f"*Note: Generated from textbook passages directly (Gemini unavailable).*"
    )


# ══════════════════════════════════════════════════════════════════════
#  VISUAL HINT DETECTOR
# ══════════════════════════════════════════════════════════════════════

_VISUAL_KW = {
    "formula", "equation", "matrix", "graph", "plot",
    "derivative", "gradient", "integral", "backpropagation",
    "loss function", "convergence", "eigenvalue", "eigenvector",
    "architecture", "decision boundary", "chain rule",
    "jacobian", "hessian",
}

def detect_visual_hint(query: str, chunks: list) -> str | None:
    ql = query.lower()
    for c in chunks:
        if any(k in (ql + " " + c.get("text", "")).lower() for k in _VISUAL_KW):
            return (
                "This topic involves mathematical or visual concepts. "
                f"Refer to {c.get('source','the textbook')}, "
                f"p.{c.get('page','?')} for diagrams or equations."
            )
    return None


# ══════════════════════════════════════════════════════════════════════
#  VIDEO RECOMMENDATION SYSTEM
#
#  Architecture:
#    TOPIC_REGISTRY   — 14 topics × exactly 2 videos (deterministic)
#    TOPIC_KEYWORDS   — strong keyword sets per topic (no generic words)
#    PRACTICE_REGISTRY — Mahesh Huddar worked examples (request-only)
#    EXPLORE_PLAYLISTS — optional "Explore More"
#
#  Matching rules:
#    • score ≥ 2 keyword hits       → confident → show 2 videos
#    • score = 1 + multi-word hit   → confident → show 2 videos
#    • score = 1 + single-word only → ambiguous → show nothing
#    • score = 0                    → no match  → show nothing
# ══════════════════════════════════════════════════════════════════════

TOPIC_REGISTRY: dict[str, list[dict]] = {

    "gradient_descent": [
        {"title":   "3Blue1Brown — Gradient Descent",
         "channel": "3Blue1Brown",
         "url":     "https://youtu.be/IHZwWFHWa-w"},
        {"title":   "Gradient Descent (Andrew Ng)",
         "channel": "DeepLearningAI",
         "url":     "https://youtu.be/sDv4f4s2SB8"},
    ],

    "neural_networks": [
        {"title":   "But what is a neural network?",
         "channel": "3Blue1Brown",
         "url":     "https://youtu.be/aircAruvnKk"},
        {"title":   "Neural Networks Explained",
         "channel": "IBM",
         "url":     "https://youtu.be/jmmW0F0biz0"},
    ],

    "backpropagation": [
        {"title":   "Backpropagation, intuitively",
         "channel": "3Blue1Brown",
         "url":     "https://youtu.be/Ilg3gGewQ5U"},
        {"title":   "Backpropagation Intuition (Andrew Ng)",
         "channel": "DeepLearningAI",
         "url":     "https://youtu.be/yXcQ4B-YSjQ"},
    ],

    "regularization": [
        {"title":   "Regularization — Ridge Regression",
         "channel": "StatQuest",
         "url":     "https://youtu.be/Q81RR3yKn30"},
        {"title":   "Why Regularization Reduces Overfitting",
         "channel": "DeepLearningAI",
         "url":     "https://youtu.be/NyG-7nRpsW8"},
    ],

    "transformers": [
        {"title":   "Transformers, the tech behind LLMs",
         "channel": "3Blue1Brown",
         "url":     "https://youtu.be/wjZofJX0v4M"},
        {"title":   "Transformers Explained",
         "channel": "DeepLearningAI",
         "url":     "https://youtu.be/SZorAJ4I-sA"},
    ],

    "knn": [
        {"title":   "K-Nearest Neighbors, Clearly Explained",
         "channel": "StatQuest",
         "url":     "https://youtu.be/HVXime0nQeI"},
        {"title":   "Nearest Neighbors — MIT OCW",
         "channel": "MIT OpenCourseWare",
         "url":     "https://youtu.be/09mb78oiPkA"},
    ],

    "svm": [
        {"title":   "Support Vector Machines — Main Ideas",
         "channel": "StatQuest",
         "url":     "https://youtu.be/efR1C6CvhmE"},
        {"title":   "Visual Guide to SVMs",
         "channel": "StatQuest",
         "url":     "https://youtu.be/_YPScrckx28"},
    ],

    "trees": [
        {"title":   "Random Forests — StatQuest",
         "channel": "StatQuest",
         "url":     "https://youtu.be/J4Wdy0Wc_xQ"},
        {"title":   "Decision Tree Classification Explained",
         "channel": "Normalized Nerd",
         "url":     "https://youtu.be/ZVR2Way4nwQ"},
    ],

    "clustering": [
        {"title":   "K-Means Clustering — StatQuest",
         "channel": "StatQuest",
         "url":     "https://youtu.be/4b5d3muPQmA"},
        {"title":   "Clustering — MIT OCW",
         "channel": "MIT OpenCourseWare",
         "url":     "https://youtu.be/esmzYhuFnds"},
    ],

    "reinforcement_learning": [
        {"title":   "Reinforcement Learning Explained",
         "channel": "IBM",
         "url":     "https://youtu.be/T_X4XFwKX8k"},
        {"title":   "Stanford CS234 — RL Lecture 1",
         "channel": "Stanford",
         "url":     "https://youtu.be/WsvFL-LjA6U"},
    ],

    "supervised_unsupervised": [
        {"title":   "Supervised vs Unsupervised Learning",
         "channel": "IBM",
         "url":     "https://youtu.be/W01tIRP_Rqs"},
        {"title":   "Intro to Machine Learning — MIT OCW",
         "channel": "MIT OpenCourseWare",
         "url":     "https://youtu.be/kTsieIl_YBA"},
    ],

    "loss_functions": [
        {"title":   "Cross Entropy — StatQuest",
         "channel": "StatQuest",
         "url":     "https://youtu.be/6ArSys5qHAU"},
        {"title":   "Loss Functions Overview (Andrew Ng)",
         "channel": "DeepLearningAI",
         "url":     "https://youtu.be/YkTcK_LXAxw"},
    ],

    "training_process": [
        {"title":   "Training Neural Networks — Forward + Backward Pass",
         "channel": "DeepLearningAI",
         "url":     "https://youtu.be/vStJoetOxJg"},
        {"title":   "Neural Network Training — MIT OCW",
         "channel": "MIT OpenCourseWare",
         "url":     "https://youtu.be/kyQ0CRkYhy4"},
    ],

    "bias_variance": [
        {"title":   "Bias and Variance — StatQuest",
         "channel": "StatQuest",
         "url":     "https://youtu.be/EuBBz3bI-aA"},
        {"title":   "Bias / Variance Tradeoff (Andrew Ng)",
         "channel": "DeepLearningAI",
         "url":     "https://youtu.be/SjQyLhQIXSM"},
    ],
}


TOPIC_KEYWORDS: dict[str, list[str]] = {

    "gradient_descent": [
        "gradient descent", "gradient", "descent",
        "sgd", "adam optimizer", "rmsprop", "momentum optimizer",
        "learning rate", "optimizer", "optimization algorithm",
        "weight update", "stochastic gradient", "mini-batch gradient",
        "convergence rate",
    ],

    "neural_networks": [
        "neural network", "neural networks", "neuron", "perceptron",
        "mlp", "multi-layer perceptron", "feedforward network",
        "deep learning", "hidden layer", "activation function",
        "relu", "sigmoid activation", "softmax layer",
        "deep neural", "artificial neural", "fully connected layer",
    ],

    "backpropagation": [
        "backpropagation", "backprop", "back propagation",
        "chain rule", "backward pass", "gradient flow",
        "vanishing gradient", "exploding gradient",
        "error propagation", "backprop algorithm",
    ],

    "regularization": [
        "regularization", "regularize", "overfitting",
        "dropout", "l1 regularization", "l2 regularization",
        "ridge regression", "lasso regression",
        "weight decay", "early stopping", "prevent overfitting",
        "l1 l2",
    ],

    "transformers": [
        "transformer", "transformers", "attention mechanism", "self-attention",
        "multi-head attention", "bert", "gpt",
        "positional encoding", "transformer model", "llm",
        "large language model", "query key value", "attention head",
        "encoder decoder transformer",
    ],

    "knn": [
        "knn", "k-nearest", "k nearest neighbor",
        "nearest neighbor", "k nearest", "k-nn",
        "instance-based learning", "knn algorithm",
        "k nearest neighbors", "k nearest neighbours",
    ],

    "svm": [
        "svm", "support vector machine", "support vector",
        "kernel trick", "max margin", "soft margin",
        "svm kernel", "svm classifier", "hyperplane svm",
        "rbf kernel", "margin maximization",
    ],

    "trees": [
        "decision tree", "random forest", "ensemble method",
        "boosting", "bagging", "xgboost", "adaboost",
        "gini impurity", "information gain", "tree split",
        "random forests", "decision trees", "leaf node",
    ],

    "clustering": [
        "clustering", "k-means", "kmeans", "k means",
        "dbscan", "hierarchical clustering", "cluster analysis",
        "centroid", "silhouette score", "elbow method",
        "unsupervised clustering",
    ],

    "reinforcement_learning": [
        "reinforcement learning", "reward function",
        "policy gradient", "q-learning", "q learning",
        "markov decision", "mdp", "rl agent",
        "value function", "bellman equation", "dqn",
        "exploration exploitation", "reward signal",
    ],

    "supervised_unsupervised": [
        "supervised learning", "unsupervised learning",
        "labeled data", "unlabeled data",
        "supervised vs unsupervised", "semi-supervised",
        "types of machine learning", "supervised classification",
    ],

    "loss_functions": [
        "loss function", "cross entropy", "cross-entropy",
        "mean squared error", "mse loss", "log loss",
        "hinge loss", "cost function", "objective function",
        "training loss", "categorical cross entropy",
        # logistic regression uses log loss / cross-entropy → covered here
        "logistic regression loss", "sigmoid output",
    ],

    "training_process": [
        "forward pass", "forward propagation", "forward prop",
        "training loop", "training process", "backward pass",
        "training epoch", "mini-batch training", "model training",
        "fit the model", "train the model", "epoch training",
        # activation functions are part of the training process topic
        "activation function", "relu activation", "sigmoid function",
        "softmax function", "tanh activation",
    ],

    "bias_variance": [
        "bias variance", "bias-variance", "bias variance tradeoff",
        "high bias", "high variance", "underfitting overfitting",
        "model complexity tradeoff", "generalization error",
        "variance tradeoff", "overfitting underfitting",
        "bias vs variance",
    ],
}


# ── Practice registry (Mahesh Huddar — explicit request only) ────────

PRACTICE_REGISTRY = [
    {
        "topic":    "backpropagation",
        "keywords": ["backpropagation", "backprop", "chain rule"],
        "video": {
            "title":   "Backpropagation — Numerical Worked Example",
            "channel": "Mahesh Huddar",
            "url":     "https://youtu.be/8d6jf7s6_Qs",
        },
    },
    {
        "topic":    "gradient_descent",
        "keywords": ["gradient descent", "gradient", "descent", "learning rate"],
        "video": {
            "title":   "Gradient Descent — Numerical Example",
            "channel": "Mahesh Huddar",
            "url":     "https://youtu.be/sDv4f4s2SB8",
        },
    },
    {
        "topic":    "neural_networks",
        "keywords": ["neural network", "neural", "perceptron", "forward pass"],
        "video": {
            "title":   "Neural Network — Forward Pass Solved Example",
            "channel": "Mahesh Huddar",
            "url":     "https://youtu.be/6aTobBzFGro",
        },
    },
    {
        "topic":    "svm",
        "keywords": ["svm", "support vector", "kernel trick", "margin"],
        "video": {
            "title":   "SVM — Solved Numerical Example",
            "channel": "Mahesh Huddar",
            "url":     "https://youtu.be/1NxnPkZM9bc",
        },
    },
    {
        "topic":    "knn",
        "keywords": ["knn", "k-nearest", "nearest neighbor"],
        "video": {
            "title":   "KNN — Solved Classification Example",
            "channel": "Mahesh Huddar",
            "url":     "https://youtu.be/4HKqjENq9OU",
        },
    },
    {
        "topic":    "clustering",
        "keywords": ["kmeans", "k-means", "k means", "clustering"],
        "video": {
            "title":   "K-Means Clustering — Worked Example",
            "channel": "Mahesh Huddar",
            "url":     "https://youtu.be/Xvwt7y2jf5E",
        },
    },
    {
        "topic":    "trees",
        "keywords": ["decision tree", "gini impurity", "information gain"],
        "video": {
            "title":   "Decision Tree — Solved Numerical Example",
            "channel": "Mahesh Huddar",
            "url":     "https://youtu.be/sgQAhG5Q7iY",
        },
    },
]


# ── Explore More playlists ────────────────────────────────────────────

EXPLORE_PLAYLISTS = [
    {
        "title":   "Mahesh Huddar — ML Solved Examples",
        "channel": "Mahesh Huddar",
        "url":     "https://www.youtube.com/playlist?list=PLuh62Q4Sv7BUREAvr2QRDsv1jZw6DMToW",
    },
    {
        "title":   "MIT 6.S191 — Introduction to Deep Learning",
        "channel": "MIT OpenCourseWare",
        "url":     "https://www.youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI",
    },
]


# ══════════════════════════════════════════════════════════════════════
#  MATCHING LOGIC
# ══════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> set[str]:
    """Lowercase + strip punctuation (keep hyphens). Returns unigrams + bigrams."""
    clean = re.sub(r"[^\w\s-]", " ", text.lower())
    words = clean.split()
    tokens: set[str] = set(words)
    for i in range(len(words) - 1):
        tokens.add(f"{words[i]} {words[i + 1]}")
    return tokens


def _score_topic(query_tokens: set[str], keywords: list[str]) -> tuple[int, bool]:
    """
    Returns (score, has_phrase_hit).
    has_phrase_hit = True if any hit contains a space or hyphen.
    """
    hits = [kw for kw in keywords if kw in query_tokens]
    has_phrase = any((" " in kw or "-" in kw) for kw in hits)
    return len(hits), has_phrase


def recommend_videos(query: str) -> list[dict]:
    """
    Returns EXACTLY 2 videos for best-matched topic, or [] if no confident match.
    Confidence: score ≥ 2 OR (score = 1 AND multi-word phrase hit).
    """
    query_tokens = _tokenize(query)
    best_topic:      str | None = None
    best_score:      int        = 0

    for topic, keywords in TOPIC_KEYWORDS.items():
        score, has_phrase = _score_topic(query_tokens, keywords)
        if score == 0:
            continue
        qualifies = (score >= 2) or has_phrase
        if qualifies and score > best_score:
            best_score = score
            best_topic = topic

    return TOPIC_REGISTRY[best_topic] if best_topic else []


def recommend_practice_video(query: str) -> dict | None:
    """Single Mahesh Huddar video if query matches a practice topic."""
    qt = _tokenize(query)
    best_score = 0
    best: dict | None = None
    for entry in PRACTICE_REGISTRY:
        score, _ = _score_topic(qt, entry["keywords"])
        if score > best_score:
            best_score = score
            best = entry["video"]
    return best if best_score > 0 else None


def wants_practice(query: str) -> bool:
    """True when the user signals they want worked examples / practice."""
    signals = {
        "practice", "example", "examples", "exercise", "exercises",
        "problem", "problems", "solved", "walkthrough",
        "step by step", "work through", "show me", "demonstrate",
        "numerical", "hands-on", "worked",
    }
    return bool(_tokenize(query) & signals)