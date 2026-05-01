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

═══ RULES ═══
1. Primarily use the provided context to answer. If the context does not cover the question, state that clearly. Do not generate facts not present in the context.
2. Never invent citations or technical claims not supported by context. For Advanced level, use equations only when they are grounded in the retrieved context.
3. No preamble ("Great question!", "Sure!"). No sign-off.
4. Prefer accuracy over verbosity.
5. If the question is not about machine learning or data science, respond only with: "This question is outside the scope of this ML tutor." Do not attempt to answer it.

═══ USER PROFILE ═══
Level : {level}
Style : {style}

{level_instruction}
{style_instruction}

═══ REQUIRED OUTPUT FORMAT — FOLLOW EXACTLY ═══
Produce exactly the sections listed below. Do not skip required sections. Do not add extras.

{format_instructions}

**📄 Sources**
- See [source filename], p.[page]

═══ FALLBACK ═══
If context is very thin, explicitly state that the textbook context is insufficient for the question.
"""

# ── Per-level and per-style instructions ─────────────────────────────

_LEVEL_NOTES = {
    "Beginner": (
        "Level instruction: Use plain, everyday language a high school student could follow. "
        "Avoid jargon entirely — if you must use a technical term, define it immediately. "
        "Lead with a real-world analogy (e.g., cooking, sports, daily life) before any technical detail. "
        "Skip ALL math, equations, and formulas even if they appear in context. "
        "Use phrases like 'Think of it like…' or 'Imagine…' to build understanding."
    ),
    "Intermediate": (
        "Level instruction: Balance intuition and technical precision. "
        "Include key formulas if they appear in the context and briefly explain each variable. "
        "Use moderate technical vocabulary but still explain non-obvious terms. "
        "Provide a concrete numerical example when possible."
    ),
    "Advanced": (
        "Level instruction: Use full technical depth and precise terminology. "
        "Use equations only when they are supported by the retrieved context; do not introduce outside formulas. "
        "Use correct mathematical notation (e.g., theta := theta - alpha * dJ/d_theta). "
        "If generating equations, ensure notation is accurate and consistent with the context. "
        "Include derivations, mathematical assumptions, and algorithmic complexities from context. "
        "Reference specific edge cases and failure modes. "
        "No hand-holding — assume strong ML/math background."
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

def _build_format_instructions(style: str, level: str) -> str:
    """
    Adaptive output structure:
    Always: Definition, How it Works, Key Points.
    Detailed only: Intuition, Example, Common Mistakes.
    """
    _ = level
    sections = [
        "**📖 Definition**\n(1–2 sentences. Precise, textbook-quality definition of the concept.)",
        "**⚙️ How it Works**\n1. (Step one — numbered list, 3–5 steps)\n2. (Step two)\n3. (Step three)",
        "**🔑 Key Points**\n- (Bullet 1 — short and precise)\n- (Bullet 2)\n- (Bullet 3)",
    ]
    if style == "Detailed":
        sections.extend([
            "**💡 Intuition**\n(2–3 sentences. Simple mental model. Explain WHY it exists / what problem it solves.)",
            "**🧪 Example**\n(1 concrete example. Beginner: analogy. Intermediate/Advanced: numerical or code example if context supports it.)",
            "**⚠️ Common Mistakes**\n- (Mistake 1)\n- (Mistake 2)\n- (Mistake 3)",
        ])
    sections.append("**📄 Sources**\n- See [source filename], p.[page]")
    return "\n\n".join(sections)


USER_TEMPLATE = """Context:
{context}

Recent conversation (most recent last):
{conversation}

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
    chat_history: list[dict] | None = None,
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
        format_instructions=_build_format_instructions(style, level),
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

    chat_history = chat_history or []
    last_turns = chat_history[-4:]
    convo_lines: list[str] = []
    for turn in last_turns:
        role = "User" if turn.get("role") == "user" else "Assistant"
        text = re.sub(r"\s+", " ", str(turn.get("content", "")).strip())
        if text:
            convo_lines.append(f"{role}: {text}")
    conversation_block = "\n".join(convo_lines) if convo_lines else "None"

    return [
        {"role": "system", "content": system_content},
        {"role": "user",
         "content": USER_TEMPLATE.format(
             context=context_block, conversation=conversation_block, question=question
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


def response_seems_truncated(text: str, style: str = "Detailed") -> bool:
    """
    Heuristic: response is likely cut off if it's missing required section
    headers for the style, OR ends mid-sentence without punctuation.
    """
    expected_headers = ["📖 Definition", "⚙️ How it Works", "🔑 Key Points"]
    if style == "Detailed":
        expected_headers.extend(["💡 Intuition", "🧪 Example", "⚠️ Common Mistakes"])
    missing = sum(1 for h in expected_headers if h not in text)
    ends_abruptly = text.strip() and text.strip()[-1] not in ".!?\"'"
    return missing >= max(1, len(expected_headers) // 2) or (missing >= 1 and ends_abruptly)


# ══════════════════════════════════════════════════════════════════════
#  STRUCTURED FALLBACK  (no LLM required)
#  Synthesizes a clean 6-section answer from chunk text.
#  Key improvement: COMBINES and REWRITES instead of filtering sentences.
# ══════════════════════════════════════════════════════════════════════

def _clean_sentence(s: str) -> str:
    """
    Clean sentence safely:
    - normalize whitespace
    - avoid punctuation rewrites when code/math/decimals are present
    """
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return s
    looks_like_code = any(tok in s for tok in ("`", "==", "!=", "->", "::", "()", "[]", "{}"))
    has_decimal = bool(re.search(r"\d+\.\d+", s))
    has_mathish = any(op in s for op in ("=", "+", "-", "*", "/", "^", "∂", "θ", "α", "β"))
    if not (looks_like_code or has_decimal or has_mathish):
        if s[-1] not in ".!?":
            s += "."
    return s


def _smooth_section(text: str, max_sentences: int = 2) -> str:
    """
    Light rewrite pass: deduplicate sentences, cap length, ensure flow.
    Turns 'assembled text' into 'readable explanation'.
    """
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    seen: set[str] = set()
    cleaned: list[str] = []
    for s in raw:
        s = s.strip()
        if not s or len(s) < 15:
            continue
        key = s[:50].lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(s)
    return " ".join(cleaned[:max_sentences])


def _extract_clean_sentences(text: str) -> list[str]:
    """Extract well-formed sentences from combined chunk text."""
    text = re.sub(r'\s+', ' ', text).strip()
    raw = re.split(r'(?<=[.!?])\s+', text)
    seen: set[str] = set()
    clean: list[str] = []
    for s in raw:
        s = _clean_sentence(s)
        if len(s) < 25:
            continue
        # Skip fragments: low alpha ratio or too few words
        alpha = sum(c.isalpha() for c in s) / max(len(s), 1)
        if alpha < 0.5 or s.count(' ') < 3:
            continue
        # Deduplicate by first 60 chars
        key = s[:60].lower()
        if key in seen:
            continue
        seen.add(key)
        clean.append(s)
    return clean


def _extract_definition(sents: list[str], question: str) -> str:
    """Find the best definitional sentence from the text."""
    signals = [" is ", " is a ", " is an ", " are ", "defined as",
               "refers to", "can be defined", "we define", "known as"]
    # Prefer sentences that match question keywords
    q_words = set(question.lower().split())
    best = None
    best_score = -1
    for s in sents:
        sl = s.lower()
        has_signal = any(sig in sl for sig in signals)
        keyword_overlap = sum(1 for w in q_words if w in sl and len(w) > 2)
        score = (2 if has_signal else 0) + keyword_overlap
        if score > best_score:
            best_score = score
            best = s
    if best and len(best) > 200:
        best = best[:200].rsplit(" ", 1)[0] + "."
    return best or sents[0] if sents else "Definition not found in available context."


def _simplify_for_intuition(sents: list[str], used: set[str]) -> str:
    """Build an intuition paragraph from unused sentences."""
    candidates = [s for s in sents if s not in used][:3]
    if not candidates:
        return "This concept helps solve a fundamental problem in machine learning by providing a systematic approach to improve model performance."
    # Take the 2 shortest candidates (simpler language)
    candidates.sort(key=len)
    picked = candidates[:2]
    return " ".join(picked)


def _extract_steps(sents: list[str], used: set[str]) -> str:
    """Extract procedural sentences as numbered steps."""
    step_signals = ["first", "then", "next", "step", "start", "compute",
                    "calculate", "update", "repeat", "iterate", "apply",
                    "initialize", "select", "choose"]
    steps = [s for s in sents if s not in used
             and any(w in s.lower() for w in step_signals)][:4]
    if not steps:
        steps = [s for s in sents if s not in used and len(s) > 30][:3]
    if steps:
        return "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    return "1. Refer to the source reference below for detailed steps."


def _extract_key_points(sents: list[str], used: set[str]) -> str:
    """Extract key insight sentences as bullet points."""
    kp_signals = {"key", "important", "note", "must", "always", "never",
                  "typically", "generally", "crucial", "fundamental", "essential"}
    points = [s for s in sents
              if any(w in s.lower() for w in kp_signals) and s not in used][:3]
    if not points:
        points = [s for s in sents if s not in used][:3]
    if points:
        return "\n".join(f"- {s[:120]}{'...' if len(s) > 120 else ''}"
                         for s in points)
    return "- See source reference for detailed key points."


def _find_example(sents: list[str], used: set[str], question: str) -> str:
    """Find an example sentence or generate a topic-aware fallback."""
    ex_signals = ["example", "e.g.", "for instance", "such as",
                  "consider", "suppose", "imagine"]
    ex = [s for s in sents
          if any(w in s.lower() for w in ex_signals) and s not in used]
    if ex:
        return ex[0]
    # Topic-aware fallback examples
    q = question.lower()
    if "gradient" in q or "descent" in q:
        return "Consider minimizing f(x) = x^2. Starting at x=4 with learning rate 0.1: x_new = 4 - 0.1*8 = 3.2. Each step moves closer to the minimum at x=0."
    if "backprop" in q:
        return "In a 2-layer network, the error at the output propagates backward: each weight's gradient is computed using the chain rule, layer by layer."
    if "svm" in q or "support vector" in q:
        return "Given two classes of points in 2D, SVM finds the line (hyperplane) that maximizes the margin between the closest points of each class."
    if "knn" in q or "nearest neighbor" in q:
        return "To classify a new point, KNN finds the k closest training points and assigns the majority class. With k=3, if 2 neighbors are 'cat' and 1 is 'dog', the prediction is 'cat'."
    if "overfit" in q or "regulariz" in q:
        return "A model that memorizes 100% of training data but scores 60% on test data is overfitting. Adding L2 regularization penalizes large weights and improves generalization."
    if "decision tree" in q or "random forest" in q:
        return "A decision tree splits data by asking yes/no questions. For predicting loan approval: first split on income > 50k, then on credit score > 700."
    return "For a worked example, refer to the source textbook at the page listed below."


def _find_mistakes(sents: list[str], used: set[str], question: str) -> str:
    """Extract warning sentences or generate topic-aware common mistakes."""
    warn_signals = ["mistake", "error", "avoid", "pitfall", "wrong",
                    "incorrect", "careful", "beware", "issue", "problem"]
    warns = [s for s in sents
             if any(w in s.lower() for w in warn_signals) and s not in used][:2]
    if warns:
        return "\n".join(f"- {s[:120]}{'...' if len(s) > 120 else ''}"
                         for s in warns)
    q = question.lower()
    if "gradient" in q or "descent" in q or "learning rate" in q:
        return ("- Setting the learning rate too high causes divergence; too low causes slow convergence.\n"
                "- Not normalizing input features leads to uneven gradient updates.")
    if "overfit" in q or "regulariz" in q or "dropout" in q:
        return ("- Tuning regularization on the test set leads to data leakage.\n"
                "- Applying too much regularization causes underfitting.")
    if "backprop" in q:
        return ("- Vanishing gradients in deep networks prevent early layers from learning.\n"
                "- Not initializing weights properly can cause training to stall.")
    if "svm" in q:
        return ("- Using a linear kernel on non-linearly separable data gives poor results.\n"
                "- Not scaling features before SVM training biases the decision boundary.")
    return ("- Always validate on held-out data to detect overfitting.\n"
            "- Ensure data is preprocessed consistently between train and test sets.")


def make_structured_fallback(chunks: list, question: str, style: str = "Detailed") -> str:
    """
    Build a clean structured answer from retrieved chunks WITHOUT any LLM.

    Key improvement: SYNTHESIZES instead of filtering.
    - Combines all chunk text into one corpus
    - Deduplicates sentences
    - Cleans broken fragments
    - Rewrites into coherent structured sections
    - Uses topic-aware fallbacks for examples/mistakes
    """
    if not chunks:
        base = (
            "**\U0001f4d6 Definition**\nNo relevant content found in the textbooks.\n\n"
            "**\u2699\ufe0f How it Works**\n1. Check that the relevant PDF is in `data/`.\n"
            "2. Rebuild the index: `python ingest.py --dir data`.\n\n"
            "**\U0001f511 Key Points**\n- No textbook passage matched this query.\n"
        )
        if style == "Detailed":
            base += (
                "\n**\U0001f4a1 Intuition**\nTry rephrasing your question or use a more specific term.\n\n"
                "**\U0001f9ea Example**\nNot available.\n\n"
                "**\u26a0\ufe0f Common Mistakes**\n- Make sure your PDFs cover this topic.\n"
            )
        return base + "\n\n**\U0001f4c4 Sources**\n- No sources available."

    # ── Combine all chunk text into one corpus ────────────────────────
    combined = " ".join(c.get("text", "") for c in chunks[:3])
    sents = _extract_clean_sentences(combined)

    # ── Build each section by synthesis ───────────────────────────────
    definition = _extract_definition(sents, question)
    used: set[str] = {definition}

    intuition = _simplify_for_intuition(sents, used)
    used.update(intuition.split(". "))

    steps = _extract_steps(sents, used)
    used.update(s.lstrip("0123456789. ") for s in steps.split("\n"))

    key_points = _extract_key_points(sents, used)
    example = _find_example(sents, used, question)
    mistakes = _find_mistakes(sents, used, question)

    # ── Smooth rewrite pass — remove repetition, cap length ──────────
    definition = _smooth_section(definition, max_sentences=2)
    intuition  = _smooth_section(intuition, max_sentences=2)
    example    = _smooth_section(example, max_sentences=3)

    # ── Sources ───────────────────────────────────────────────────────
    seen_src: set = set()
    source_lines = []
    for c in chunks:
        key = (c.get("source", "?"), c.get("page", "?"))
        if key not in seen_src:
            seen_src.add(key)
            source_lines.append(f"- See {key[0]}, p.{key[1]}")
    sources = "\n".join(source_lines) or "- Source metadata unavailable."

    body = (
        f"**\U0001f4d6 Definition**\n{definition}\n\n"
        f"**\u2699\ufe0f How it Works**\n{steps}\n\n"
        f"**\U0001f511 Key Points**\n{key_points}\n"
    )
    if style == "Detailed":
        body += (
            f"\n**\U0001f4a1 Intuition**\n{intuition}\n\n"
            f"**\U0001f9ea Example**\n{example}\n\n"
            f"**\u26a0\ufe0f Common Mistakes**\n{mistakes}\n"
        )
    body += (
        f"\n\n**\U0001f4c4 Sources**\n{sources}\n\n"
        f"*Note: Generated from textbook passages directly (LLM unavailable).*"
    )
    return body


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

def _fallback_video_message(query: str) -> list[dict] | None:
    """
    Graceful fallback when no video is found.
    Returns a single "no match" item instead of empty list.
    """
    return [{
        "title": f"No specific video matched '{query}'.",
        "channel": "Recommendation System",
        "url": None,
        "is_fallback": True,
    }]

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


# Synonym map for broader topic detection coverage
_TOPIC_SYNONYMS: dict[str, list[str]] = {
    "backpropagation":       ["backprop", "back propagation", "neural gradient", "error backpropagation"],
    "gradient_descent":      ["gradient descent", "gd algorithm", "optimization step", "steepest descent"],
    "neural_networks":       ["neural network", "neural net", "deep learning", "mlp", "perceptron"],
    "svm":                   ["svm", "support vector", "max margin", "kernel trick"],
    "knn":                   ["knn", "k-nearest", "nearest neighbor", "k nearest"],
    "trees":                 ["decision tree", "random forest", "ensemble", "boosting", "xgboost"],
    "transformers":          ["transformer", "attention mechanism", "self-attention", "bert", "gpt"],
    "regularization":        ["regulariz", "overfitting", "dropout", "weight decay", "l1 l2"],
    "clustering":            ["clustering", "k-means", "kmeans", "dbscan", "centroid"],
    "reinforcement_learning":["reinforcement learning", "q-learning", "rl agent", "policy gradient"],
    "supervised_unsupervised":["supervised vs unsupervised", "labeled vs unlabeled"],
    "loss_functions":        ["loss function", "cross entropy", "cost function", "mse loss", "log loss"],
    "bias_variance":         ["bias variance", "bias-variance", "underfitting overfitting"],
    "training_process":      ["forward pass", "backward pass", "training loop", "training epoch", "activation function"],
}


def detect_topic(query: str) -> str | None:
    """
    Robust topic detection using synonym-based substring matching.
    Uses _TOPIC_SYNONYMS for broad coverage of phrasing variants.
    Returns the topic key or None.
    """
    q = query.lower()
    # Check synonym lists — first match wins (ordered by specificity)
    for topic, synonyms in _TOPIC_SYNONYMS.items():
        if any(syn in q for syn in synonyms):
            return topic
    # Handle compound checks that need AND logic
    if "supervised" in q and "unsupervised" in q:
        return "supervised_unsupervised"
    if "bias" in q and "variance" in q:
        return "bias_variance"
    if "training" in q and ("process" in q or "loop" in q or "epoch" in q):
        return "training_process"
    # Single-word fallbacks
    if "gradient" in q:
        return "gradient_descent"
    return None


def recommend_videos(query: str) -> list[dict]:
    """
    Returns EXACTLY 2 videos for best-matched topic, or fallback message if no match.

    Uses a two-tier approach:
      1. detect_topic() — fast substring matching (catches most cases)
      2. _score_topic() — keyword scoring fallback (handles edge cases)

    Always returns videos if a topic is confidently detected.
    Shows informative fallback instead of silently returning nothing.
    """
    # Tier 1: Direct substring detection (most reliable)
    topic = detect_topic(query)
    if topic and topic in TOPIC_REGISTRY:
        return TOPIC_REGISTRY[topic]

    # Tier 2: Keyword scoring fallback
    query_tokens = _tokenize(query)
    best_topic:      str | None = None
    best_score:      int        = 0

    for topic_key, keywords in TOPIC_KEYWORDS.items():
        score, has_phrase = _score_topic(query_tokens, keywords)
        if score == 0:
            continue
        qualifies = (score >= 2) or has_phrase
        if qualifies and score > best_score:
            best_score = score
            best_topic = topic_key

    if best_topic:
        return TOPIC_REGISTRY[best_topic]
    
    # Fallback: return informative message instead of empty list
    return _fallback_video_message(query)


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