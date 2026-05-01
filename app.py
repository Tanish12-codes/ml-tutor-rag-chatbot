"""
app.py — ML Tutor  (production — optimized)

LLM layer is provider-agnostic. Switch provider by editing .env only:

    LLM_PROVIDER=gemini          # gemini | openai | claude
    GEMINI_MODEL=models/gemini-2.5-flash
    OPENAI_MODEL=gpt-4o-mini
    CLAUDE_MODEL=claude-3-haiku-20240307
    GEMINI_API_KEY=...
    OPENAI_API_KEY=...
    ANTHROPIC_API_KEY=...

No code changes are required to switch providers.
RAG retrieval, fallback, and prompt logic are untouched.
"""

import hashlib
import os
import time
import random

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from rag_pipeline import RAGPipeline, HybridRAGPipeline
from prompt_builder import (
    build_messages,
    llm_response_is_adequate,
    response_seems_truncated,
    make_structured_fallback,
    detect_visual_hint,
    recommend_videos,
    recommend_practice_video,
    wants_practice,
    EXPLORE_PLAYLISTS,
)

DEV_MODE: bool = False


# ══════════════════════════════════════════════════════════════════════
#  QUERY PREPROCESSING
# ══════════════════════════════════════════════════════════════════════

def preprocess_query(question: str) -> str:
    """
    Light normalization for semantic retrieval models.
    Preserve natural-language phrasing and only expand key contractions.
    """
    import re
    q = question
    # Expand contractions
    contractions = {
        r"\bwhat's\b": "what is",
        r"\bit's\b": "it is",
        r"\bcan't\b": "cannot",
        r"\bdon't\b": "do not",
    }
    for pattern, replacement in contractions.items():
        q = re.sub(pattern, replacement, q, flags=re.IGNORECASE)
    # Strip casual/noisy phrases that hurt retrieval while preserving semantics.
    q = re.sub(r"\blike (i'?m|i am) \d+\b", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\b(idk|lol|tbh|ngl)\b", "", q, flags=re.IGNORECASE)
    # Collapse whitespace
    return re.sub(r"\s+", " ", q).strip()


# ══════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ML Tutor",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── PALETTE ───────────────────────────────────────────────────── */
:root {
    --bg:       #0F172A;
    --sidebar:  #0F172A;
    --card:     #1E293B;
    --border:   #334155;
    --text:     #F1F5F9;
    --muted:    #94A3B8;
    --accent:   #38BDF8;
    --accent2:  #A78BFA;
    --success:  #34D399;
    --warn:     #FBBF24;
}

/* ── BASE ─────────────────────────────────────────────────────── */
html, body, .stApp, .stApp > div,
section[data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}

/* ── HIDE STREAMLIT BRANDING ───────────────────────────────────── */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}

/* ── NATIVE APP FEEL (CURSOR & SELECTION FIX) ──────────────────── */
/* 1. Nuke the I-beam cursor and text highlighting globally */
html, body, .stApp {
    cursor: default !important;
    user-select: none !important;
    -webkit-user-select: none !important;
}

/* 2. Re-enable I-beam and highlighting ONLY where it makes sense */
.stChatMessage, .stChatMessage *, /* The tutor's actual answers */
.stChatInput textarea,            /* The user's typing box */
code, pre {                       /* Any code snippets */
    cursor: text !important;
    user-select: text !important;
    -webkit-user-select: text !important;
}

/* 3. Force the pointer (hand) cursor on clickable UI elements */
button, div[data-testid="stCheckbox"], div[data-testid="stToggle"] {
    cursor: pointer !important;
}

/* kill blue links globally */
a, a:link, a:visited, a:hover, a:active,
.stMarkdown a { color: var(--text) !important; text-decoration: none !important; }

/* ── SIDEBAR ───────────────────────────────────────────────────── */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {
    background-color: var(--sidebar) !important;
    border-right: 1px solid var(--border) !important;
    padding-top: 1.2rem;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── CHAT INPUT ────────────────────────────────────────────────── */
.stChatInput textarea, .stChatInput > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
.stChatInput textarea:focus { border-color: var(--accent) !important; }

/* ── CHAT MESSAGES ─────────────────────────────────────────────── */
.stChatMessage {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
    margin-bottom: 0.65rem !important;
    animation: fadein 0.22s ease;
}
@keyframes fadein {
    from { opacity: 0; transform: translateY(5px); }
    to   { opacity: 1; transform: translateY(0);   }
}

/* ── ANSWER SECTION HEADERS ─────────────────────────────────────── */
/* Bold section headers slightly larger for readability */
.stMarkdown strong { font-weight: 700 !important; }
/* h3 section headers inside chat — teaching-focused layout */
.stChatMessage h3 {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--accent);
    margin: 1.1rem 0 0.35rem 0;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
    letter-spacing: -0.01em;
}
.stChatMessage h3:first-of-type { margin-top: 0.4rem; }

/* ── VIDEO CARD ────────────────────────────────────────────────── */
.vc {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 11px 14px;
    margin-bottom: 8px;
    transition: border-color 0.18s, transform 0.18s, box-shadow 0.18s;
}
.vc:hover {
    border-color: var(--accent);
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(56,189,248,0.10);
}
.vc a {
    font-size: 0.85rem; font-weight: 600;
    color: var(--text) !important; text-decoration: none !important;
    line-height: 1.4; display: block;
}
.vc a:hover { color: var(--accent) !important; }
.vc .ch { font-size: 0.72rem; color: var(--muted); margin-top: 4px; }

/* ── SOURCE TAG ─────────────────────────────────────────────────── */
.src-tag {
    display: inline-flex; align-items: center; gap: 5px;
    background: var(--card); border: 1px solid var(--border);
    border-radius: 6px; padding: 4px 10px;
    font-size: 0.74rem; color: var(--muted); margin: 3px 2px; white-space: nowrap;
}

/* ── PANEL HEADER ───────────────────────────────────────────────── */
.panel-header {
    font-size: 0.70rem; font-weight: 700;
    letter-spacing: 0.09em; text-transform: uppercase;
    color: var(--muted); margin: 0 0 10px 1px;
}

/* ── SECTION LABEL ──────────────────────────────────────────────── */
.section-label {
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.07em; text-transform: uppercase;
    color: var(--accent2); margin-bottom: 5px;
}

/* ── EMPTY STATE ────────────────────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
}
.empty-state-icon { font-size: 2.8rem; margin-bottom: 0.6rem; }
.empty-state-title {
    font-size: 1.05rem; font-weight: 600;
    color: var(--text); margin-bottom: 0.4rem;
}
.empty-state-sub {
    font-size: 0.85rem; color: var(--muted); max-width: 420px; margin: 0 auto 1.2rem;
}
.empty-chip {
    display: inline-block;
    background: var(--card); border: 1px solid var(--border);
    border-radius: 20px; padding: 5px 14px; margin: 4px;
    font-size: 0.78rem; color: var(--muted); cursor: pointer;
}

/* ── ALERTS ─────────────────────────────────────────────────────── */
.stAlert > div {
    background: var(--card) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; color: var(--text) !important;
}

/* ── EXPANDER ───────────────────────────────────────────────────── */
.stExpander > div {
    background: var(--card) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── MISC ───────────────────────────────────────────────────────── */
hr { border-color: var(--border) !important; margin: 0.75rem 0 !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  RESOURCE LOADING
# ══════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading knowledge base…")
def get_rag() -> HybridRAGPipeline:
    rag = HybridRAGPipeline(faiss_k=5, bm25_k=3, max_chunks=3, use_reranker=True)
    try:
        rag.load_index()
    except FileNotFoundError:
        st.error("Knowledge base not found. Run `python ingest.py --dir data` first.")
        st.stop()
    # Warm-load reranker once in cached resource to avoid first-query cold start.
    rag._get_reranker()
    return rag


rag = get_rag()


# ══════════════════════════════════════════════════════════════════════
#  LLM CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

# Switch provider and model entirely via .env — no code changes needed.
PROVIDER: str   = os.getenv("LLM_PROVIDER", "gemini")
_MAX_TOKENS: int = 1800


def get_model(provider: str) -> str:
    """Return the model identifier for the given provider from the environment."""
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
    if provider == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if provider == "claude":
        return os.getenv("CLAUDE_MODEL", "claude-3-haiku-20240307")
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


def clean_text(text: str) -> str:
    """Strip surrogate code points and non-printable characters from LLM output.
    Preserves valid math symbols (θ, ∂, α, etc.) — only drops true surrogates."""
    if not isinstance(text, str):
        return ""
    # Remove ONLY invalid surrogate characters (math symbols are valid UTF-8)
    text = text.encode("utf-8", "ignore").decode("utf-8")
    # Remove control characters except newline + tab
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    return text


def _is_rate_limit_error(exc: Exception) -> bool:
    """Identify retryable API errors including rate limits and server timeouts."""
    low = str(exc).lower()
    retryable_keywords = [
        "429", "rate limit", "quota", "exhausted", "too many requests",
        "503", "500", "service unavailable", "timeout", "internal error", "overloaded"
    ]
    return any(x in low for x in retryable_keywords)


def _retry_with_backoff(fn, max_retries: int = 3, base_delay: float = 2.0):
    """
    Retry API calls with exponential backoff for rate-limit style failures.
    Backoff sequence: 2s, 4s, 8s (bounded by max_retries).
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_retries - 1 or not _is_rate_limit_error(exc):
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0.0, 0.25)
            time.sleep(delay)


# ══════════════════════════════════════════════════════════════════════
#  PROVIDER IMPLEMENTATIONS
# Each function: lazy SDK import, reads its own key, shared _MAX_TOKENS,
# exponential backoff retry for rate limits, returns clean_text(result).
# ══════════════════════════════════════════════════════════════════════

def call_gemini(prompt: str, model: str, style: str) -> str:
    import google.genai as genai                        # lazy — not penalised unless used
    from google.genai import types as genai_types

    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)
    cfg    = genai_types.GenerateContentConfig(
        temperature=0.3,
        max_output_tokens=_MAX_TOKENS,
    )

    def _call_once():
            response = client.models.generate_content(
                model=model, contents=prompt, config=cfg,
            )
            result = clean_text(getattr(response, "text", None) or "")

            # One truncation retry — re-use same config.
            if result and response_seems_truncated(result, style=style):
                r2 = client.models.generate_content(model=model, contents=prompt, config=cfg)
                r2_text = clean_text(getattr(r2, "text", None) or "")
                if r2_text and not response_seems_truncated(r2_text, style=style):
                    return r2_text

            return result
    return _retry_with_backoff(_call_once)


def call_openai(prompt: str, model: str, style: str) -> str:
    from openai import OpenAI                           # lazy — not penalised unless used

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

    def _call_once():
            res = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=_MAX_TOKENS,
                temperature=0.3,
            )
            return clean_text(res.choices[0].message.content or "")
    return _retry_with_backoff(_call_once)


def call_claude(prompt: str, model: str, style: str) -> str:
    from anthropic import Anthropic                     # lazy — not penalised unless used

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set.")

    client = Anthropic(api_key=api_key)

    def _call_once():
            msg = client.messages.create(
                model=model,
                max_tokens=_MAX_TOKENS,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text if msg.content and len(msg.content) > 0 else ""
            return clean_text(raw)
    return _retry_with_backoff(_call_once)


# ══════════════════════════════════════════════════════════════════════
#  UNIFIED ROUTER
# ══════════════════════════════════════════════════════════════════════

def call_llm(prompt: str, provider: str, style: str) -> str:
    """Single entry point for all LLM calls. Routed by LLM_PROVIDER env var."""
    model = get_model(provider)
    print(f"[LLM] {provider} | {model} | len={len(prompt)}")

    if provider == "gemini":
        return call_gemini(prompt, model, style)
    if provider == "openai":
        return call_openai(prompt, model, style)
    if provider == "claude":
        return call_claude(prompt, model, style)
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


def _append_history(role: str, content: str) -> None:
    """Append and cap session history to last 20 messages."""
    st.session_state.history.append({"role": role, "content": clean_text(content)})
    st.session_state.history = st.session_state.history[-20:]


@st.cache_data(show_spinner=False, ttl=3600)
def cached_llm_call(cache_question: str, cache_level: str, cache_style: str, _prompt: str, _provider: str):
    """Cache by user intent tuple; prompt payload is excluded from cache key."""
    return call_llm(_prompt, _provider, cache_style)


# Navbar badge — reflects active provider at startup.
_LLM_MODE = f"{PROVIDER.capitalize()} / {get_model(PROVIDER).split('/')[-1]}"


# ══════════════════════════════════════════════════════════════════════
#  NOTE: Definition-boost scoring is handled inside rag_pipeline.py.
#  No separate post-processing needed here.
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
#  LEFT SIDEBAR
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        '<h2 style="margin:0 0 2px;font-size:1.2rem;color:#F1F5F9">📖 ML Tutor</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:0.74rem;color:#94A3B8;margin:0 0 10px">Concepts explained simply</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    answer_depth = 4
    if DEV_MODE:
        answer_depth = st.select_slider(
            "Retrieval depth", options=[1, 2, 3, 4, 5, 6, 7, 8], value=4,
        )

    st.divider()
    st.markdown("**🔧 Display**")
    show_videos  = st.toggle("Video explanations", value=True)
    show_sources = st.toggle("Source references",  value=True)
    show_chunks  = st.toggle("Raw passages (dev)", value=False) if DEV_MODE else False

    st.divider()
    _topic_labels = [
        "Gradient Descent", "Neural Networks", "Backpropagation",
        "Regularization / Overfitting", "Transformers", "KNN",
        "SVM", "Decision Trees / Random Forests", "Clustering",
        "Reinforcement Learning", "Supervised vs Unsupervised",
        "Loss Functions", "Training Process", "Bias vs Variance",
    ]
    with st.expander("📚 Topics Covered"):
        for t in _topic_labels:
            st.markdown(
                f'<span style="font-size:0.77rem;color:#94A3B8">· {t}</span>',
                unsafe_allow_html=True,
            )

    _books = [
        "DL_NeuralNetworks_Nielsen.pdf",
        "ML_Ethem_Alpaydin.pdf",
        "ML_Math_Foundation.pdf",
        "ML_SoftComputing_Kecman.pdf",
    ]
    with st.expander("📖 Textbook Library"):
        for book in _books:
            st.markdown(
                f'<span style="font-size:0.76rem;color:#64748B">· {book}</span>',
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown(
        f'<span style="font-size:0.67rem;color:#475569">'
        f'FAISS · BM25 · MiniLM · {_LLM_MODE}'
        f'</span>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════

col_chat, col_right = st.columns([3, 1], gap="large")
right_slot          = col_right.empty()

with col_chat:

    # ── Top navbar ────────────────────────────────────────────────────
    nav_col1, nav_col2, nav_col3 = st.columns([2, 1, 1])
    with nav_col1:
        # Raw HTML prevents Streamlit from turning this into a clickable link.
        # user-select: none and cursor: default make it feel like a real app logo.
        st.markdown(
            '<h3 style="margin-bottom: 0; padding-top: 0; user-select: none; cursor: default;">📖 ML Tutor</h3>',
            unsafe_allow_html=True
        )
        st.caption(f"⚡ Grounded in Textbooks | {_LLM_MODE}")
    with nav_col2:
        user_level = st.selectbox(
            "Level",
            ["Beginner", "Intermediate", "Advanced"], 
            index=1,
            key="ui_level",
            label_visibility="collapsed",
        )
    with nav_col3:
        response_style = st.selectbox(
            "Style",
            ["Quick", "Detailed"], 
            index=1,
            key="ui_style",
            label_visibility="collapsed",
        )
    st.divider()

    # ── Chat history ──────────────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state.history       = []
        st.session_state.last_q_hash   = ""
        st.session_state.last_sources  = []
        st.session_state.last_videos   = []
        st.session_state.last_chunks   = []

    recent = st.session_state.history[-12:]
    for msg in recent:
        with st.chat_message(msg["role"]):
            safe_msg = clean_text(msg["content"])
            try:
                st.markdown(safe_msg)
            except Exception:
                st.text(safe_msg)

    # ── Empty state with CLICKABLE suggestion buttons ─────────────────
    if not st.session_state.history:
        st.markdown(
            '<div class="empty-state">'
            '  <div class="empty-state-icon">🎓</div>'
            '  <div class="empty-state-title">Ask any machine learning question</div>'
            '  <div class="empty-state-sub">'
            '    Start with gradient descent, neural networks, or backpropagation — '
            '    and get structured, textbook-grounded answers instantly.'
            '  </div>'
            '</div>',
            unsafe_allow_html=True,
        )
        # Clickable suggestion buttons
        _suggestions = [
            "What is gradient descent?",
            "Explain backpropagation",
            "Bias vs variance tradeoff",
            "How does SVM work?",
            "What is overfitting?",
        ]
        cols = st.columns(len(_suggestions))
        for i, s in enumerate(_suggestions):
            if cols[i].button(s, key=f"suggest_{i}", use_container_width=True):
                st.session_state["_suggestion_query"] = s
                st.rerun()

    question = st.chat_input("Ask a machine learning question…")
    # Handle suggestion button clicks
    if not question and st.session_state.get("_suggestion_query"):
        question = st.session_state.pop("_suggestion_query")


# ══════════════════════════════════════════════════════════════════════
#  QUESTION HANDLER
# ══════════════════════════════════════════════════════════════════════

if question:

    if not question.strip():
        with col_chat:
            st.warning("Please enter a question.")
        st.stop()

    # Duplicate-call guard
    dedupe_key = f"{question.lower().strip()}*{user_level}*{response_style}"
    q_hash = hashlib.md5(dedupe_key.encode()).hexdigest()
    if q_hash == st.session_state.get("last_q_hash", ""):
        st.stop()
    st.session_state.last_q_hash = q_hash

    _append_history("user", question)
    with col_chat:
        with st.chat_message("user"):
            safe_q = clean_text(question)
            try:
                st.markdown(safe_q)
            except Exception:
                st.text(safe_q)

    # ── Step 1: Retrieval ──────────────────────────────────────────────
    with col_chat:
        with st.status("Generating explanation from textbooks…",
                       expanded=False) as status:
            # Build retrieval query: concat last user turn + current question for context
            retrieval_query = question
            if len(st.session_state.history) >= 2:
                # Get the last user message before this one
                for msg in reversed(st.session_state.history[:-1]):
                    if msg.get("role") == "user":
                        last_question = msg.get("content", "").strip()
                        if last_question and last_question != question:
                            retrieval_query = f"{last_question} {question}"
                        break
            # Preprocess query before embedding
            retrieval_query = preprocess_query(retrieval_query)
            chunks = rag.retrieve(
                retrieval_query, top_k=answer_depth, score_threshold=0.20,
                level=user_level,
            )
            if not chunks:
                chunks = rag.retrieve(
                    question, top_k=answer_depth, score_threshold=0.0,
                    level=user_level,
                )

            if chunks:
                # Pipeline handles scoring/boosting internally now
                status.update(
                    label=f"✓ Textbook knowledge loaded ({len(chunks)} passages)",
                    state="complete",
                )
            else:
                status.update(label="⚠ No relevant passages found", state="error")

            # Debug logging
            print(f"[RAG] chunks={len(chunks)} top_score={chunks[0]['score'] if chunks else 'NA'}")

    if not chunks:
        with col_chat:
            with st.chat_message("assistant"):
                reply = (
                    "I couldn't find relevant information in the textbooks for this question. "
                    "Try rephrasing, or check that the relevant PDF is in `data/` "
                    "and the index has been rebuilt."
                )
                try:
                    st.markdown(reply)
                except Exception:
                    st.text(reply)
        st.session_state.history.append({"role": "assistant", "content": clean_text(reply)})
        st.stop()

    # ── Step 2: Build prompt ────────────────────────────────────────────
    messages = build_messages(
        question=question,
        chunks=chunks,
        level=user_level,
        style=response_style,
        chat_history=st.session_state.history[-4:],
    )
    full_prompt = messages[0]["content"] + "\n\n" + messages[1]["content"]

    # ── Step 3: LLM → fallback chain ───────────────────────────────────
    with col_chat:
        with st.chat_message("assistant"):

            st.markdown(
                '<p class="section-label">📘 Answer from textbooks</p>',
                unsafe_allow_html=True,
            )

            answer_placeholder = st.empty()
            full_response      = ""
            _api_error         = False

            with st.spinner("Thinking..."):
                try:
                    full_response = cached_llm_call(
                        question,
                        user_level,
                        response_style,
                        full_prompt,
                        PROVIDER,
                    )

                    # Advanced: relaxed gate — only catch broken outputs
                    # Other levels: full quality gate (removed dead context_adequacy_check)
                    if user_level == "Advanced":
                        if (
                            not full_response.strip()
                            or response_seems_truncated(full_response, style=response_style)
                        ):
                            full_response = make_structured_fallback(chunks, question, style=response_style)
                    else:
                        if (
                            not full_response.strip()
                            or not llm_response_is_adequate(full_response)
                            or response_seems_truncated(full_response, style=response_style)
                        ):
                            full_response = make_structured_fallback(chunks, question, style=response_style)

                except Exception as exc:
                    _api_error = True
                    low = str(exc).lower()

                    # Print the exact exception to the terminal for debugging
                    print(f"\nDEBUG LLM CRASH: {repr(exc)}")

                    if any(x in low for x in ["429", "quota", "rate", "exhausted", "503", "500", "overloaded"]):
                        st.warning(
                            "API busy or rate limit reached. "
                            "Generating explanation from textbook passages instead."
                        )
                    elif any(x in low for x in ["401", "api key", "invalid", "permission"]):
                        st.error(
                            "API key error. "
                            f"Check the API key for provider `{PROVIDER}` in your `.env`."
                        )
                    elif "not initialised" in low:
                        st.info("No LLM connected. Showing textbook explanation.")
                    else:
                        st.warning(
                            "LLM unavailable. "
                            "Showing structured textbook explanation instead."
                        )

                    full_response = make_structured_fallback(chunks, question, style=response_style)

            # Upgrade section headers for teaching-focused display
            _header_map = {
                "**📖 Definition**": "### 📖 Definition",
                "**💡 Intuition**": "### 💡 Intuition",
                "**⚙️ How it Works**": "### ⚙️ How it Works",
                "**🔑 Key Points**": "### 🔑 Key Points",
                "**🧪 Example**": "### 🧪 Example",
                "**⚠️ Common Mistakes**": "### ⚠️ Common Mistakes",
                "**📄 Sources**": "### 📄 Sources",
            }
            for old, new in _header_map.items():
                full_response = full_response.replace(old, new)

            safe = clean_text(full_response)
            try:
                answer_placeholder.markdown(safe)
            except Exception:
                answer_placeholder.text(safe)

            # Post-answer quality notes
            if not _api_error and not llm_response_is_adequate(full_response):
                st.info("ℹ️ The retrieved context may not fully cover this question.")

            hint = detect_visual_hint(question, chunks)
            if hint:
                st.info(f"📐 {hint}")

            if show_chunks:
                with st.expander("🔬 Retrieved passages (dev)", expanded=False):
                    for i, c in enumerate(chunks, 1):
                        passage_header = (
                            f"**Passage {i}** — `{c.get('source','?')}` "
                            f"p.{c.get('page','?')}  "
                            f"score: {c.get('rerank_score', c.get('score','?'))}"
                        )
                        try:
                            st.markdown(clean_text(passage_header))
                        except Exception:
                            st.text(clean_text(passage_header))
                        txt = clean_text(c.get("text", ""))
                        st.text(txt[:400] + ("…" if len(txt) > 400 else ""))

    _append_history("assistant", full_response)
    # Persist sources and chunks for stable right panel
    st.session_state.last_chunks = chunks

    # ══════════════════════════════════════════════════════════════════
    #  RIGHT PANEL — videos + sources (persisted in session_state)
    # ══════════════════════════════════════════════════════════════════

    # Compute and persist videos
    concept_videos = recommend_videos(question)
    st.session_state.last_videos = concept_videos

    # Compute and persist sources
    seen_src: set  = set()
    refs: list = []
    for c in chunks:
        key = (c.get("source", "?"), c.get("page", "?"))
        if key not in seen_src:
            seen_src.add(key)
            refs.append(key)
    st.session_state.last_sources = refs


# ══════════════════════════════════════════════════════════════════════
#  RIGHT PANEL — always rendered from session_state (never disappears)
# ══════════════════════════════════════════════════════════════════════

with right_slot.container():

    if show_videos:
        st.markdown(
            '<p class="panel-header">🎬 Video Explanations</p>',
            unsafe_allow_html=True,
        )

        _videos = st.session_state.get("last_videos", [])

        if _videos:
            for v in _videos:
                st.markdown(
                    f'<div class="vc">'
                    f'  <a href="{v["url"]}" target="_blank">{v["title"]}</a>'
                    f'  <div class="ch">{v["channel"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        elif st.session_state.get("history"):
            st.markdown(
                '<p style="font-size:0.78rem;color:#475569;padding:4px 2px">'
                'No video match for this query.</p>',
                unsafe_allow_html=True,
            )

        # Practice videos — check last question from history
        _last_q = ""
        for msg in reversed(st.session_state.get("history", [])):
            if msg["role"] == "user":
                _last_q = msg["content"]
                break
        if _last_q and wants_practice(_last_q):
            pv = recommend_practice_video(_last_q)
            if pv:
                st.markdown(
                    '<p class="panel-header" style="margin-top:14px">'
                    '🧮 Practice Example</p>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="vc">'
                    f'  <a href="{pv["url"]}" target="_blank">{pv["title"]}</a>'
                    f'  <div class="ch">{pv["channel"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with st.expander("📚 Explore More", expanded=False):
            for pl in EXPLORE_PLAYLISTS:
                st.markdown(
                    f'<div class="vc">'
                    f'  <a href="{pl["url"]}" target="_blank">{pl["title"]}</a>'
                    f'  <div class="ch">{pl["channel"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<hr/>', unsafe_allow_html=True)

    if show_sources:
        st.markdown(
            '<p class="panel-header">📄 Sources</p>',
            unsafe_allow_html=True,
        )
        _refs = st.session_state.get("last_sources", [])

        if _refs:
            tags = "".join(
                f'<span class="src-tag">📄 {s} — p.{p}</span>'
                for s, p in _refs
            )
            st.markdown(
                f'<div style="display:flex;flex-wrap:wrap;gap:4px">{tags}</div>',
                unsafe_allow_html=True,
            )
        elif st.session_state.get("history"):
            st.markdown(
                '<p style="font-size:0.77rem;color:#475569">No source metadata.</p>',
                unsafe_allow_html=True,
            )