"""
app.py — ML Tutor  (production)
SDK   : google-genai
Model : gemini-2.5-flash

Changes in this version:
  • Gemini-only — Claude/Anthropic removed completely
  • Model: models/gemini-2.5-flash (confirmed in gemini_test.py)
  • max_tokens 1800 — prevents answer cut-offs
  • Exponential backoff on 429 / rate limit errors
  • response_seems_truncated() → auto-retry once on cut-off
  • make_structured_fallback() — clean 6-section output when API unavailable
  • Top navbar with app title + Gemini mode badge
  • Empty / welcome state before first question
  • Sidebar: 14 topics + textbook library + clean footer
"""

import hashlib
import os
import random
import time

from dotenv import load_dotenv
load_dotenv()

import google.genai as genai
from google.genai import types as genai_types
import streamlit as st

from rag_pipeline import RAGPipeline, HybridRAGPipeline
from prompt_builder import (
    build_messages,
    context_adequacy_check,
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

/* kill blue links globally */
a, a:link, a:visited, a:hover, a:active,
.stMarkdown a { color: var(--text) !important; text-decoration: none !important; }

/* ── NAVBAR ────────────────────────────────────────────────────── */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 0 14px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 18px;
}
.navbar-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.01em;
}
.navbar-sub {
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 1px;
}
.navbar-badge {
    font-size: 0.70rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    padding: 4px 10px;
    border-radius: 20px;
    border: 1px solid var(--border);
    color: var(--accent);
    background: rgba(56,189,248,0.08);
}

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
    rag = HybridRAGPipeline(faiss_k=12, bm25_k=12, max_chunks=5, use_reranker=True)
    try:
        rag.load_index()
    except FileNotFoundError:
        st.error("Knowledge base not found. Run `python ingest.py --dir data` first.")
        st.stop()
    return rag


@st.cache_resource(show_spinner="Connecting to Gemini…")
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


rag           = get_rag()
gemini_client = get_gemini_client()

GEMINI_MODEL = "models/gemini-2.5-flash"

# Navbar badge — Gemini only
_LLM_MODE = "Gemini 2.5 Flash" if gemini_client is not None else "Offline"


# ══════════════════════════════════════════════════════════════════════
#  LLM CALL FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

# Token budget — 1800 tokens gives full 6-section answers with room to spare.
_MAX_TOKENS = 1800


def _call_gemini(prompt: str, max_tokens: int = _MAX_TOKENS, temp: float = 0.3) -> str:
    """
    Gemini call — exponential backoff on 429 / ResourceExhausted.
    Retries once if response appears truncated.
    """
    if gemini_client is None:
        raise ValueError("Gemini client not initialised.")

    cfg = genai_types.GenerateContentConfig(
        temperature=temp,
        max_output_tokens=max_tokens,
    )
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=cfg,
            )
            result = getattr(response, "text", None) or ""

            # Retry once if response appears truncated
            if result and response_seems_truncated(result):
                retry = gemini_client.models.generate_content(
                    model=GEMINI_MODEL, contents=prompt, config=cfg,
                )
                result2 = getattr(retry, "text", None) or ""
                if result2 and not response_seems_truncated(result2):
                    return result2

            return result

        except Exception as exc:
            err = str(exc).lower()
            is_rate = any(x in err for x in
                          ["429", "quota", "rate", "exhausted", "resource"])
            if is_rate and attempt < max_retries - 1:
                wait = (2 ** (attempt + 1)) + random.uniform(0, 1)
                st.toast(f"⏳ Rate limit — retrying in {wait:.0f}s…", icon="⏳")
                time.sleep(wait)
                continue
            raise

    return ""


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_llm_call(prompt: str) -> str:
    """
    Cached Gemini call.
    Cache key = MD5 of prompt → identical questions use stored answer for 1 hour.
    """
    if gemini_client is not None:
        return _call_gemini(prompt)
    raise ValueError("Gemini client not available. Set GEMINI_API_KEY.")


# ══════════════════════════════════════════════════════════════════════
#  DEFINITION-BOOST RERANKING
#  Elevates chunks that contain definitional language so the LLM
#  gets the most answer-ready passage first.
# ══════════════════════════════════════════════════════════════════════

_DEF_SIGNALS = {
    " is ", " is defined ", " is a ", " is an ",
    "defined as", "refers to", "algorithm is", "method is",
    "technique is", "can be defined", "we define",
}

def _boost_definition_chunks(chunks: list[dict]) -> list[dict]:
    def _score(c: dict) -> float:
        text  = c.get("text", "").lower()
        base  = c.get("rerank_score", c.get("score", 0.0))
        bonus = sum(0.05 for sig in _DEF_SIGNALS if sig in text)
        return base + min(bonus, 0.20)
    return sorted(chunks, key=_score, reverse=True)


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

    st.markdown("**⚙️ Settings**")

    user_level = st.selectbox(
        "Your level",
        ["Beginner", "Intermediate", "Advanced"],
        index=1,
        help="Beginner = analogies first. Intermediate = adds formulas. Advanced = full depth.",
    )

    response_style = st.radio(
        "Response style",
        ["Quick", "Detailed"],
        index=1,
        help="Quick = concise 3-step answer. Detailed = full 6-section explanation.",
    )

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
    st.markdown("**📚 Topics Covered**")
    _topic_labels = [
        "Gradient Descent", "Neural Networks", "Backpropagation",
        "Regularization / Overfitting", "Transformers", "KNN",
        "SVM", "Decision Trees / Random Forests", "Clustering",
        "Reinforcement Learning", "Supervised vs Unsupervised",
        "Loss Functions", "Training Process", "Bias vs Variance",
    ]
    for t in _topic_labels:
        st.markdown(
            f'<span style="font-size:0.77rem;color:#94A3B8">· {t}</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("**📖 Textbook Library**")
    for book in [
        "DL_NeuralNetworks_Nielsen.pdf",
        "ML_Ethem_Alpaydin.pdf",
        "ML_Math_Foundation.pdf",
        "ML_SoftComputing_Kecman.pdf",
    ]:
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
    st.markdown(
        f'<div class="navbar">'
        f'  <div>'
        f'    <div class="navbar-title">📖 ML Tutor</div>'
        f'    <div class="navbar-sub">Concepts explained simply · grounded in textbooks</div>'
        f'  </div>'
        f'  <div class="navbar-badge">⚡ {_LLM_MODE}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Chat history ──────────────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state.history     = []
        st.session_state.last_q_hash = ""

    recent = st.session_state.history[-12:]
    for msg in recent:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Empty state (shown before first question) ─────────────────────
    if not st.session_state.history:
        st.markdown(
            '<div class="empty-state">'
            '  <div class="empty-state-icon">🎓</div>'
            '  <div class="empty-state-title">Ask any machine learning question</div>'
            '  <div class="empty-state-sub">'
            '    Start with gradient descent, neural networks, or backpropagation — '
            '    and get structured, textbook-grounded answers instantly.'
            '  </div>'
            '  <div>'
            '    <span class="empty-chip">What is gradient descent?</span>'
            '    <span class="empty-chip">Explain backpropagation</span>'
            '    <span class="empty-chip">Bias vs variance tradeoff</span>'
            '    <span class="empty-chip">How does SVM work?</span>'
            '    <span class="empty-chip">What is overfitting?</span>'
            '  </div>'
            '</div>',
            unsafe_allow_html=True,
        )

    question = st.chat_input("Ask a machine learning question…")


# ══════════════════════════════════════════════════════════════════════
#  QUESTION HANDLER
# ══════════════════════════════════════════════════════════════════════

if question:

    if not question.strip():
        with col_chat:
            st.warning("Please enter a question.")
        st.stop()

    # Duplicate-call guard
    q_hash = hashlib.md5(question.strip().lower().encode()).hexdigest()
    if q_hash == st.session_state.get("last_q_hash", ""):
        st.stop()
    st.session_state.last_q_hash = q_hash

    st.session_state.history.append({"role": "user", "content": question})
    with col_chat:
        with st.chat_message("user"):
            st.markdown(question)

    # ── Step 1: Retrieval ──────────────────────────────────────────────
    with col_chat:
        with st.status("Generating explanation from textbooks…",
                       expanded=False) as status:
            chunks = rag.retrieve(
                question, top_k=answer_depth, score_threshold=0.20
            )
            if not chunks:
                chunks = rag.retrieve(
                    question, top_k=answer_depth, score_threshold=0.0
                )

            if chunks:
                chunks = _boost_definition_chunks(chunks)
                status.update(
                    label=f"✓ Textbook knowledge loaded ({len(chunks)} passages)",
                    state="complete",
                )
            else:
                status.update(label="⚠ No relevant passages found", state="error")

    if not chunks:
        with col_chat:
            with st.chat_message("assistant"):
                reply = (
                    "I couldn't find relevant information in the textbooks for this question. "
                    "Try rephrasing, or check that the relevant PDF is in `data/` "
                    "and the index has been rebuilt."
                )
                st.markdown(reply)
        st.session_state.history.append({"role": "assistant", "content": reply})
        st.stop()

    # ── Step 2: Build prompt ────────────────────────────────────────────
    messages = build_messages(
        question=question,
        chunks=chunks,
        level=user_level,
        style=response_style,
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

            with st.spinner(""):
                try:
                    full_response = _cached_llm_call(full_prompt)

                    if not full_response.strip():
                        raise ValueError("Empty LLM response.")

                except Exception as exc:
                    _api_error = True
                    low = str(exc).lower()

                    if any(x in low for x in ["429", "quota", "rate", "exhausted"]):
                        st.warning(
                            "⏳ **API rate limit reached.** "
                            "Generating explanation from textbook passages instead…"
                        )
                    elif any(x in low for x in ["401", "api key", "invalid", "permission"]):
                        st.error(
                            "🔑 **API key error.** "
                            "Check `GEMINI_API_KEY` in your `.env` or `.streamlit/secrets.toml`."
                        )
                        # Still show structured fallback — don't crash
                    elif "not initialised" in low:
                        st.info("ℹ️ No LLM connected. Showing textbook explanation.")
                    else:
                        st.warning(
                            f"⚠️ LLM unavailable. "
                            "Showing structured textbook explanation instead."
                        )

                    # Always produce clean structured output — never a raw dump
                    full_response = make_structured_fallback(chunks, question)

            answer_placeholder.markdown(full_response)

            # Post-answer quality notes
            if not _api_error and not llm_response_is_adequate(full_response):
                st.info("ℹ️ The retrieved context may not fully cover this question.")

            hint = detect_visual_hint(question, chunks)
            if hint:
                st.info(f"📐 {hint}")

            if show_chunks:
                with st.expander("🔬 Retrieved passages (dev)", expanded=False):
                    for i, c in enumerate(chunks, 1):
                        st.markdown(
                            f"**Passage {i}** — `{c.get('source','?')}` "
                            f"p.{c.get('page','?')}  "
                            f"score: {c.get('rerank_score', c.get('score','?'))}"
                        )
                        txt = c.get("text", "")
                        st.text(txt[:400] + ("…" if len(txt) > 400 else ""))

    st.session_state.history.append(
        {"role": "assistant", "content": full_response}
    )

    # ══════════════════════════════════════════════════════════════════
    #  RIGHT PANEL — videos + sources
    # ══════════════════════════════════════════════════════════════════

    with right_slot.container():

        if show_videos:
            st.markdown(
                '<p class="panel-header">🎬 Video Explanations</p>',
                unsafe_allow_html=True,
            )

            concept_videos = recommend_videos(question)

            if concept_videos:
                for v in concept_videos:
                    st.markdown(
                        f'<div class="vc">'
                        f'  <a href="{v["url"]}" target="_blank">{v["title"]}</a>'
                        f'  <div class="ch">{v["channel"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<p style="font-size:0.78rem;color:#475569;padding:4px 2px">'
                    'No video match for this query.</p>',
                    unsafe_allow_html=True,
                )

            if wants_practice(question):
                pv = recommend_practice_video(question)
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
            seen: set  = set()
            refs: list = []
            for c in chunks:
                key = (c.get("source", "?"), c.get("page", "?"))
                if key not in seen:
                    seen.add(key)
                    refs.append(key)

            if refs:
                tags = "".join(
                    f'<span class="src-tag">📄 {s} — p.{p}</span>'
                    for s, p in refs
                )
                st.markdown(
                    f'<div style="display:flex;flex-wrap:wrap;gap:4px">{tags}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<p style="font-size:0.77rem;color:#475569">No source metadata.</p>',
                    unsafe_allow_html=True,
                )