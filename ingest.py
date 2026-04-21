"""
ingest.py
PDF extraction → clean → fix sentences → fix spaced words
→ filter bad chunks → attach metadata → build FAISS index

Key improvement over previous version:
  x_tolerance=1  (was 2) — tighter horizontal grouping fixes smashed words
  keep_blank_chars=True  — preserves spacing signals for word boundary detection
Both changes work together to eliminate concatenated tokens like
"Thisisasentence" that degraded LLM output quality.
"""

import argparse
import re
import sys
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    sys.exit("Run: pip install pdfplumber")

from rag_pipeline import RAGPipeline, chunk_text


# ══════════════════════════════════════════════════════════════════════
#  PDF EXTRACTION
# ══════════════════════════════════════════════════════════════════════

def extract_pdf_pages(
    path:       Path,
    skip_pages: int = 10,
) -> list[tuple[int, str]]:
    """
    Returns (1-based page number, raw text) for every extractable page.
    Skips the first `skip_pages` pages (cover, TOC, preface, copyright).

    Extraction settings:
      x_tolerance=1        — tighter column grouping; prevents character merging
      y_tolerance=3        — standard line grouping
      keep_blank_chars=True — preserves inter-word spacing cues
    """
    pages: list[tuple[int, str]] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            if i < skip_pages:
                continue
            text = page.extract_text(
                x_tolerance=1,           # ← tightened (was 2)
                y_tolerance=3,
                keep_blank_chars=True,   # ← new: preserves spacing signals
            )
            if text and text.strip():
                pages.append((i + 1, text))
            else:
                print(f"  ⚠️  Page {i + 1} — no text (image-based or blank)")
    return pages


# ══════════════════════════════════════════════════════════════════════
#  TEXT CLEANING
# ══════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)       # rejoin hyphenated line breaks
    text = re.sub(r"http\S+", "", text)           # remove URLs
    text = re.sub(r"\(cid:\d+\)", "", text)       # pdfplumber encoding artifacts

    lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.fullmatch(r"[-–\s\d]+", line):     # bare page numbers
            continue
        if re.search(r"\.{4,}", line):            # TOC dot-leaders
            continue
        if len(line) < 10:                        # micro-fragments
            continue
        lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"[^\x20-\x7E\n]", " ", text) # strip non-printable chars
    text = re.sub(r"[ \t]+", " ", text)          # collapse runs of spaces/tabs
    return text.strip()


# ══════════════════════════════════════════════════════════════════════
#  SENTENCE / WORD FIXES
# ══════════════════════════════════════════════════════════════════════

def fix_sentences(text: str) -> str:
    """Join soft line-breaks; preserve paragraph breaks."""
    text = re.sub(r"(?<![.?!])\n", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text


def fix_spaced_words(text: str) -> str:
    """
    Collapse spaced-out acronyms / titles like 'M A T H' → 'MATH'.
    Pattern: three or more alternating single-letter + space sequences.
    """
    return re.sub(
        r"\b(?:[A-Za-z]\s){2,}[A-Za-z]\b",
        lambda m: m.group(0).replace(" ", ""),
        text,
    )


# ══════════════════════════════════════════════════════════════════════
#  BAD CHUNK FILTER
# ══════════════════════════════════════════════════════════════════════

_RE_BIBLIO   = re.compile(r"\[\d{1,3}\]|\bdoi:\S+|arxiv:\S+", re.I)
_RE_CAPTION  = re.compile(
    r"^(Figure|Fig\.|Table|Algorithm|Listing|Appendix)\s+\d+", re.I
)
_RE_EQUATION = re.compile(r"\\frac|\\sum|\\int|\$\$|\\\[|\\\(")
_RE_MATRIX   = re.compile(r"(\d+[\s,]+){4,}\d+")
_RE_INDEX    = re.compile(r"^[A-Z][a-z]+,\s+\d+")
_RE_TOC_LINE = re.compile(r"\b\w[\w\s]{3,40}\s+\d{1,4}\s*$")


def is_bad_chunk(chunk: str) -> bool:
    # 1. Length bounds
    if len(chunk) < 80:   return True   # too short — no useful content
    if len(chunk) > 1200: return True   # too long  — likely merged mess

    # 2. Alpha density — filters symbol tables, matrices, equation lines
    alpha = sum(c.isalpha() for c in chunk)
    if alpha / len(chunk) < 0.55: return True

    # 3. Real-word count
    real_words = re.findall(r"[a-zA-Z]{3,}", chunk)
    if len(real_words) < 8: return True

    # 4. Average word length — rejects abbreviation / symbol soup
    avg_len = sum(len(w) for w in real_words) / len(real_words)
    if avg_len < 3.5: return True

    # 5. Table of Contents — >50 % of lines end with a bare page number
    lines = [l.strip() for l in chunk.splitlines() if l.strip()]
    if len(lines) >= 2:
        toc_lines = sum(1 for l in lines if _RE_TOC_LINE.search(l))
        if toc_lines / len(lines) > 0.5: return True

    # 6. High digit density — TOC / index / table noise
    if sum(c.isdigit() for c in chunk) / len(chunk) > 0.15: return True

    # 7. Bibliography / reference entries
    if _RE_BIBLIO.search(chunk):  return True

    # 8. Figure / table captions
    if _RE_CAPTION.match(chunk.strip()): return True

    # 9. LaTeX equation-heavy lines
    if _RE_EQUATION.search(chunk): return True

    # 10. Matrix / data table rows
    if _RE_MATRIX.search(chunk): return True

    # 11. Book index entries
    if _RE_INDEX.match(chunk.strip()): return True

    # 12. Repetitive headers / footers (low unique-word ratio)
    words = chunk.split()
    if len(words) > 6 and len(set(words)) / len(words) < 0.55: return True

    # 13. Pseudocode / indented code blocks
    code_lines = sum(
        1 for line in chunk.splitlines()
        if line.startswith(("    ", "\t"))
    )
    if code_lines > 2: return True

    # 14. Must have ≥ 2 sentence-ending punctuation marks
    if len(re.findall(r"[.!?](?:\s|$)", chunk)) < 2: return True

    return False


# ══════════════════════════════════════════════════════════════════════
#  QUALITY REPORT
# ══════════════════════════════════════════════════════════════════════

def print_quality_report(all_chunks: list[dict]) -> None:
    if not all_chunks:
        return
    print("\n── Quality check — 5 sample chunks ──")
    step    = max(1, len(all_chunks) // 5)
    samples = [all_chunks[i * step] for i in range(5)
               if i * step < len(all_chunks)]
    for s in samples:
        print(f"\n  [{s['source']}  p.{s['page']}]")
        print(f"  {s['text'][:220]}")
        print(f"  chars={len(s['text'])}  words={len(s['text'].split())}")

    print("\n── Per-source chunk counts ──")
    from collections import Counter
    counts = Counter(c["source"] for c in all_chunks)
    for src, n in counts.most_common():
        print(f"  {src:<45} {n:>5} chunks")


# ══════════════════════════════════════════════════════════════════════
#  MAIN INGESTION
# ══════════════════════════════════════════════════════════════════════

def ingest_directory(doc_dir: str, skip_pages: int = 10) -> list[dict]:
    base      = Path(doc_dir)
    pdf_files = list(base.rglob("*.pdf"))

    if not pdf_files:
        sys.exit(f"No PDFs found in '{doc_dir}'.")

    all_chunks: list[dict] = []

    for pdf_path in pdf_files:
        source_name = pdf_path.name
        print(f"\n📄 {source_name}  (skipping first {skip_pages} pages)")

        pages      = extract_pdf_pages(pdf_path, skip_pages=skip_pages)
        raw_total  = 0
        good_total = 0

        for page_num, page_text in pages:
            clean = clean_text(page_text)
            clean = fix_sentences(clean)
            clean = fix_spaced_words(clean)

            text_chunks = chunk_text(clean)
            raw_total  += len(text_chunks)

            for text in text_chunks:
                if is_bad_chunk(text):
                    continue
                all_chunks.append({
                    "text":   text,
                    "source": source_name,
                    "page":   page_num,
                })
                good_total += 1

        removed = raw_total - good_total
        pct     = 100 * removed // max(raw_total, 1)
        print(
            f"   Raw: {raw_total:,}  |  "
            f"Removed: {removed:,} ({pct}%)  |  "
            f"Kept: {good_total:,}"
        )

    print(f"\n✅ Total good chunks: {len(all_chunks):,}")
    print_quality_report(all_chunks)
    return all_chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest PDFs and build FAISS index for ML Tutor"
    )
    parser.add_argument("--dir",  default="data",
                        help="Directory containing PDF files (default: data)")
    parser.add_argument("--skip", type=int, default=10,
                        help="Pages to skip at start of each PDF (default: 10)")
    args = parser.parse_args()

    chunks = ingest_directory(args.dir, skip_pages=args.skip)

    print("\n🔢 Building FAISS index…")
    rag = RAGPipeline()
    rag.build_index_from_chunks(chunks)
    print("🚀 Done!   Run:  streamlit run app.py")


if __name__ == "__main__":
    main()