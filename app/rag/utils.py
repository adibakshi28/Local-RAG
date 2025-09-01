import hashlib
from typing import List, Dict, Tuple
import fitz  # PyMuPDF

# simple BERT-ish token counter using whitespace + punctuation (fast, no extra deps)
def _rough_token_count(text: str) -> int:
    # not exact tokens, but close enough for stable chunk sizes
    return len(text.replace("\n", " ").split())

def read_pdf_pages(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Returns list of (page_index, page_text). Uses PyMuPDF for better extraction + page ids.
    """
    pages = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text") or ""
            pages.append((i, text))
    return pages

def page_aware_chunks(text: str, page_index: int, target_tokens: int = 450, overlap_tokens: int = 60) -> List[Dict]:
    """
    Token-ish chunking (approx). Tries to keep ~target_tokens per chunk with ~overlap_tokens overlap.
    Returns a list of dicts with 'text' and 'page'.
    """
    words = text.replace("\r", " ").split()
    if not words:
        return []

    chunks = []
    start = 0
    n = len(words)
    size = target_tokens
    overlap = min(overlap_tokens, size // 3)

    while start < n:
        end = min(start + size, n)
        # prefer to end on sentence boundary if we can
        # scan backwards up to 1/3 of the window for a period
        cut = None
        for j in range(end-1, max(start, end - size // 3) - 1, -1):
            if words[j].endswith((".", "!", "?")):
                cut = j + 1
                break
        if cut and cut > start + size * 0.5:
            end = cut

        chunk_words = words[start:end]
        if chunk_words:
            chunks.append({
                "text": " ".join(chunk_words).strip(),
                "page": page_index
            })

        if end >= n:
            break
        start = max(end - overlap, start + 1)

    return chunks

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()
