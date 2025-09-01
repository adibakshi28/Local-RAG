import re
from typing import List, Dict
from pypdf import PdfReader

def read_pdfs(pdf_dir: str) -> List[Dict]:
    import glob, os
    docs = []
    for path in glob.glob(f"{pdf_dir}/*.pdf"):
        reader = PdfReader(path)
        text = "\n".join(p.extract_text() or "" for p in reader.pages)
        text = normalize(text)
        docs.append({"path": path, "text": text})
    return docs

def normalize(text: str) -> str:
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def chunk_text(text: str, size: int = 1200, overlap: int = 150) -> List[str]:
    # character-based chunking with sentence-friendly cuts
    if overlap >= size:
        overlap = max(0, size // 10)
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunk = text[start:end]
        if end < n:
            cut = text.rfind(".", start, end)
            if cut != -1 and cut > start + int(size * 0.6):
                end = cut + 1
                chunk = text[start:end]
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - overlap, start + 1)
    return chunks
