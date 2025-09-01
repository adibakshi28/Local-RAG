import os
from typing import List, Dict
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from .utils import read_pdfs, chunk_text
from app.core.config import get_settings

def _client():
    s = get_settings()
    os.makedirs(s.CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(
        path=s.CHROMA_DIR,
        settings=ChromaSettings(anonymized_telemetry=False)
    )

def _collection(client):
    return client.get_or_create_collection(
        name="papers",
        metadata={"hnsw:space": "cosine"}
    )

def build_index():
    s = get_settings()
    client = _client()
    # reset collection (safe if not present)
    try:
        client.delete_collection("papers")
    except Exception:
        pass
    coll = _collection(client)

    docs = read_pdfs(s.PDF_DIR)
    if not docs:
        raise RuntimeError(f"No PDFs found in {s.PDF_DIR}")

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    records: List[Dict] = []
    for d in docs:
        chunks = chunk_text(d["text"], size=s.CHUNK_SIZE, overlap=s.CHUNK_OVERLAP)
        fname = os.path.basename(d["path"])
        for i, ch in enumerate(chunks):
            records.append({
                "id": f"{fname}:::{i}",
                "document": ch,
                "metadata": {"source": fname, "chunk_id": f"{fname}:::{i}"}
            })

    if not records:
        raise RuntimeError("No text extracted from PDFs.")

    texts = [r["document"] for r in records]
    embs = embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    coll.add(
        ids=[r["id"] for r in records],
        documents=texts,
        metadatas=[r["metadata"] for r in records],
        embeddings=embs.tolist()
    )

    return {"vectors": len(records), "sources": sorted({r['metadata']['source'] for r in records})}
