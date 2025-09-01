import os, json
from typing import List, Dict
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from app.core.config import get_settings
from .utils import read_pdf_pages, page_aware_chunks, sha256_file

MANIFEST = "manifest.json"  # saved under CHROMA_DIR

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

def _load_manifest(dir_path: str) -> Dict:
    path = os.path.join(dir_path, MANIFEST)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"files": {}}

def _save_manifest(dir_path: str, data: Dict):
    path = os.path.join(dir_path, MANIFEST)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def build_index():
    s = get_settings()
    client = _client()
    coll = _collection(client)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # manifest for incremental ingest
    manifest = _load_manifest(s.CHROMA_DIR)

    # scan PDFs
    os.makedirs(s.PDF_DIR, exist_ok=True)
    pdfs = [f for f in os.listdir(s.PDF_DIR) if f.lower().endswith(".pdf")]
    added, updated, skipped = 0, 0, 0
    all_records: List[Dict] = []

    for fname in sorted(pdfs):
        fpath = os.path.join(s.PDF_DIR, fname)
        file_hash = sha256_file(fpath)
        prev = manifest["files"].get(fname)

        # if unchanged, skip re-embedding
        if prev and prev.get("hash") == file_hash:
            skipped += 1
            continue

        # delete old chunks for this source (if any)
        try:
            coll.delete(where={"source": fname})
        except Exception:
            pass

        # extract page-by-page, chunk with page numbers
        records = []
        for page_idx, text in read_pdf_pages(fpath):
            chunks = page_aware_chunks(
                text,
                page_index=page_idx,
                target_tokens=450,
                overlap_tokens=60
            )
            for i, ch in enumerate(chunks):
                records.append({
                    "id": f"{fname}:::{page_idx}:::{i}",
                    "document": ch["text"],
                    "metadata": {
                        "source": fname,
                        "page": page_idx,
                        "chunk_id": f"{fname}:::{page_idx}:::{i}"
                    }
                })

        # embed + add
        if records:
            texts = [r["document"] for r in records]
            embs = embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

            coll.add(
                ids=[r["id"] for r in records],
                documents=texts,
                metadatas=[r["metadata"] for r in records],
                embeddings=embs.tolist()
            )
            updated += 1 if prev else 0
            added += 0 if prev else 1

            # save doc stats to manifest
            manifest["files"][fname] = {
                "hash": file_hash,
                "chunks": len(records)
            }

        all_records.extend(records)

    _save_manifest(s.CHROMA_DIR, manifest)

    return {
        "status": "ok",
        "sources": sorted([f for f in manifest["files"].keys()]),
        "added_files": added,
        "updated_files": updated,
        "skipped_files": skipped,
        "vectors": coll.count()
    }
