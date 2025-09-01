from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import os, shutil

from app.core.config import get_settings
from app.rag.ingest import build_index
from app.rag.retriever import Retriever
from app.rag.llm import generate_answer

app = FastAPI(title="ChromaSeek RAG (API)")

# ---------- API MODELS ----------
class AskIn(BaseModel):
    question: str
    top_k: Optional[int] = None

# ---------- UTILS ----------
def list_pdfs() -> List[dict]:
    s = get_settings()
    os.makedirs(s.PDF_DIR, exist_ok=True)
    items = []
    for name in sorted(os.listdir(s.PDF_DIR)):
        p = os.path.join(s.PDF_DIR, name)
        if os.path.isfile(p) and name.lower().endswith(".pdf"):
            items.append({"filename": name, "bytes": os.path.getsize(p)})
    return items

def chroma_count() -> int:
    # non-fatal stats helper
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        s = get_settings()
        client = chromadb.PersistentClient(
            path=s.CHROMA_DIR, settings=ChromaSettings(anonymized_telemetry=False)
        )
        coll = client.get_or_create_collection(name="papers", metadata={"hnsw:space": "cosine"})
        return coll.count()
    except Exception:
        return 0

# ---------- API ----------
@app.get("/api/health")
def health():
    return {"ok": True}

@app.get("/api/pdfs")
def get_pdfs():
    return {"pdfs": list_pdfs()}

@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)):
    s = get_settings()
    os.makedirs(s.PDF_DIR, exist_ok=True)
    saved = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF allowed: {f.filename}")
        dest = os.path.join(s.PDF_DIR, f.filename)
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append({"filename": f.filename, "bytes": os.path.getsize(dest)})
    return {"saved": saved, "total": len(saved)}

@app.post("/api/ingest")
def ingest():
    stats = build_index()
    stats["collection_count"] = chroma_count()
    return {"status": "ok", **stats}

@app.get("/api/stats")
def stats():
    return {
        "pdfs": list_pdfs(),
        "collection_count": chroma_count(),
    }

@app.post("/api/ask")
def ask(body: AskIn):
    s = get_settings()
    k = body.top_k or s.TOP_K
    ret = Retriever()
    hits = ret.search(body.question, k)
    out = generate_answer(body.question, hits)
    out["retrieved"] = len(hits)
    return out

# Optional: reset the on-disk Chroma store
@app.post("/api/reset_index")
def reset_index():
    s = get_settings()
    try:
        import shutil
        shutil.rmtree(s.CHROMA_DIR, ignore_errors=True)
        os.makedirs(s.CHROMA_DIR, exist_ok=True)
        return {"status": "ok", "message": "Chroma storage cleared. Run /api/ingest next."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- STATIC UI ----------
app.mount("/", StaticFiles(directory="app/ui", html=True), name="ui")
