# Local RAG

A CPU-friendly Retrieval-Augmented Generation (RAG) service:

- **Embeddings & retrieval:** local, free, and persistent with **SentenceTransformers + ChromaDB**
- **LLM generation:** **DeepSeek** (OpenAI-compatible REST) using your credits
- **API:** FastAPI with two endpoints (`/ingest`, `/ask`) and a **built-in web UI** at `/`

Works great for a handful of PDFs (or dozens) without a GPU. Built for simple deployment and hacking.

---

## Features

- Upload PDFs → chunk → embed → store in **ChromaDB** (persistent on disk)
- Query → retrieve top-K chunks → **DeepSeek** composes the final answer
- Returns **answer + sources + retrieved passages**
- Minimal **web UI** for testing (no Node/React required)
- CPU-only, zero cloud vector DB, easy to extend

---

## Tech

- **FastAPI**, **Uvicorn**
- **ChromaDB** (SQLite HNSW)
- **SentenceTransformers**: `all-MiniLM-L6-v2` (CPU OK)
- **Optional reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **DeepSeek** via **plain `requests`** (`/v1/chat/completions`)

---

