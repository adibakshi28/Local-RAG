from typing import List, Dict
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer, CrossEncoder
from app.core.config import get_settings

class Retriever:
    def __init__(self):
        s = get_settings()
        self.s = s
        self.client = chromadb.PersistentClient(
            path=s.CHROMA_DIR,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.coll = self.client.get_or_create_collection(
            name="papers",
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") if s.USE_RERANKER else None

    def search(self, query: str, k: int) -> List[Dict]:
        qv = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()
        res = self.coll.query(
            query_embeddings=[qv],
            n_results=k,
            include=["metadatas", "documents", "distances"]
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        hits = []
        for doc, meta, dist in zip(docs, metas, dists):
            hits.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "chunk_id": meta.get("chunk_id", ""),
                "score": 1.0 - float(dist)  # cosine sim proxy
            })

        if self.reranker and len(hits) > 2:
            pairs = [(query, h["text"]) for h in hits]
            rr = self.reranker.predict(pairs)
            hits = [h for _, h in sorted(zip(rr, hits), key=lambda t: t[0], reverse=True)]

        return hits
