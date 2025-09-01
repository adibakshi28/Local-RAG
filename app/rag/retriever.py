from typing import List, Dict, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from app.core.config import get_settings

def mmr_select(query_vec, doc_vecs, k=6, lambda_mult=0.7):
    """
    Simple Maximal Marginal Relevance over cosine sims (assumes vectors are normalized).
    query_vec: (d,), doc_vecs: List[(d,)], returns selected indices in order
    """
    import numpy as np
    doc_vecs = np.array(doc_vecs)
    sims = doc_vecs @ query_vec  # cosine since normalized
    selected, candidates = [], list(range(len(doc_vecs)))
    if not candidates:
        return []

    # pick the best first
    first = int(sims.argmax())
    selected.append(first)
    candidates.remove(first)

    while len(selected) < min(k, len(doc_vecs)):
        max_score, best_idx = -1e9, None
        for c in candidates:
            sim_to_query = sims[c]
            sim_to_selected = max([doc_vecs[c] @ doc_vecs[s] for s in selected]) if selected else 0.0
            score = lambda_mult * sim_to_query - (1 - lambda_mult) * sim_to_selected
            if score > max_score:
                max_score, best_idx = score, c
        selected.append(best_idx)
        candidates.remove(best_idx)

    return selected

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

        # build BM25 corpus once (small/medium corpora ok)
        # fetch up to 100k docs; increase if you need more
        all_docs = self.coll.get(include=["documents", "metadatas"], limit=100000)
        self.docs = all_docs.get("documents", [])
        self.metas = all_docs.get("metadatas", [])
        self.ids = all_docs.get("ids", [])
        # BM25 tokenization is simple whitespace
        self.bm25 = BM25Okapi([d.split() for d in self.docs]) if self.docs else None

        # cache doc embeddings (optional, speeds MMR)
        try:
            import numpy as np
            self.doc_vecs = None
            if self.docs:
                # we can lazily build if needed; here we pre-build
                self.doc_vecs = self.embedder.encode(self.docs, normalize_embeddings=True, convert_to_numpy=True)
        except Exception:
            self.doc_vecs = None

    def _vector_search(self, query: str, k: int, prefetch: int = 50):
        q = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        res = self.coll.query(
            query_embeddings=[q.tolist()],
            n_results=prefetch,
            include=["metadatas", "documents", "distances", "embeddings"]
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        embs = res.get("embeddings", [[]])[0]

        # MMR to diversify top 'prefetch' before cutting to top k
        try:
            idxs = mmr_select(q, embs, k=k, lambda_mult=0.7)
        except Exception:
            idxs = list(range(min(k, len(docs))))

        hits = []
        for i in idxs:
            hits.append({
                "text": docs[i],
                "source": metas[i].get("source", "unknown"),
                "page": metas[i].get("page", None),
                "chunk_id": metas[i].get("chunk_id", ""),
                "score": 1.0 - float(dists[i])
            })
        return hits, q

    def _bm25_search(self, query: str, k: int):
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(query.split())
        # top k from BM25
        top = sorted(enumerate(scores), key=lambda t: t[1], reverse=True)[:k]
        hits = []
        for idx, sc in top:
            hits.append({
                "text": self.docs[idx],
                "source": self.metas[idx].get("source", "unknown") if self.metas else "unknown",
                "page": self.metas[idx].get("page", None) if self.metas else None,
                "chunk_id": self.metas[idx].get("chunk_id", "") if self.metas else "",
                "score": float(sc)  # lexical score, not normalized
            })
        return hits

    def search(self, query: str, k: int) -> List[Dict]:
        # hybrid: union of BM25 top-k and vector+MMR top-k, then rerank with cross-encoder
        vec_hits, _ = self._vector_search(query, k=k, prefetch=max(30, k*5))
        bm25_hits = self._bm25_search(query, k=k)

        # union by chunk_id
        pool = {}
        for h in vec_hits + bm25_hits:
            pool[h["chunk_id"]] = h
        merged = list(pool.values())

        # optional cross-encoder rerank
        if self.reranker and len(merged) > 1:
            pairs = [(query, h["text"]) for h in merged]
            rr = self.reranker.predict(pairs)
            merged = [h for _, h in sorted(zip(rr, merged), key=lambda t: t[0], reverse=True)]

        # per-source cap to avoid one doc dominating
        capped, per_src, cap = [], {}, max(2, k//2)
        for h in merged:
            src = h.get("source", "unknown")
            per_src[src] = per_src.get(src, 0) + 1
            if per_src[src] <= cap:
                capped.append(h)
            if len(capped) >= k:
                break

        return capped
