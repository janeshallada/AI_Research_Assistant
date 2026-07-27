"""
Vector Database Indexing & Retrieval
-------------------------------------
Persists chunk embeddings in ChromaDB (local, file-backed, no external
service required) alongside metadata (doc_id, file_name, page_number) and
exposes three search modes:

  - semantic_search : dense vector cosine-similarity search
  - keyword_search   : sparse BM25-style keyword matching over stored chunk text
  - hybrid_search     : merges both rankings (weighted score fusion)

When each mode is most appropriate (see README for full discussion):
  - Semantic: conceptual / paraphrased questions ("what are the drawbacks of
    this approach?") where exact wording won't match the source text.
  - Keyword: precise lookups (IDs, exact terms, acronyms, numbers) where
    embedding similarity can under-rank an exact but rare token.
  - Hybrid: general-purpose default — combines recall of semantic search
    with the precision of exact keyword hits.
"""
import logging
import re
from collections import Counter
from typing import List, Dict, Any, Optional

import chromadb

from config.settings import settings
from src.vector_store.embeddings import get_embedding_provider

logger = logging.getLogger(__name__)

COLLECTION_NAME = "document_chunks"


class VectorStoreManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.vector_db_dir)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        self._embedder = None  # lazy-loaded (avoids loading the model on every import)

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedding_provider()
        return self._embedder

    # ---------- Indexing ----------
    def index_chunks(self, chunks: List[Dict[str, Any]], file_name: str) -> None:
        if not chunks:
            return
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_documents(texts)
        ids = [c["chunk_id"] for c in chunks]
        metadatas = [
            {"doc_id": c["doc_id"], "file_name": file_name, "page_number": c["page_number"]}
            for c in chunks
        ]
        self.collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        logger.info("Indexed %d chunks for '%s' into the vector store.", len(chunks), file_name)

    def delete_document(self, doc_id: str) -> None:
        self.collection.delete(where={"doc_id": doc_id})

    def total_chunks(self) -> int:
        return self.collection.count()

    # ---------- Search ----------
    def semantic_search(
        self, query: str, top_k: int = 4, doc_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        query_embedding = self.embedder.embed_query(query)
        where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k, where=where
        )
        return self._format_results(results)

    def keyword_search(
        self, query: str, top_k: int = 4, doc_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Simple BM25-style term-frequency scoring over all stored chunk documents.
        Adequate for a self-contained project without standing up a separate
        search engine (e.g. Elasticsearch)."""
        where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
        all_docs = self.collection.get(where=where, include=["documents", "metadatas"])
        terms = [t for t in re.findall(r"\w+", query.lower()) if t]
        if not terms or not all_docs["ids"]:
            return []

        scored = []
        for i, doc_text in enumerate(all_docs["documents"]):
            tokens = re.findall(r"\w+", doc_text.lower())
            counts = Counter(tokens)
            score = sum(counts.get(t, 0) for t in terms)
            if score > 0:
                scored.append((score, i))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]
        return [
            {
                "chunk_id": all_docs["ids"][i],
                "text": all_docs["documents"][i],
                "metadata": all_docs["metadatas"][i],
                "score": float(score),
            }
            for score, i in top
        ]

    def hybrid_search(
        self, query: str, top_k: int = 4, doc_ids: Optional[List[str]] = None,
        semantic_weight: float = 0.7,
    ) -> List[Dict[str, Any]]:
        sem = self.semantic_search(query, top_k=top_k * 2, doc_ids=doc_ids)
        kw = self.keyword_search(query, top_k=top_k * 2, doc_ids=doc_ids)

        def normalize(results, key="score"):
            if not results:
                return {}
            vals = [r[key] for r in results]
            lo, hi = min(vals), max(vals)
            rng = (hi - lo) or 1.0
            return {r["chunk_id"]: (r[key] - lo) / rng for r in results}

        sem_norm = normalize(sem)
        kw_norm = normalize(kw)
        all_ids = set(sem_norm) | set(kw_norm)

        fused = []
        lookup = {r["chunk_id"]: r for r in (sem + kw)}
        for cid in all_ids:
            fused_score = semantic_weight * sem_norm.get(cid, 0.0) + (1 - semantic_weight) * kw_norm.get(cid, 0.0)
            item = dict(lookup[cid])
            item["score"] = fused_score
            fused.append(item)

        fused.sort(key=lambda r: r["score"], reverse=True)
        return fused[:top_k]

    @staticmethod
    def _format_results(results) -> List[Dict[str, Any]]:
        formatted = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0] if results.get("distances") else [None] * len(ids)
        for cid, text, meta, dist in zip(ids, docs, metas, dists):
            similarity = (1 - dist) if dist is not None else None
            formatted.append({"chunk_id": cid, "text": text, "metadata": meta, "score": similarity})
        return formatted


_vector_store_instance: Optional[VectorStoreManager] = None


def get_vector_store() -> VectorStoreManager:
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStoreManager()
    return _vector_store_instance
