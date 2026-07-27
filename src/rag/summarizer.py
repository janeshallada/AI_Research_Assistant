"""
Document Summarization Engine
------------------------------
Produces a structured, multi-tier summary (Executive Summary, Technical
Summary, Bullet Point Breakdown, Key Takeaways) for a single document,
grounded strictly in that document's indexed chunks.
"""
from typing import Dict, Any

from src.rag.llm import get_llm
from src.vector_store.manager import get_vector_store

SUMMARY_PROMPT = """You are an AI Research Assistant. Using ONLY the document excerpts below,
produce a structured summary with these exact sections:

### Executive Summary
(2-3 sentences, high level, for a non-technical audience)

### Technical Summary
(a more detailed paragraph covering methodology, data, and results)

### Bullet Point Breakdown
(5-8 concise bullet points of the main content)

### Key Takeaways
(3-5 bullet points of the most important conclusions/implications)

Document excerpts:
{context}
"""


class DocumentSummarizer:
    def __init__(self):
        self.vector_store = get_vector_store()
        self.llm = get_llm()

    def summarize(self, doc_id: str, file_name: str, max_chunks: int = 40) -> Dict[str, Any]:
        result = self.vector_store.collection.get(
            where={"doc_id": doc_id}, include=["documents", "metadatas"]
        )
        if not result["ids"]:
            return {"summary": "No indexed content found for this document.", "doc_id": doc_id}

        # Order chunks by page number so the summary follows document flow.
        pairs = sorted(
            zip(result["documents"], result["metadatas"]),
            key=lambda p: p[1].get("page_number", 0),
        )[:max_chunks]
        context = "\n\n".join(text for text, _ in pairs)

        prompt = SUMMARY_PROMPT.format(context=context)
        summary_text = self.llm.complete(prompt)

        return {"doc_id": doc_id, "file_name": file_name, "summary": summary_text}
