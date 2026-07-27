"""
Multi-Document Comparison Engine
----------------------------------
Compares 2+ uploaded documents across methodology, advantages/disadvantages,
similarities, differences, conclusions, and implementation approach, using
only retrieved context from the selected documents (no hallucinated facts).
"""
from typing import List, Dict, Any

from src.rag.llm import get_llm
from src.vector_store.manager import get_vector_store

COMPARISON_PROMPT = """You are an AI Research Assistant comparing multiple documents. Using ONLY
the excerpts provided below (grouped by document), produce a structured
comparison covering:

### Methodologies
### Advantages / Disadvantages
### Similarities
### Differences
### Conclusions
### Implementation Approaches

If a document does not provide enough information for a section, say so
explicitly rather than guessing.

{context_blocks}
"""


class DocumentComparator:
    def __init__(self):
        self.vector_store = get_vector_store()
        self.llm = get_llm()

    def compare(self, doc_id_to_name: Dict[str, str], max_chunks_per_doc: int = 20) -> Dict[str, Any]:
        if len(doc_id_to_name) < 2:
            raise ValueError("At least two documents are required for comparison.")

        context_blocks = ""
        for doc_id, file_name in doc_id_to_name.items():
            result = self.vector_store.collection.get(
                where={"doc_id": doc_id}, include=["documents", "metadatas"]
            )
            pairs = sorted(
                zip(result["documents"], result["metadatas"]),
                key=lambda p: p[1].get("page_number", 0),
            )[:max_chunks_per_doc]
            doc_context = "\n".join(text for text, _ in pairs) or "(no indexed content found)"
            context_blocks += f"\n=== Document: {file_name} (doc_id={doc_id}) ===\n{doc_context}\n"

        prompt = COMPARISON_PROMPT.format(context_blocks=context_blocks)
        comparison_text = self.llm.complete(prompt)

        return {
            "documents_compared": list(doc_id_to_name.values()),
            "comparison": comparison_text,
        }
