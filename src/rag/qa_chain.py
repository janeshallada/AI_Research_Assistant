"""
Retrieval-Augmented Generation question-answering with strict context
grounding, source citations (document + page), and conversation memory.
"""
import logging
from typing import Optional, List, Dict, Any

from config.settings import settings
from src.rag.llm import get_llm
from src.vector_store.manager import get_vector_store

logger = logging.getLogger(__name__)

FALLBACK_MESSAGE = "I cannot determine the answer from the provided documents."

PROMPT_TEMPLATE = """You are an AI Research Assistant. Answer the user's question using ONLY the
provided document context below. Do not use outside knowledge. If the context
does not contain sufficient information to answer, state clearly: "{fallback}"

Conversation History:
{history}

Context:
{context}

Question: {question}

Provide a clear, direct answer followed by an explicit list of source documents and page references.
"""


class RAGQuestionAnswering:
    def __init__(self):
        self.vector_store = get_vector_store()
        self.llm = get_llm()

    def answer_question(
        self,
        query: str,
        doc_ids: Optional[List[str]] = None,
        history_messages: Optional[List[Dict[str, str]]] = None,
        search_mode: str = "hybrid",
        top_k: int = None,
    ) -> Dict[str, Any]:
        top_k = top_k or settings.retrieval_top_k

        # Conversation memory: resolve follow-up references by feeding recent
        # turns into the prompt so pronouns ("it", "its", "their") resolve
        # against the prior discussion.
        history_str = self._format_history(history_messages)
        resolved_query = self._contextualize_query(query, history_messages)

        if search_mode == "semantic":
            docs = self.vector_store.semantic_search(resolved_query, top_k=top_k, doc_ids=doc_ids)
        elif search_mode == "keyword":
            docs = self.vector_store.keyword_search(resolved_query, top_k=top_k, doc_ids=doc_ids)
        else:
            docs = self.vector_store.hybrid_search(resolved_query, top_k=top_k, doc_ids=doc_ids)

        if not docs:
            return {
                "answer": FALLBACK_MESSAGE,
                "citations": [],
                "retrieved_context": [],
                "confidence": 0.0,
            }

        context_str, citations = self._build_context(docs)

        prompt = PROMPT_TEMPLATE.format(
            fallback=FALLBACK_MESSAGE,
            history=history_str or "(none)",
            context=context_str,
            question=query,
        )

        answer = self.llm.complete(prompt)
        confidence = self._estimate_confidence(docs)

        return {
            "answer": answer,
            "citations": citations,
            "retrieved_context": [d["text"] for d in docs],
            "confidence": confidence,
        }

    @staticmethod
    def _build_context(docs: List[Dict[str, Any]]):
        context_str = ""
        citations = []
        for d in docs:
            meta = d["metadata"]
            doc_name = meta.get("file_name", "Unknown")
            page_no = meta.get("page_number", "N/A")
            context_str += f"\n--- Source: {doc_name} (Page {page_no}) ---\n{d['text']}\n"
            citations.append({"document": doc_name, "doc_id": meta.get("doc_id"), "page": page_no})
        return context_str, citations

    @staticmethod
    def _format_history(history_messages: Optional[List[Dict[str, str]]]) -> str:
        if not history_messages:
            return ""
        lines = [f"{m['role'].capitalize()}: {m['content']}" for m in history_messages[-6:]]
        return "\n".join(lines)

    @staticmethod
    def _contextualize_query(query: str, history_messages: Optional[List[Dict[str, str]]]) -> str:
        """Lightweight reference resolution: if the query is short and contains
        a pronoun, prepend the last user question to give retrieval enough
        signal to find the right document (the LLM prompt also receives full
        history for final-answer grounding)."""
        pronouns = {"it", "its", "they", "their", "them", "this", "that"}
        if history_messages and set(query.lower().split()) & pronouns:
            last_user = next(
                (m["content"] for m in reversed(history_messages) if m["role"] == "user"), ""
            )
            if last_user:
                return f"{last_user} {query}"
        return query

    @staticmethod
    def _estimate_confidence(docs: List[Dict[str, Any]]) -> float:
        scores = [d["score"] for d in docs if d.get("score") is not None]
        if not scores:
            return 0.5
        return round(sum(scores) / len(scores), 3)
