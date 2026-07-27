import uuid
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.database.base import get_db
from src.database.models import ChatSession, ChatMessage, QueryLog, QueryDocumentHit
from src.rag.qa_chain import RAGQuestionAnswering
from src.vector_store.manager import get_vector_store
from src.schemas import SearchRequest, SearchResultItem, QARequest, QAResponse

router = APIRouter(tags=["Search & Question Answering"])
logger = logging.getLogger(__name__)

qa_engine = RAGQuestionAnswering()


@router.post("/search", response_model=list[SearchResultItem])
def search(request: SearchRequest):
    """Semantic / keyword / hybrid search across one or more uploaded documents."""
    store = get_vector_store()
    if request.mode == "semantic":
        results = store.semantic_search(request.query, top_k=request.top_k, doc_ids=request.doc_ids)
    elif request.mode == "keyword":
        results = store.keyword_search(request.query, top_k=request.top_k, doc_ids=request.doc_ids)
    elif request.mode == "hybrid":
        results = store.hybrid_search(request.query, top_k=request.top_k, doc_ids=request.doc_ids)
    else:
        raise HTTPException(status_code=400, detail="mode must be one of: semantic, keyword, hybrid")
    return results


@router.post("/qa", response_model=QAResponse)
def ask_question(request: QARequest, db: Session = Depends(get_db)):
    """Retrieval-Augmented Generation question answering with citations and
    session-based conversation memory."""
    session_id = request.session_id
    if session_id:
        session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found.")
    else:
        session = ChatSession(session_id=str(uuid.uuid4()))
        db.add(session)
        db.commit()
        db.refresh(session)
        session_id = session.session_id

    history = [
        {"role": m.role, "content": m.content}
        for m in db.query(ChatMessage).filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at).all()
    ]

    result = qa_engine.answer_question(
        query=request.question,
        doc_ids=request.doc_ids,
        history_messages=history,
        search_mode=request.mode,
    )

    db.add(ChatMessage(session_id=session_id, role="user", content=request.question))
    db.add(ChatMessage(session_id=session_id, role="assistant", content=result["answer"]))

    query_log = QueryLog(session_id=session_id, question=request.question, answer=result["answer"])
    db.add(query_log)
    db.flush()
    for c in result["citations"]:
        if c.get("doc_id"):
            db.add(QueryDocumentHit(query_id=query_log.query_id, doc_id=c["doc_id"], page_number=c.get("page")))
    db.commit()

    return QAResponse(
        answer=result["answer"],
        citations=result["citations"],
        retrieved_context=result["retrieved_context"],
        confidence=result["confidence"],
        session_id=session_id,
    )
