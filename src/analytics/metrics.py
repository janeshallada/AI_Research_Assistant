"""
System Analytics
-----------------
Computes usage statistics: total documents, total processed chunks, total
embeddings generated, category distribution, top-queried documents, and
total questions answered.
"""
from typing import Dict, Any, List
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.database.models import Document, QueryLog, QueryDocumentHit, ProcessingStatus
from src.vector_store.manager import get_vector_store


def get_system_analytics(db: Session) -> Dict[str, Any]:
    total_documents = db.query(Document).count()
    processed_documents = db.query(Document).filter(
        Document.processing_status == ProcessingStatus.PROCESSED
    ).count()
    total_chunks = db.query(func.sum(Document.total_chunks)).scalar() or 0
    total_questions = db.query(QueryLog).count()

    category_counts = (
        db.query(Document.category, func.count(Document.doc_id))
        .filter(Document.category.isnot(None))
        .group_by(Document.category)
        .all()
    )

    total_embeddings = get_vector_store().total_chunks()

    return {
        "total_documents": total_documents,
        "processed_documents": processed_documents,
        "total_chunks": int(total_chunks),
        "total_embeddings_generated": total_embeddings,
        "total_questions_answered": total_questions,
        "category_distribution": {cat: count for cat, count in category_counts},
    }


def get_top_queried_documents(db: Session, limit: int = 10) -> List[Dict[str, Any]]:
    rows = (
        db.query(
            QueryDocumentHit.doc_id,
            Document.file_name,
            func.count(QueryDocumentHit.id).label("hit_count"),
        )
        .join(Document, Document.doc_id == QueryDocumentHit.doc_id)
        .group_by(QueryDocumentHit.doc_id, Document.file_name)
        .order_by(func.count(QueryDocumentHit.id).desc())
        .limit(limit)
        .all()
    )
    return [{"doc_id": doc_id, "file_name": file_name, "times_referenced": hit_count} for doc_id, file_name, hit_count in rows]
