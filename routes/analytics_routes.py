from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.database.base import get_db
from src.analytics.metrics import get_system_analytics, get_top_queried_documents

router = APIRouter(prefix="/analytics", tags=["System Analytics"])


@router.get("/summary")
def analytics_summary(db: Session = Depends(get_db)):
    """Total documents, total chunks, total embeddings, category distribution,
    and total questions answered."""
    return get_system_analytics(db)


@router.get("/top-documents")
def top_documents(limit: int = 10, db: Session = Depends(get_db)):
    """Most frequently retrieved/queried documents."""
    return get_top_queried_documents(db, limit=limit)
