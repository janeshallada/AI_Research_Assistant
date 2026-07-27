from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.database.base import get_db
from src.database.models import Document
from src.rag.summarizer import DocumentSummarizer
from src.rag.comparator import DocumentComparator
from src.ml.predictor import DocumentClassifier
from src.schemas import SummarizeRequest, CompareRequest, ClassifyRequest

router = APIRouter(prefix="/analysis", tags=["Analysis: Summarize / Compare / Classify"])

summarizer = DocumentSummarizer()
comparator = DocumentComparator()


@router.post("/summarize")
def summarize_document(request: SummarizeRequest, db: Session = Depends(get_db)):
    """Generates Executive Summary, Technical Summary, Bullet Point Breakdown,
    and Key Takeaways for a single document."""
    doc = db.query(Document).filter(Document.doc_id == request.doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return summarizer.summarize(doc_id=doc.doc_id, file_name=doc.file_name)


@router.post("/compare")
def compare_documents(request: CompareRequest, db: Session = Depends(get_db)):
    """Compares 2+ documents: methodologies, advantages/disadvantages,
    similarities, differences, conclusions, implementation approaches."""
    if len(request.doc_ids) < 2:
        raise HTTPException(status_code=400, detail="Provide at least two doc_ids to compare.")

    docs = db.query(Document).filter(Document.doc_id.in_(request.doc_ids)).all()
    found_ids = {d.doc_id for d in docs}
    missing = set(request.doc_ids) - found_ids
    if missing:
        raise HTTPException(status_code=404, detail=f"Documents not found: {sorted(missing)}")

    doc_id_to_name = {d.doc_id: d.file_name for d in docs}
    return comparator.compare(doc_id_to_name)


@router.post("/classify")
def classify_document(request: ClassifyRequest, db: Session = Depends(get_db)):
    """Returns the (already-computed, or freshly re-run) TensorFlow category
    prediction for a document."""
    doc = db.query(Document).filter(Document.doc_id == request.doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    classifier = DocumentClassifier.get_instance()
    if not classifier.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Classifier model is not trained yet. Run `python -m src.ml.train_classifier`.",
        )

    if doc.category is None:
        raise HTTPException(status_code=400, detail="Document has not finished processing yet.")

    return {"doc_id": doc.doc_id, "category": doc.category, "confidence": doc.category_confidence}
