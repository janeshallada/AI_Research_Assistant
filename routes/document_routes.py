import os
import uuid
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from config.settings import settings
from src.database.base import get_db
from src.database.models import Document, ProcessingStatus
from src.document_processing.pipeline import process_pdf_pipeline, reprocess_document
from src.vector_store.manager import get_vector_store
from src.schemas import DocumentOut, UploadResponse

router = APIRouter(prefix="/documents", tags=["Document Management"])
logger = logging.getLogger(__name__)

os.makedirs(settings.raw_documents_dir, exist_ok=True)


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Uploads a PDF document and triggers background processing
    (parsing, classification, chunking, vector indexing)."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = str(uuid.uuid4())
    safe_name = file.filename.replace("/", "_")
    file_path = os.path.join(settings.raw_documents_dir, f"{doc_id}_{safe_name}")

    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    doc = Document(
        doc_id=doc_id,
        file_name=file.filename,
        file_path=file_path,
        processing_status=ProcessingStatus.PENDING,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    background_tasks.add_task(process_pdf_pipeline, doc_id, file_path, file.filename)

    return UploadResponse(
        message="Document uploaded successfully. Processing started in the background.",
        document=DocumentOut.model_validate(doc),
    )


@router.get("", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)):
    """Lists all uploaded documents and their processing status."""
    return db.query(Document).order_by(Document.upload_timestamp.desc()).all()


@router.get("/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.doc_id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


@router.delete("/{doc_id}")
def delete_document(doc_id: str, db: Session = Depends(get_db)):
    """Deletes a document's metadata, file, and vector-store entries."""
    doc = db.query(Document).filter(Document.doc_id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    get_vector_store().delete_document(doc_id)
    if os.path.exists(doc.file_path):
        os.remove(doc.file_path)

    db.delete(doc)
    db.commit()
    return {"message": f"Document {doc_id} deleted successfully."}


@router.post("/{doc_id}/reprocess")
def reprocess(doc_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Re-runs the ingestion pipeline for a document (e.g. after a chunking
    strategy or model update)."""
    doc = db.query(Document).filter(Document.doc_id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    doc.processing_status = ProcessingStatus.PENDING
    db.commit()
    background_tasks.add_task(reprocess_document, doc_id)
    return {"message": f"Reprocessing started for document {doc_id}."}
