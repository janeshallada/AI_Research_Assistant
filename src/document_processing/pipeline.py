"""
Orchestrates the full ingestion pipeline for one document:
  parse (page text) -> classify (TensorFlow) -> chunk -> embed & index -> persist metadata

Designed to run as a FastAPI BackgroundTask so uploads don't block the event
loop while parsing/embedding a large PDF.
"""
import logging

from config.settings import settings
from src.database.base import db_session
from src.database.models import Document, Chunk, ProcessingStatus
from src.document_processing.pdf_parser import PDFParser, PDFParsingError
from src.document_processing.chunker import DocumentChunker
from src.ml.predictor import DocumentClassifier
from src.vector_store.manager import get_vector_store

logger = logging.getLogger(__name__)


def process_pdf_pipeline(doc_id: str, file_path: str, file_name: str) -> None:
    parser = PDFParser()
    chunker = DocumentChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    vector_store = get_vector_store()
    classifier = DocumentClassifier.get_instance()

    with db_session() as db:
        doc = db.query(Document).filter(Document.doc_id == doc_id).first()
        if doc is None:
            logger.error("Document %s not found when starting pipeline.", doc_id)
            return
        doc.processing_status = ProcessingStatus.PROCESSING
        db.add(doc)

    try:
        parsed = parser.extract_text_with_metadata(file_path, doc_id)
        full_text = " ".join(p["text"] for p in parsed["pages"])[:20000]

        category, confidence = classifier.predict(full_text) if full_text else (None, None)

        chunks = chunker.create_chunks(parsed["pages"])
        vector_store.index_chunks(chunks, file_name=file_name)

        with db_session() as db:
            doc = db.query(Document).filter(Document.doc_id == doc_id).first()
            doc.total_pages = parsed["total_pages"]
            doc.total_chunks = len(chunks)
            doc.category = category
            doc.category_confidence = confidence
            doc.processing_status = ProcessingStatus.PROCESSED
            db.add(doc)
            for c in chunks:
                db.add(Chunk(
                    chunk_id=c["chunk_id"], doc_id=c["doc_id"],
                    page_number=c["page_number"], char_count=c["char_count"],
                ))
        logger.info("Successfully processed document %s (%d chunks).", doc_id, len(chunks))

    except PDFParsingError as exc:
        _mark_failed(doc_id, str(exc))
    except Exception as exc:  # noqa: BLE001 - top-level pipeline guard
        logger.exception("Unexpected error processing document %s", doc_id)
        _mark_failed(doc_id, str(exc))


def _mark_failed(doc_id: str, message: str) -> None:
    with db_session() as db:
        doc = db.query(Document).filter(Document.doc_id == doc_id).first()
        if doc:
            doc.processing_status = ProcessingStatus.FAILED
            doc.error_message = message
            db.add(doc)


def reprocess_document(doc_id: str) -> None:
    with db_session() as db:
        doc = db.query(Document).filter(Document.doc_id == doc_id).first()
        if not doc:
            raise ValueError("Document not found")
        file_path, file_name = doc.file_path, doc.file_name

    get_vector_store().delete_document(doc_id)
    with db_session() as db:
        db.query(Chunk).filter(Chunk.doc_id == doc_id).delete()

    process_pdf_pipeline(doc_id, file_path, file_name)
