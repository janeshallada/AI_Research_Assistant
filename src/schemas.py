from datetime import datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    doc_id: str
    file_name: str
    upload_timestamp: datetime
    total_pages: int
    total_chunks: int
    processing_status: str
    category: Optional[str] = None
    category_confidence: Optional[float] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True


class UploadResponse(BaseModel):
    message: str
    document: DocumentOut


class SearchRequest(BaseModel):
    query: str
    top_k: int = 4
    doc_ids: Optional[List[str]] = None
    mode: str = Field(default="hybrid", description="semantic | keyword | hybrid")


class SearchResultItem(BaseModel):
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    score: Optional[float] = None


class QARequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    doc_ids: Optional[List[str]] = None
    mode: str = "hybrid"


class Citation(BaseModel):
    document: str
    doc_id: Optional[str] = None
    page: Any


class QAResponse(BaseModel):
    answer: str
    citations: List[Citation]
    retrieved_context: List[str]
    confidence: float
    session_id: str


class SummarizeRequest(BaseModel):
    doc_id: str


class CompareRequest(BaseModel):
    doc_ids: List[str]


class ClassifyRequest(BaseModel):
    doc_id: str
