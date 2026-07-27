"""
ORM schemas for document metadata, chunk records, chat sessions/messages, and
query analytics. Matches the metadata fields required by the assignment:
Document ID, Document Name, Upload Timestamp, Total Pages, Total Chunks,
Processing Status, plus Category (from the TensorFlow classifier).
"""
import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Integer, DateTime, Enum, ForeignKey, Text, Float
)
from sqlalchemy.orm import relationship

from src.database.base import Base


def gen_uuid() -> str:
    return str(uuid.uuid4())


class ProcessingStatus(str, enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"


class Document(Base):
    __tablename__ = "documents"

    doc_id = Column(String, primary_key=True, default=gen_uuid)
    file_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    total_pages = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    processing_status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING)
    category = Column(String, nullable=True)
    category_confidence = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    query_hits = relationship("QueryDocumentHit", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    """
    Mirrors what is stored in the vector DB so metadata (page numbers, chunk
    ids) can be queried relationally as well as via the vector index.
    """
    __tablename__ = "chunks"

    chunk_id = Column(String, primary_key=True)
    doc_id = Column(String, ForeignKey("documents.doc_id"), nullable=False)
    page_number = Column(Integer, nullable=False)
    char_count = Column(Integer, default=0)

    document = relationship("Document", back_populates="chunks")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    session_id = Column(String, primary_key=True, default=gen_uuid)
    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    """Stores conversation turns so follow-up references (e.g. 'its', 'their') resolve correctly."""
    __tablename__ = "chat_messages"

    message_id = Column(String, primary_key=True, default=gen_uuid)
    session_id = Column(String, ForeignKey("chat_sessions.session_id"), nullable=False)
    role = Column(String, nullable=False)  # "user" | "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")


class QueryLog(Base):
    """One row per question asked, for analytics (top-queried docs, question counts)."""
    __tablename__ = "query_logs"

    query_id = Column(String, primary_key=True, default=gen_uuid)
    session_id = Column(String, nullable=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    hits = relationship("QueryDocumentHit", back_populates="query", cascade="all, delete-orphan")


class QueryDocumentHit(Base):
    """Join table: which documents were cited/retrieved for a given query (drives 'top documents')."""
    __tablename__ = "query_document_hits"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(String, ForeignKey("query_logs.query_id"), nullable=False)
    doc_id = Column(String, ForeignKey("documents.doc_id"), nullable=False)
    page_number = Column(Integer, nullable=True)

    query = relationship("QueryLog", back_populates="hits")
    document = relationship("Document", back_populates="query_hits")
