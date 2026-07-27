"""
AI Research & Knowledge Assistant — FastAPI application entry point.

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Swagger UI:  http://localhost:8000/docs
ReDoc:       http://localhost:8000/redoc
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from src.database.base import init_db
from routes import document_routes, search_routes, analysis_routes, analytics_routes

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Research & Knowledge Assistant",
    description=(
        "Production-oriented backend for uploading, searching, and querying "
        "large repositories of PDF documents using Retrieval-Augmented "
        "Generation, hybrid search, and a TensorFlow document classifier."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    logger.info("Initializing database...")
    init_db()
    logger.info("Startup complete. LLM provider=%s | Embedding provider=%s",
                settings.llm_provider, settings.embedding_provider)


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "AI Research & Knowledge Assistant"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


app.include_router(document_routes.router)
app.include_router(search_routes.router)
app.include_router(analysis_routes.router)
app.include_router(analytics_routes.router)
