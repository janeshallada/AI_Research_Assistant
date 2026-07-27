"""
Centralized, environment-based configuration for the AI Research & Knowledge
Assistant. All tunables (paths, model names, chunking params, retrieval
params) are read from environment variables / .env so the app can move
between dev, test, and production environments without code changes.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"
    llm_provider: str = "mock"  # "openai" | "mock"

    # Embeddings
    embedding_provider: str = "local"  # "local" | "openai"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Storage
    raw_documents_dir: str = "./data/raw_documents"
    vector_db_dir: str = "./data/vector_db"
    sqlite_db_path: str = "./data/app.db"
    model_path: str = "./models/tf_classifier.keras"
    tokenizer_path: str = "./models/tokenizer.pickle"
    labels_path: str = "./models/labels.json"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Retrieval
    retrieval_top_k: int = 4

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
