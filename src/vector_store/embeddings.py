"""
Embedding provider abstraction. Supports:
  - "local": sentence-transformers/all-MiniLM-L6-v2 (no external API calls)
  - "openai": OpenAI embeddings API

Selected via EMBEDDING_PROVIDER in .env.
"""
import logging
from typing import List

from config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError


class LocalEmbeddingProvider(EmbeddingProvider):
    """Wraps sentence-transformers so embeddings can be generated fully offline."""

    def __init__(self, model_name: str = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name or settings.local_embedding_model)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], show_progress_bar=False)[0].tolist()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_embedding_model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def get_embedding_provider() -> EmbeddingProvider:
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingProvider()
    return LocalEmbeddingProvider()
