"""
LLM provider abstraction.

- "openai": calls OpenAI chat completions (requires OPENAI_API_KEY).
- "mock":   a deterministic, fully offline extractive responder used for
            local development, automated tests, and demos without incurring
            API costs or requiring network access. It still respects the
            "answer only from context" contract so downstream RAG behaviour
            (citations, fallback messaging) can be exercised end-to-end.

Selected via LLM_PROVIDER in .env.
"""
import logging
from config.settings import settings

logger = logging.getLogger(__name__)


class LLMProvider:
    def complete(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAILLMProvider(LLMProvider):
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_chat_model

    def complete(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


class MockLLMProvider(LLMProvider):
    """Extractive fallback: returns the most relevant sentences from the
    supplied context rather than calling an external model."""

    def complete(self, prompt: str) -> str:
        context = self._extract_section(prompt, "Context:", "Question:")
        question = self._extract_section(prompt, "Question:", "Provide")
        if not context.strip():
            return "I cannot determine the answer from the provided documents."

        sentences = [s.strip() for s in context.replace("\n", " ").split(".") if s.strip()]
        q_terms = set(w.lower() for w in question.split() if len(w) > 3)
        scored = sorted(
            sentences,
            key=lambda s: sum(1 for t in q_terms if t in s.lower()),
            reverse=True,
        )
        best = [s for s in scored[:3] if s]
        if not best:
            return "I cannot determine the answer from the provided documents."
        return "[MOCK-LLM extractive answer] " + ". ".join(best) + "."

    @staticmethod
    def _extract_section(text: str, start_marker: str, end_marker: str) -> str:
        try:
            start = text.index(start_marker) + len(start_marker)
            end = text.index(end_marker, start)
            return text[start:end].strip()
        except ValueError:
            return ""


def get_llm() -> LLMProvider:
    if settings.llm_provider == "openai" and settings.openai_api_key:
        return OpenAILLMProvider()
    if settings.llm_provider == "openai" and not settings.openai_api_key:
        logger.warning("LLM_PROVIDER=openai but no OPENAI_API_KEY set — falling back to mock mode.")
    return MockLLMProvider()
