import pytest

from src.rag.llm import MockLLMProvider


def test_mock_llm_returns_fallback_when_no_context():
    llm = MockLLMProvider()
    prompt = "Conversation History:\n(none)\n\nContext:\n\nQuestion: What is X?\n\nProvide answer"
    result = llm.complete(prompt)
    assert "cannot determine" in result.lower()


def test_mock_llm_extracts_relevant_sentence():
    llm = MockLLMProvider()
    prompt = (
        "Conversation History:\n(none)\n\n"
        "Context:\n--- Source: paper.pdf (Page 1) ---\n"
        "The chunk size used is 1000 characters. The overlap is 150 characters. "
        "This is unrelated filler text about cats.\n\n"
        "Question: What chunk size is used?\n\nProvide a clear answer"
    )
    result = llm.complete(prompt)
    assert "1000" in result


def test_qa_fallback_message_constant():
    from src.rag.qa_chain import FALLBACK_MESSAGE
    assert FALLBACK_MESSAGE == "I cannot determine the answer from the provided documents."
