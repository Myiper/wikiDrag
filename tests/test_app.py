from __future__ import annotations

import app


def test_build_rag_prompt_includes_context_and_user_prompt() -> None:
    prompt = app.build_rag_prompt(
        "What is X?",
        retrieved_chunks=[
            {"document": "CTX1", "metadata": {"entity": "A"}},
            {"document": "CTX2", "metadata": {"entity": "B"}},
        ],
    )
    assert "Answer the user based ONLY on this context:" in prompt
    assert "CTX1" in prompt and "CTX2" in prompt
    assert prompt.endswith("What is X?")


def test_build_rag_prompt_no_context() -> None:
    prompt = app.build_rag_prompt("Hello", retrieved_chunks=[])
    assert "(No context was retrieved.)" in prompt
    assert prompt.endswith("Hello")

