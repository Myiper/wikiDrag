from __future__ import annotations

from typing import Any

import retriever


def test_query_database_empty_prompt_returns_empty() -> None:
    assert retriever.query_database("") == []
    assert retriever.query_database("   ") == []


def test_embed_text_parses_typed_response() -> None:
    class FakeOllama:
        def embeddings(self, model: str, prompt: str):  # noqa: ANN001
            class R:
                embedding = [0.0, 1.0]

            return R()

    assert retriever.embed_text(FakeOllama(), "x") == [0.0, 1.0]


def test_query_database_shapes(monkeypatch: Any) -> None:
    # Stub embedder
    monkeypatch.setattr(retriever, "embed_text", lambda *args, **kwargs: [0.0, 0.0])

    class FakeCollection:
        def get(self, **kwargs):  # noqa: ANN001
            return {"metadatas": [{"entity": "Albert Einstein"}]}

        def query(self, **kwargs):  # noqa: ANN001
            # If entity filtering is used, ensure it is passed correctly.
            if "where" in kwargs:
                assert kwargs["where"] == {"entity": "Albert Einstein"}
            return {
                "documents": [["doc1", "doc2", "doc3"]],
                "metadatas": [[{"a": 1}, {"b": 2}, {"c": 3}]],
                "distances": [[0.1, 0.2, 0.3]],
            }

    monkeypatch.setattr(retriever, "get_chroma_collection", lambda **kwargs: FakeCollection())

    out = retriever.query_database("hi", top_k=3)
    assert len(out) == 3
    assert out[0]["document"] == "doc1"
    assert out[0]["metadata"] == {"a": 1}
    assert out[0]["distance"] == 0.1

