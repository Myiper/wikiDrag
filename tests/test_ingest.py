from __future__ import annotations

import types

import ingest


def test_chunk_text_overlap_and_sizes() -> None:
    # 0..2499 => length 2500
    text = "".join(str(i % 10) for i in range(2500))
    chunks = ingest.chunk_text(text, chunk_size=1000, overlap=200)

    assert len(chunks) == 3
    assert len(chunks[0]) == 1000
    assert len(chunks[1]) == 1000
    assert len(chunks[2]) == 900
    # overlap: last 200 chars of chunk0 == first 200 chars of chunk1
    assert chunks[0][-200:] == chunks[1][:200]
    assert chunks[1][-200:] == chunks[2][:200]


def test_chunk_text_empty_returns_empty_list() -> None:
    assert ingest.chunk_text("") == []
    assert ingest.chunk_text("   \n\t  ") == []


def test_embed_text_parses_typed_response() -> None:
    class FakeOllama:
        def embeddings(self, model: str, prompt: str):  # noqa: ANN001
            return types.SimpleNamespace(embedding=[1.0, 2.0, 3.0])

    emb = ingest.embed_text(FakeOllama(), "hello", model="nomic-embed-text")
    assert emb == [1.0, 2.0, 3.0]


def test_embed_text_parses_dict_embedding() -> None:
    class FakeOllama:
        def embeddings(self, model: str, prompt: str):  # noqa: ANN001
            return {"embedding": [0.1, 0.2]}

    emb = ingest.embed_text(FakeOllama(), "hello", model="nomic-embed-text")
    assert emb == [0.1, 0.2]

