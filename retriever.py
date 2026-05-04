from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

import chromadb
from ollama import Client as OllamaClient


def get_chroma_collection(
    *,
    collection_name: str = "wiki_rag",
    db_dir: Optional[Path] = None,
) -> Any:
    """
    Return the persistent Chroma collection used by `ingest.py`.

    By default, this points at a `chroma_db/` folder next to this file.
    """
    if db_dir is None:
        db_dir = Path(__file__).with_name("chroma_db")

    client = chromadb.PersistentClient(path=str(db_dir))
    return client.get_or_create_collection(name=collection_name)


def embed_text(
    ollama: OllamaClient,
    text: str,
    *,
    model: str = "nomic-embed-text",
) -> list[float]:
    """
    Return an embedding vector using the official Ollama Python client.

    Supports both response shapes seen across Ollama client versions:
    - {"embedding": [...]}  (older)
    - {"embeddings": [[...], ...]} (newer / batch-style)
    """
    resp: Any = ollama.embeddings(model=model, prompt=text)

    # Newer official client returns typed responses with attributes.
    embedding_attr = getattr(resp, "embedding", None)
    if isinstance(embedding_attr, list) and embedding_attr:
        return embedding_attr

    # Back-compat: older clients / wrappers return dicts.
    if isinstance(resp, dict):
        if "embedding" in resp:
            return resp["embedding"]
        if "embeddings" in resp and resp["embeddings"]:
            return resp["embeddings"][0]

    raise RuntimeError(f"Unexpected ollama embeddings response shape: {type(resp)!r}")


def _unique_entities_from_collection(collection: Any) -> list[str]:
    """
    Extract unique entity names from collection metadata.

    This is fast for small/medium collections and lets us do simple entity routing
    without hardcoding the PEOPLE/PLACES lists here.
    """
    try:
        res: dict[str, Any] = collection.get(include=["metadatas"])
    except Exception:
        return []

    metadatas = res.get("metadatas") or []
    out: list[str] = []
    seen: set[str] = set()
    for meta in metadatas:
        if not isinstance(meta, dict):
            continue
        ent = meta.get("entity")
        if isinstance(ent, str):
            key = ent.casefold()
            if key not in seen:
                seen.add(key)
                out.append(ent)
    return out


def _best_entity_match(user_prompt: str, entities: list[str], *, min_score: float = 0.78) -> Optional[str]:
    """
    Try to find an entity mentioned in the prompt, tolerant of small typos.

    Strategy:
    - exact substring match (case-insensitive)
    - otherwise, fuzzy-match entity names against short n-grams of prompt tokens
    """
    prompt_cf = user_prompt.casefold()

    # 1) direct substring match
    for ent in entities:
        if ent and ent.casefold() in prompt_cf:
            return ent

    # 2) fuzzy match against n-grams
    tokens = re.findall(r"[a-zA-Z]+", user_prompt.casefold())
    if not tokens:
        return None

    grams: list[str] = []
    for n in (1, 2, 3, 4):
        for i in range(0, len(tokens) - n + 1):
            grams.append(" ".join(tokens[i : i + n]))

    best_ent: Optional[str] = None
    best_score = 0.0
    for ent in entities:
        ent_cf = ent.casefold()
        for g in grams:
            score = SequenceMatcher(None, ent_cf, g).ratio()
            if score > best_score:
                best_score = score
                best_ent = ent

    if best_ent is not None and best_score >= min_score:
        return best_ent
    return None


def _format_query_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    out: list[dict[str, Any]] = []
    for i in range(min(len(documents), len(metadatas))):
        out.append(
            {
                "document": documents[i],
                "metadata": metadatas[i],
                "distance": distances[i] if i < len(distances) else None,
            }
        )
    return out


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z]+", text.casefold()))


def _rerank_candidates(user_prompt: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Re-rank candidates by combining vector distance with lightweight keyword overlap.

    This helps when the vector search returns the wrong entity in small datasets.
    """
    q_tokens = _token_set(user_prompt)
    if not q_tokens:
        return candidates

    def score(row: dict[str, Any]) -> float:
        doc = row.get("document") or ""
        d_tokens = _token_set(doc)
        inter = len(q_tokens & d_tokens)
        union = len(q_tokens | d_tokens) or 1
        jaccard = inter / union

        dist = row.get("distance")
        dist_val = float(dist) if isinstance(dist, (int, float)) else 1.0

        # Higher is better: prefer low distance and high token overlap.
        return (0.65 * (1.0 - dist_val)) + (0.35 * jaccard)

    return sorted(candidates, key=score, reverse=True)


def query_database(user_prompt: str, *, top_k: int = 3) -> list[dict[str, Any]]:
    """
    Embed `user_prompt`, query ChromaDB, and return the top matches.

    Returns a list of dicts with:
    - document: str
    - metadata: dict
    - distance: float | None
    """
    if not user_prompt or not user_prompt.strip():
        return []

    ollama = OllamaClient()
    query_emb = embed_text(ollama, user_prompt, model="nomic-embed-text")

    collection = get_chroma_collection()
    entities = _unique_entities_from_collection(collection)
    matched_entity = _best_entity_match(user_prompt, entities)

    # Pull more than `top_k` then re-rank for robustness.
    candidate_k = max(top_k * 6, 18)

    # Try a metadata-filtered query first if we can confidently identify an entity.
    if matched_entity:
        try:
            filtered: dict[str, Any] = collection.query(
                query_embeddings=[query_emb],
                n_results=candidate_k,
                where={"entity": matched_entity},
                include=["documents", "metadatas", "distances"],
            )
            out = _format_query_result(filtered)
            if out:
                return _rerank_candidates(user_prompt, out)[:top_k]
        except Exception:
            pass

    # Fallback: normal similarity search across the whole collection.
    result: dict[str, Any] = collection.query(
        query_embeddings=[query_emb],
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"],
    )
    return _rerank_candidates(user_prompt, _format_query_result(result))[:top_k]


__all__ = [
    "query_database",
    "get_chroma_collection",
    "embed_text",
]

