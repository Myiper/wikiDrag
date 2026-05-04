"""
Download Wikipedia summaries for famous people and places, chunk them, and report counts.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import chromadb
import chromadb.api.models.Collection
import requests
import wikipedia
import wikipedia.wikipedia as _wikipedia_internal
from ollama import Client as OllamaClient
from tqdm import tqdm
from wikipedia.exceptions import WikipediaException


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Split text into fixed-size chunks with a sliding window overlap.

    Each chunk is at most `chunk_size` characters. The next chunk starts
    `chunk_size - overlap` characters after the previous start so that the
    tail of one chunk repeats at the head of the next.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    stripped = text.strip()
    if not stripped:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    n = len(stripped)

    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(stripped[start:end])
        if end == n:
            break
        start += step

    return chunks


def fetch_summary(title: str) -> str:
    """Resolve disambiguation when possible and return full summary text."""
    try:
        return wikipedia.summary(title, sentences=0, auto_suggest=False)
    except wikipedia.exceptions.DisambiguationError as e:
        if not e.options:
            raise
        return wikipedia.summary(e.options[0], sentences=0, auto_suggest=False)
    except wikipedia.exceptions.PageError:
        return wikipedia.summary(title, sentences=0, auto_suggest=True)


def fetch_summary_with_retries(
    title: str,
    *,
    max_attempts: int = 6,
    base_sleep_s: float = 1.0,
) -> str:
    """
    Fetch a Wikipedia summary with retries + exponential backoff.

    Wikimedia often throttles or transiently fails; retries make ingestion converge
    to "all entities ingested" after one or two runs.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fetch_summary(title)
        except (WikipediaException, requests.exceptions.RequestException, ValueError) as exc:
            # ValueError covers occasional decoding / parsing issues from dependencies.
            last_exc = exc
            if attempt >= max_attempts:
                break
            sleep_s = base_sleep_s * (2 ** (attempt - 1))
            sleep_s *= random.uniform(0.85, 1.15)
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


EntityType = Literal["person", "place"]


def load_entities_from_json(path: Path) -> list[tuple[str, EntityType]]:
    """
    Load entities from a JSON file.

    Supported formats:
    1) {"people": [...], "places": [...]}  (recommended)
    2) [{"entity": "...", "entity_type": "person"|"place"}, ...]
    """
    data = json.loads(path.read_text(encoding="utf-8"))

    out: list[tuple[str, EntityType]] = []

    if isinstance(data, dict):
        people = data.get("people", [])
        places = data.get("places", [])
        if isinstance(people, list):
            for p in people:
                if isinstance(p, str) and p.strip():
                    out.append((p.strip(), "person"))
        if isinstance(places, list):
            for p in places:
                if isinstance(p, str) and p.strip():
                    out.append((p.strip(), "place"))
    elif isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            ent = row.get("entity")
            typ = row.get("entity_type")
            if isinstance(ent, str) and ent.strip() and typ in ("person", "place"):
                out.append((ent.strip(), typ))

    # de-duplicate while preserving order
    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[str, EntityType]] = []
    for ent, typ in out:
        key = (ent.casefold(), typ)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((ent, typ))

    if not deduped:
        raise ValueError(
            f"No entities found in {path.name}. Expected keys people/places or a list of objects."
        )
    return deduped


def get_wiki_entities(entities_path: Path) -> list[tuple[str, EntityType]]:
    return load_entities_from_json(entities_path)


def _missing_file_path() -> Path:
    return Path(__file__).with_name("missing_entities.json")


def load_missing_entities() -> list[dict[str, Any]]:
    """
    Load missing entities from the last ingest run (if present).

    Shape: [{"entity": str, "entity_type": "person"|"place", "error": str}, ...]
    """
    p = _missing_file_path()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out: list[dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        ent = row.get("entity")
        typ = row.get("entity_type")
        err = row.get("error", "")
        if isinstance(ent, str) and typ in ("person", "place"):
            out.append({"entity": ent, "entity_type": typ, "error": str(err)})
    return out


def save_missing_entities(rows: list[dict[str, Any]]) -> None:
    _missing_file_path().write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def select_entities_to_process(*, process_all: bool, entities_path: Path) -> list[tuple[str, EntityType]]:
    all_entities = get_wiki_entities(entities_path)
    if process_all:
        return all_entities

    missing = load_missing_entities()
    if not missing:
        # First run (or file deleted): process everything.
        return all_entities

    missing_set = {(m["entity"], m["entity_type"]) for m in missing}
    return [(name, typ) for (name, typ) in all_entities if (name, typ) in missing_set]


def embed_text(ollama: OllamaClient, text: str, model: str = "nomic-embed-text") -> list[float]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entities",
        default="entities.json",
        help="Path to JSON file containing people/places to ingest (default: entities.json).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all entities (ignore missing_entities.json).",
    )
    args = parser.parse_args()
    entities_path = Path(args.entities)
    if not entities_path.is_absolute():
        entities_path = Path(__file__).with_name(args.entities)
    if not entities_path.exists():
        raise FileNotFoundError(f"Entities file not found: {entities_path}")

    wikipedia.set_lang("en")
    # The bundled client defaults to http://; HTTPS and a descriptive User-Agent
    # align with Wikimedia API expectations and avoid empty/non-JSON responses.
    _wikipedia_internal.API_URL = _wikipedia_internal.API_URL.replace(
        "http://", "https://"
    )
    wikipedia.set_user_agent(
        "wikiDrag-ingest/1.0 (university project; Python/requests) wikiDrag"
    )
    # Be conservative: helps avoid 429/rate-limit blocks, especially on shared networks.
    wikipedia.set_rate_limiting(True, min_wait=timedelta(milliseconds=900))

    # 1) Fetch + chunk while preserving metadata per chunk
    chunk_rows: list[dict[str, Any]] = []
    ingested_entities: set[str] = set()
    skipped_entities: list[tuple[str, str, str]] = []

    to_process = select_entities_to_process(process_all=bool(args.all), entities_path=entities_path)

    for entity_name, entity_type in tqdm(to_process, desc="Fetching Wikipedia", unit="entity"):
        try:
            summary = fetch_summary_with_retries(entity_name)
        except Exception as exc:
            skipped_entities.append((entity_name, entity_type, str(exc)))
            continue

        chunks = chunk_text(summary)
        for i, chunk in enumerate(chunks):
            chunk_rows.append(
                {
                    "id": f"{entity_type}:{entity_name}:{i}",
                    "document": chunk,
                    "metadata": {
                        "entity": entity_name,
                        "entity_type": entity_type,
                        "chunk_index": i,
                    },
                }
            )
        ingested_entities.add(entity_name)

    # 2) Initialize persistent local ChromaDB + collection
    db_path = Path(__file__).with_name("chroma_db")
    client = chromadb.PersistentClient(path=str(db_path))
    collection: chromadb.api.models.Collection.Collection = client.get_or_create_collection(
        name="wiki_rag"
    )

    # 3) Embed with Ollama and upsert into Chroma with a progress bar
    ollama = OllamaClient()
    batch_ids: list[str] = []
    batch_docs: list[str] = []
    batch_embs: list[list[float]] = []
    batch_metas: list[dict[str, Any]] = []

    batch_size = 32
    embed_failures = 0
    for row in tqdm(chunk_rows, desc="Embedding + upserting", unit="chunk"):
        try:
            emb = embed_text(ollama, row["document"], model="nomic-embed-text")
        except Exception:
            embed_failures += 1
            continue

        batch_ids.append(row["id"])
        batch_docs.append(row["document"])
        batch_embs.append(emb)
        batch_metas.append(row["metadata"])

        if len(batch_ids) >= batch_size:
            collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=batch_embs,
                metadatas=batch_metas,
            )
            batch_ids, batch_docs, batch_embs, batch_metas = [], [], [], []

    if batch_ids:
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_embs,
            metadatas=batch_metas,
        )

    # Summary: total chunks built (from successful fetches), plus what's stored.
    print(
        f"entities_ok={len(ingested_entities)}/{len(to_process)} "
        f"chunks_built={len(chunk_rows)} embed_failures={embed_failures} "
        f"chroma_count={collection.count()}"
    )

    missing_rows = [
        {"entity": name, "entity_type": typ, "error": err}
        for (name, typ, err) in skipped_entities
    ]
    save_missing_entities(missing_rows)

    if missing_rows:
        examples = "; ".join(
            f"{r['entity']} ({r['entity_type']}): {str(r['error'])[:60]}"
            for r in missing_rows[:5]
        )
        print(
            f"missing_saved={_missing_file_path().name} "
            f"missing_count={len(missing_rows)} examples={examples}"
        )
    else:
        print(f"missing_saved={_missing_file_path().name} missing_count=0")


if __name__ == "__main__":
    main()
