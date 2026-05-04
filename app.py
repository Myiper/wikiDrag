"""
Streamlit RAG chat: retrieve context from Chroma, prompt Ollama (llama3.2), stream reply.
"""

from __future__ import annotations

from typing import Any, Iterator

import streamlit as st
from ollama import Client as OllamaClient

from retriever import query_database


def _stream_chat_delta(client: OllamaClient, *, model: str, user_content: str) -> Iterator[str]:
    stream = client.chat(
        model=model,
        messages=[{"role": "user", "content": user_content}],
        stream=True,
    )
    for chunk in stream:
        msg = getattr(chunk, "message", None)
        if msg is None:
            continue
        piece = getattr(msg, "content", None) or ""
        if piece:
            yield piece


def build_rag_prompt(user_prompt: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    context_block = "\n\n".join(
        row["document"] for row in retrieved_chunks if row.get("document")
    ).strip()

    inserted = context_block if context_block else "(No context was retrieved.)"

    return (
        "You are an AI assistant. Answer the user based ONLY on this context: "
        f"{inserted}. If the answer is not in the context, say I don't know.\n\n"
        f"{user_prompt}"
    )


def main() -> None:
    st.set_page_config(page_title="wikiDrag Chat", layout="wide")
    st.title("wikiDrag")

    if "messages" not in st.session_state:
        st.session_state.messages = list[dict[str, Any]]()

    client = OllamaClient()

    for entry in st.session_state.messages:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry["role"] == "assistant" and entry.get("chunk_metadatas") is not None:
                with st.expander("Sources"):
                    _render_sources(entry["chunk_metadatas"])

    if user_prompt := st.chat_input("Ask something about the indexed Wikipedia summaries…"):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        results = query_database(user_prompt)
        full_llm_prompt = build_rag_prompt(user_prompt, results)

        chunk_metadatas = [row.get("metadata") or {} for row in results]

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            try:
                for delta in _stream_chat_delta(client, model="llama3.2", user_content=full_llm_prompt):
                    full_response += delta
                    placeholder.markdown(full_response)
            except Exception as exc:
                placeholder.markdown(f"_Error calling Ollama: {exc}_")
                full_response = ""

            placeholder.markdown(full_response)

            with st.expander("Sources"):
                _render_sources(chunk_metadatas)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "chunk_metadatas": chunk_metadatas,
            }
        )


def _render_sources(metadatas: list[Any]) -> None:
    if not metadatas:
        st.caption("No chunks retrieved.")
        return
    for i, meta in enumerate(metadatas, start=1):
        st.markdown(f"**Chunk {i}**")
        st.json(meta)


if __name__ == "__main__":
    main()
