# Product Requirement Document (PRD): Local Wikipedia RAG Assistant

## 1. Objective
To build a fully local, privacy-preserving retrieval-augmented generation (RAG) assistant that answers user queries about famous people and places using Wikipedia data. The system must run entirely on a local machine without reliance on external APIs.

## 2. Scope
The application will ingest, chunk, embed, and store Wikipedia articles for a minimum of 20 historical figures and 20 famous landmarks. It will provide a chat interface for users to ask questions, retrieving relevant local data to ground the LLM's answers and prevent hallucinations.

## 3. Technical Architecture
*   **Language:** Python 3.10+
*   **UI Framework:** Streamlit
*   **LLM Engine:** Ollama (Local)
*   **Language Model:** Llama 3.2 3B
*   **Embedding Model:** nomic-embed-text
*   **Vector Database:** ChromaDB (Local persistent storage)
*   **Data Source:** Wikipedia (via Python API)

## 4. Functional Requirements
*   **Ingestion:** The system must programmatically download Wikipedia pages for the specified entities.
*   **Processing:** Text must be cleaned and split into chunks of ~1000 characters with a 200-character overlap.
*   **Storage:** A single ChromaDB collection will be used, distinguishing entities via metadata tags (`category: person | place`, `entity: name`).
*   **Routing & Retrieval:** The system must determine the intent of the query and retrieve the top-K (e.g., top 3) most relevant chunks from the vector database.
*   **Generation:** The LLM must synthesize an answer based *only* on retrieved context. It must reply "I don't know" if the context is insufficient.
*   **Interface:** A chat interface allowing sequential Q&A, displaying the answer and citing the referenced source chunks.

## 5. Non-Functional Requirements
*   **Latency:** Retrieval should take < 1 second. Time-to-first-token (TTFT) for generation should be < 2 seconds.
*   **Portability:** The project must be easily runnable on macOS/Windows/Linux via standard `pip` requirements and Ollama.
*   **Independence:** No API keys (OpenAI, Anthropic, Pinecone) are permitted.