# wikiDrag

Local Wikipedia **RAG** chatbot using **ChromaDB** + **Ollama** + **Streamlit**.

This project:
- Downloads Wikipedia summaries for a fixed list of famous **people** and **places**
- Splits them into overlapping chunks
- Embeds them with Ollama (`nomic-embed-text`)
- Stores them in a persistent ChromaDB collection (`wiki_rag`)
- Runs a Streamlit chat UI that retrieves relevant chunks and asks a local LLM (`llama3.2`) to answer **only from retrieved context**

## Requirements

- **Python 3.10+** (tested with Python 3.12 on Windows)
- **Ollama** installed and running (local model + embeddings)

## 1) Install dependencies

From the project root:

```bash
python -m venv venv
```

Activate the virtual environment:

```bash
venv\Scripts\activate
```

Install Python packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Run the local model (Ollama)

Install Ollama from the official site:
- [Ollama download](https://ollama.com/download)

Start Ollama (one-time per machine boot / session):

```bash
ollama serve
```

Pull the required models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Quick sanity check (optional):

```bash
ollama run llama3.2 "Say hello"
```

## 3) Ingest data (build the vector database)

This creates a local persistent ChromaDB folder `chroma_db/` and a collection named **`wiki_rag`**.

Run ingestion:

```bash
python ingest.py
```

### About missing entities (important)

Wikipedia can sometimes throttle requests. `ingest.py` will:
- Retry requests with exponential backoff
- Save any entities that still failed into `missing_entities.json`

If `missing_entities.json` exists, running:

```bash
python ingest.py
```

will ingest **only the missing ones** (faster).

To force ingestion of **all** entities (ignores `missing_entities.json`):

```bash
python ingest.py --all
```

After ingestion, you’ll see a summary like:
- `entities_ok=.../40`
- `chunks_built=...`
- `chroma_count=...`
- `missing_count=...`

## 4) Start the application (Streamlit)

Run:

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints in your terminal (usually `http://localhost:8501`).

## 5) Example queries

Try questions about entities that were ingested successfully:

- `Tell me about Albert Einstein.`
- `What is the Taj Mahal?`
- `Where is Machu Picchu located?`
- `What is the Eiffel Tower?`
- `Who was Nelson Mandela?`

If the relevant answer is not present in the retrieved chunks, the assistant will respond:
- `I don't know.`

## (Optional) Run tests

```bash
pytest -q
```

## Project files

- `ingest.py`: fetches Wikipedia summaries, chunks them, embeds with Ollama, upserts into ChromaDB
- `retriever.py`: embeds user query and retrieves the top chunks from ChromaDB
- `app.py`: Streamlit chat UI with streaming responses from Ollama (`llama3.2`)
- `chroma_db/`: local persistent database (ignored by git)
- `missing_entities.json`: entities that failed to ingest in the last run (used to retry)
