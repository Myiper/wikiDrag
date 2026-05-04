# Deployment recommendations (next steps)

This document outlines practical next steps to deploy this project so an instructor (or a small demo audience) can run it reliably.

## Recommended deployment paths

### Option A — Single VM (simplest)

Run **Streamlit + Ollama + ChromaDB** on the same machine.

- **Pros**: easiest to set up, minimal moving parts
- **Cons**: scaling is limited; UI and model share resources

**Best for**: class demos, short-lived deployments.

### Option B — Split services (better structure)

Split into:
- **UI service** (Streamlit)
- **Model service** (Ollama)
- **Shared persistence** (ChromaDB directory volume)

- **Pros**: easier to manage resources, can move model to a stronger machine
- **Cons**: slightly more setup

**Best for**: longer-running demos, better reliability.

### Option C — Docker Compose (recommended for reproducibility)

Package the app in containers so setup is “one command”.

- **Pros**: reproducible environment, cleaner install story
- **Cons**: Ollama GPU acceleration depends on the host and Docker setup

**Best for**: instructors, TAs, and demo environments.

## Configuration improvements (recommended)

Add environment variables to avoid hardcoding:
- **`CHROMA_DIR`**: default `./chroma_db`
- **`CHROMA_COLLECTION`**: default `wiki_rag`
- **`OLLAMA_HOST`**: default `http://localhost:11434`
- **`CHAT_MODEL`**: default `llama3.2`
- **`EMBED_MODEL`**: default `nomic-embed-text`
- **`ENTITIES_FILE`**: default `entities.json`

This makes VM/Docker deployments easier and avoids code edits.

## Reliability improvements (recommended)

### Ingestion robustness
- Keep the current retry/backoff behavior.
- Consider adding a **longer min-wait** (e.g. 1.5–2.0s) if Wikipedia throttles often.
- After ingestion, show a clear summary and keep using `missing_entities.json` to rerun missing entities quickly.

### Startup checks
Before starting the Streamlit UI:
- Check **Ollama** is reachable.
- Check required models exist (`llama3.2`, `nomic-embed-text`).
- Check Chroma collection has documents (`count > 0`).

## Deployment steps (VM approach)

1) **Install**
- Python 3.10+
- Ollama
- `pip install -r requirements.txt`

2) **Start Ollama**
- `ollama serve`
- `ollama pull llama3.2`
- `ollama pull nomic-embed-text`

3) **Ingest**
- `python ingest.py --all`
- Re-run `python ingest.py` until `missing_count=0`

4) **Run the UI**
- `streamlit run app.py --server.address 0.0.0.0 --server.port 8501`

5) **Networking**
- Open/allow inbound access to port `8501` (or put behind a reverse proxy with HTTPS).

## Security notes (if public)

- Do **not** expose Ollama publicly; keep it internal.
- Add basic access control for Streamlit (reverse proxy auth) if the UI is public.
- Consider limiting max prompt length and request rate.

## Next technical step: Docker Compose (suggested deliverable)

Create:
- `Dockerfile` for the Streamlit app
- `docker-compose.yml` with:
  - `app` service (Streamlit)
  - `ollama` service
  - a persistent volume for `chroma_db`

This provides a clean “clone → compose up” workflow.

