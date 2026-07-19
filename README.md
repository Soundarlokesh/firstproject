# Soni's Memory Assistant — Backend (Phase 2)

FastAPI + LangChain + FAISS RAG backend. Separate project from the frontend,
on purpose — deploy and version them independently.

## What's actually verified

- **Chunking → embedding → FAISS build → save → load → retrieve**: tested
  end-to-end with a mocked embedder (my sandbox's network whitelist blocks
  huggingface.co, so I couldn't download the real embedding model here).
  The pipeline logic works; the only untested part is the actual network
  call to fetch the model weights, which needs a normal internet connection
  you'll have and I didn't.
- **FastAPI endpoints, CORS, file upload, error handling**: written and
  structurally sound, not load-tested. It's a personal project backend, not
  something built to survive traffic spikes.
- **Groq generation call**: written against the real `langchain-groq` API
  surface, but never fired against a live key — I don't have one and
  wasn't going to invent test traffic against your future account.

Translation: don't assume zero bugs. Test it yourself with a real key before
you trust it for the actual event.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env — paste a real key from https://console.groq.com/keys (free, no card)

uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive API docs (FastAPI gives you
this for free — no extra work).

## How the "manual upload" you asked for works

Two dummy placeholder files are in `data/documents/` right now so the
pipeline has something to index on first run. Real usage:

1. `POST /documents/upload` — multipart form, one or more files
   (`.txt`, `.md`, `.pdf`, `.docx`). Automatically re-indexes after saving.
2. Delete the placeholder files from `data/documents/` once you upload real
   ones (or leave them — up to you, they're clearly labeled as placeholders).
3. `GET /documents` — see what's currently indexed.
4. `POST /documents/reindex` — rebuild the index without uploading anything
   new (useful after manually editing a file on disk).
5. `POST /chat` — `{"question": "..."}` → retrieves relevant chunks, asks
   Groq to answer using ONLY those chunks, refuses if the answer isn't there.

There's no upload UI yet — that's a frontend piece for later. For now, use
the `/docs` page, curl, or Postman.

## Why fastembed instead of sentence-transformers

The obvious choice (sentence-transformers) pulls in PyTorch, which pulls in
~2.5GB of CUDA dependencies you will never use for local CPU embedding on a
personal project. Switched to `fastembed` (ONNX-based) — same embedding
quality for this use case, ~200MB total instead of 4GB+. Not a compromise,
just the better choice.

## Before you deploy this anywhere public

- Set `FRONTEND_ORIGIN` in `.env` to your real deployed frontend URL, not
  `localhost`.
- The upload endpoint has no auth. Anyone who finds the URL can upload files
  and rebuild your index. Fine for a private gift project nobody will
  stumble on; not fine if you post the link publicly. Add at minimum a
  shared-secret header check before that happens.
- Groq's free tier is rate-limited (~30 req/min) — plenty for one person
  chatting with the assistant, not something to load-test.

## Structure

```
app/
  main.py     FastAPI app + endpoints
  rag.py      document loading, chunking, FAISS build/load/retrieve
  llm.py      Groq call + the system prompt that enforces "only from memories"
  config.py   settings from .env
data/documents/   drop files here (or use the upload endpoint)
vectorstore/      FAISS index lives here after the first reindex
```
