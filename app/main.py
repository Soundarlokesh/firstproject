from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import settings
from app import rag
from app.llm import answer_question

app = FastAPI(title="Soni's Memory Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serves uploaded images at GET /media/<filename> — the frontend builds full
# URLs from this (API_URL + image_url returned by /chat). Deliberately NOT
# mounted at /images — that path collides with the /images/upload and
# /images (list) API routes below, since a Mount intercepts every sub-path
# under its prefix before FastAPI's own routes get a chance (this caused a
# real 405 on /images/upload during testing — /media avoids it entirely).
app.mount("/media", StaticFiles(directory=str(settings.images_path)), name="media")


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    image_urls: list[str] = []


@app.get("/health")
def health():
    index_exists = (settings.vectorstore_path / "index.faiss").exists()
    doc_count = sum(1 for p in settings.documents_path.iterdir() if p.is_file())
    return {"status": "ok", "index_built": index_exists, "documents_on_disk": doc_count}


@app.get("/documents")
def list_documents():
    return {
        "documents": [p.name for p in settings.documents_path.iterdir() if p.is_file()]
    }


@app.post("/documents/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    """Drop real friendship documents in here from the running program —
    no code changes, no redeploy needed. Automatically reindexes afterward."""
    saved = []
    for f in files:
        suffix = Path(f.filename).suffix.lower()
        if suffix not in (".txt", ".md", ".pdf", ".docx"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {f.filename}. Use .txt, .md, .pdf, or .docx",
            )
        dest = settings.documents_path / f.filename
        contents = await f.read()
        dest.write_bytes(contents)
        saved.append(f.filename)

    try:
        chunk_count = rag.build_index()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Files saved but indexing failed: {e}")

    return {"saved": saved, "chunks_indexed": chunk_count}


@app.get("/images")
def list_images():
    return {"captions": rag.load_captions()}


@app.post("/images/upload")
async def upload_image(file: UploadFile = File(...), caption: str = Form(...)):
    """Upload one real photo with a short caption describing it (e.g. "us at
    Marina Beach, Jan 2024"). The caption is what makes it findable — without
    one, the chatbot has no way to know when this photo is relevant."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".jpg", ".jpeg", ".png", ".webp"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {file.filename}. Use .jpg, .jpeg, .png, or .webp",
        )
    if not caption.strip():
        raise HTTPException(status_code=400, detail="caption cannot be empty")

    dest = settings.images_path / file.filename
    contents = await file.read()
    dest.write_bytes(contents)
    rag.add_caption(file.filename, caption.strip())

    try:
        chunk_count = rag.build_index()
    except Exception as e:
        raise HTTPException(
            status_code=502, detail=f"Image saved but indexing failed: {e}"
        )

    return {"saved": file.filename, "caption": caption.strip(), "chunks_indexed": chunk_count}


@app.post("/documents/reindex")
def reindex():
    """Manually rebuild the index without uploading anything new — useful if
    you edit a document in place on disk."""
    try:
        chunk_count = rag.build_index()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Indexing failed: {e}")
    return {"chunks_indexed": chunk_count}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    try:
        chunks = rag.retrieve(req.question, k=4)
        answer = answer_question(req.question, chunks)
    except RuntimeError as e:
        # e.g. missing/placeholder GROQ_API_KEY — a config problem, not a server crash
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        # Anything else (Groq API error, embedding failure, etc.) — surface it
        # instead of letting it become an opaque 500 with no CORS headers.
        raise HTTPException(status_code=502, detail=f"Chat failed: {e}")

    sources = sorted({c["source"] for c in chunks})
    image_urls = [f"/media/{c['image_file']}" for c in chunks if c.get("image_file")]
    return ChatResponse(answer=answer, sources=sources, image_urls=image_urls)