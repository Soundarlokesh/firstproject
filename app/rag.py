"""
RAG core: turn whatever's in data/documents/ (and image captions from
data/images/captions.json) into a FAISS index, and answer retrieval queries
against it. Deliberately dumb and readable — this is a personal project's
backend, not a platform. Swap pieces later if you outgrow it (e.g. FAISS ->
Chroma is a ~10 line change, the interface below doesn't change).
"""

import json
from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

_LOADERS = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
}

_embeddings = None


def get_embeddings():
    """Lazily construct the embedding model — it's the slow part to init,
    so we only pay for it once per process, not once per request."""
    global _embeddings
    if _embeddings is None:
        _embeddings = FastEmbedEmbeddings(model_name=settings.embedding_model)
    return _embeddings


def load_documents() -> list:
    """Load every supported file in DOCUMENTS_DIR. Unsupported extensions are
    skipped, not errored on — someone will eventually drop a .jpg in there."""
    docs = []
    for path in settings.documents_path.iterdir():
        if not path.is_file():
            continue
        loader_cls = _LOADERS.get(path.suffix.lower())
        if loader_cls is None:
            continue
        try:
            loader = loader_cls(str(path))
            file_docs = loader.load()
            for d in file_docs:
                d.metadata["source"] = path.name
            docs.extend(file_docs)
        except Exception as exc:
            # A single bad file should not take down indexing for everything else.
            print(f"[rag] skipped {path.name}: {exc}")
    return docs


def _captions_file() -> Path:
    return settings.images_path / "captions.json"


def load_captions() -> list[dict]:
    """Each entry: {"filename": "...", "caption": "..."}."""
    f = _captions_file()
    if not f.exists():
        return []
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[rag] couldn't read captions.json: {exc}")
        return []


def add_caption(filename: str, caption: str) -> None:
    entries = load_captions()
    entries.append({"filename": filename, "caption": caption})
    _captions_file().write_text(json.dumps(entries, indent=2), encoding="utf-8")


def load_image_documents() -> list[Document]:
    """Turn each caption entry into a tiny retrievable Document, tagged so
    /chat can tell it apart from ordinary text and attach the real image."""
    docs = []
    for entry in load_captions():
        docs.append(
            Document(
                page_content=entry["caption"],
                metadata={
                    "source": entry["filename"],
                    "type": "image",
                    "image_file": entry["filename"],
                },
            )
        )
    return docs


def build_index() -> int:
    """Rebuild the FAISS index from scratch off whatever's currently in
    DOCUMENTS_DIR plus image captions. Returns the number of chunks indexed.
    Call this after every upload."""
    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs) if docs else []

    # Image captions are short and already atomic — don't split them further,
    # splitting could cut a caption in half and orphan it from its image_file tag.
    chunks.extend(load_image_documents())

    if not chunks:
        return 0

    index = FAISS.from_documents(chunks, get_embeddings())
    index.save_local(str(settings.vectorstore_path))
    return len(chunks)


def load_index() -> FAISS | None:
    index_file = settings.vectorstore_path / "index.faiss"
    if not index_file.exists():
        return None
    return FAISS.load_local(
        str(settings.vectorstore_path),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def retrieve(question: str, k: int = 4) -> list:
    """Return the k most relevant chunks for a question, each as a dict:
    {"text": ..., "source": ..., "image_file": ... or None}.
    Empty list if there's no index yet."""
    index = load_index()
    if index is None:
        return []
    results = index.similarity_search(question, k=k)
    return [
        {
            "text": r.page_content,
            "source": r.metadata.get("source", "unknown"),
            "image_file": r.metadata.get("image_file"),
        }
        for r in results
    ]
