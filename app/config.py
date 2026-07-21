from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    documents_dir: str = "data/documents"
    images_dir: str = "data/images"
    vectorstore_dir: str = "vectorstore"
    frontend_origin: str = "http://localhost:5173"
    # FAISS returns L2 distance — LOWER means more similar. Chunks with a
    # distance above this are treated as "not actually relevant" and dropped
    # before ever reaching the LLM, instead of letting the model see weak
    # matches and blend in outside knowledge to compensate.
    # This needs tuning on YOUR real documents — see README for how.
    similarity_threshold: float = 0.5
    # Separate, STRICTER threshold just for images. Attaching the wrong photo
    # to an unrelated reply is more jarring than a slightly-off text answer,
    # so this needs to be tighter than similarity_threshold.
    image_similarity_threshold: float = 0.4

    @property
    def documents_path(self) -> Path:
        p = Path(self.documents_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def images_path(self) -> Path:
        p = Path(self.images_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def vectorstore_path(self) -> Path:
        p = Path(self.vectorstore_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()