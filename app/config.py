from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    documents_dir: str = "data/documents"
    images_dir: str = "data/images"
    vectorstore_dir: str = "vectorstore"
    frontend_origin: str = "http://localhost:5173"

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
