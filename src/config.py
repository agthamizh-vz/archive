"""Project configuration loaded from `.env`.

This file is intentionally simple and shared by all modules.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    google_api_key: str = ""
    llm_model: str = "gemini-2.0-flash"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index_name: str = "viewzen-docs"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Retrieval / chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5

    # Runtime
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Files
    data_dir: str = "data"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# Commonly used paths
DATA_DIR = Path(settings.data_dir)
RAW_PATH = DATA_DIR / "raw" / "scraped_docs.json"
CHUNKS_PATH = DATA_DIR / "processed" / "chunks.json"