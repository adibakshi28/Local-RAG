from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # paths
    PDF_DIR: str = "app/data/pdfs"
    CHROMA_DIR: str = "app/data/chroma"

    # retrieval
    TOP_K: int = 6
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 150
    USE_RERANKER: bool = True
    COMPRESS_BEFORE_ANSWER: bool = False

    # DeepSeek (REST)
    DEEPSEEK_API_KEY: str | None = None
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-chat"

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()
