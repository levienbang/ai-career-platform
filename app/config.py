from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    database_url: str = "postgresql+psycopg://career:career@localhost:5432/career"
    readonly_database_url: str | None = None
    llm_model: str = "gemini-3.5-flash"
    llm_api_key: SecretStr | None = None
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    ollama_base_url: str | None = None
    ollama_model: str = "gemma3:12b"
    ollama_timeout_seconds: float = Field(default=180.0, gt=0, le=600)
    ollama_num_predict: int = Field(default=1024, ge=128, le=8192)
    agent_model: str = "gemini-3.5-flash"
    agent_api_key: SecretStr | None = None
    agent_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    agent_max_retries: int = Field(default=1, ge=0, le=5)
    agent_search_limit: int = Field(default=5, ge=1, le=20)
    agent_tool_max_retries: int = Field(default=1, ge=0, le=2)
    sql_max_rows: int = Field(default=100, ge=1, le=1_000)
    sql_statement_timeout_ms: int = Field(default=3_000, ge=100, le=30_000)
    embedding_provider: str = "google"
    embedding_model: str = "gemini-embedding-001"
    embedding_api_key: SecretStr | None = None
    embedding_dimensions: int = Field(default=768, ge=128, le=3072)
    embedding_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    hybrid_candidate_limit: int = Field(default=20, ge=2, le=100)
    rrf_k: int = Field(default=60, ge=1, le=1_000)
    reranker_model: str = "gemini-3.5-flash"
    reranker_api_key: SecretStr | None = None
    reranker_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    reranker_max_retries: int = Field(default=1, ge=0, le=5)
    reranker_max_candidates: int = Field(default=20, ge=2, le=50)
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "jobs"
    qdrant_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    max_import_bytes: int = Field(default=1_000_000, gt=0, le=10_000_000)
    max_cv_bytes: int = Field(default=2_000_000, gt=0, le=10_000_000)
    max_cv_pages: int = Field(default=5, ge=1, le=20)
    cv_model: str = "gemini-3.5-flash"
    cv_api_key: SecretStr | None = None
    cv_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "ai-career-platform"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
