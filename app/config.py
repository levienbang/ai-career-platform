from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    skill_data_dir: Path = Path("data")
    skill_alias_learning: bool = True
    skill_alias_auto_confidence: float = Field(default=0.9, ge=0.5, le=1.0)
    skill_alias_batch_size: int = Field(default=40, ge=1, le=100)
    embedding_batch_size: int = Field(default=20, ge=1, le=100)
    embedding_requests_per_minute: int = Field(default=0, ge=0, le=1000)
    embedding_max_retries: int = Field(default=5, ge=0, le=10)
    embedding_retry_base_seconds: float = Field(default=20, ge=1, le=300)

    @field_validator("skill_data_dir", mode="before")
    @classmethod
    def default_skill_data_dir(cls, value):
        return value or Path("data")

    database_url: str = "postgresql+psycopg://career:career@localhost:5432/career"
    readonly_database_url: str | None = None
    deepseek_api_key: SecretStr | None = None
    deepseek_model: str = Field(default="deepseek-v4-flash", min_length=1)
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_structured_output_method: Literal["function_calling", "json_mode"] = "function_calling"
    # deepseek-v4-flash thinks by default, and thinking mode rejects forced tool calls.
    deepseek_thinking: Literal["enabled", "disabled"] = "disabled"
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_batch_size: int = Field(default=8, ge=1, le=25)
    llm_requests_per_minute: int = Field(default=0, ge=0, le=1000)
    llm_structured_fast_path: bool = True
    llm_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
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
    reranker_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    reranker_max_retries: int = Field(default=1, ge=0, le=5)
    reranker_max_candidates: int = Field(default=20, ge=2, le=50)
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "jobs"
    qdrant_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    max_import_bytes: int = Field(default=1_000_000, gt=0, le=10_000_000)
    max_cv_bytes: int = Field(default=2_000_000, gt=0, le=10_000_000)
    max_cv_pages: int = Field(default=5, ge=1, le=20)
    cv_match_candidates: int = Field(default=50, ge=1, le=200)
    cv_match_semantic_weight: float = Field(default=0.6, ge=0, le=1)
    cv_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "ai-career-platform"

    @field_validator("deepseek_model")
    @classmethod
    def validate_chat_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("DEEPSEEK_MODEL must not be empty")
        return value.strip()

    @field_validator("deepseek_base_url")
    @classmethod
    def validate_chat_url(cls, value: str) -> str:
        from pydantic import HttpUrl

        HttpUrl(value)
        return value.strip()

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
