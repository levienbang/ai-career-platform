from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    database_url: str = "postgresql+psycopg://career:career@localhost:5432/career"
    llm_provider: str = "openai"
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    max_import_bytes: int = Field(default=1_000_000, gt=0, le=10_000_000)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
