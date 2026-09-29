import pytest

from app.config import Settings
from app.retrieval.embeddings import (
    EmbeddingConfigurationError,
    GoogleJobEmbeddingProvider,
)


def test_embedding_provider_requires_api_key() -> None:
    settings = Settings(
        embedding_api_key=None,
        llm_api_key="chat-only-key",
        _env_file=None,
    )

    with pytest.raises(EmbeddingConfigurationError, match="EMBEDDING_API_KEY"):
        GoogleJobEmbeddingProvider(settings)


def test_embedding_provider_rejects_unknown_provider() -> None:
    settings = Settings(
        embedding_provider="unknown",
        embedding_api_key="test-key",
        _env_file=None,
    )

    with pytest.raises(EmbeddingConfigurationError, match="Unsupported embedding provider"):
        GoogleJobEmbeddingProvider(settings)
