import pytest

from app.config import Settings
from app.retrieval.reranker import GoogleJobReranker, RerankerConfigurationError


def test_reranker_requires_api_key() -> None:
    settings = Settings(reranker_api_key=None, llm_api_key=None, _env_file=None)

    with pytest.raises(RerankerConfigurationError, match="API_KEY"):
        GoogleJobReranker(settings)


def test_reranker_rejects_unknown_provider() -> None:
    settings = Settings(
        reranker_provider="unknown",
        reranker_api_key="test-key",
        _env_file=None,
    )

    with pytest.raises(RerankerConfigurationError, match="Unsupported reranker provider"):
        GoogleJobReranker(settings)
