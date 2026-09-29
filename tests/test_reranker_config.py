import pytest

from app.config import Settings
from app.retrieval.reranker import LangChainJobReranker, RerankerConfigurationError


def test_reranker_requires_api_key() -> None:
    settings = Settings(reranker_api_key=None, llm_api_key=None, _env_file=None)

    with pytest.raises(RerankerConfigurationError, match="Gemini API key or OLLAMA_BASE_URL"):
        LangChainJobReranker(settings)
