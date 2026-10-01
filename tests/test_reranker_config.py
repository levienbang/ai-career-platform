import pytest

from app.config import Settings
from app.retrieval.reranker import LangChainJobReranker, RerankerConfigurationError


def test_reranker_requires_api_key() -> None:
    settings = Settings(deepseek_api_key=None, _env_file=None)

    with pytest.raises(RerankerConfigurationError, match="DEEPSEEK_API_KEY"):
        LangChainJobReranker(settings)
