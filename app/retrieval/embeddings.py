import math
from typing import Protocol

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import Settings


class EmbeddingConfigurationError(RuntimeError):
    pass


class EmbeddingServiceError(RuntimeError):
    pass


class EmbeddingProvider(Protocol):
    @property
    def dimensions(self) -> int: ...

    def embed_documents(self, texts: list[str], titles: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        raise EmbeddingServiceError("Embedding provider returned a zero vector")
    return [value / magnitude for value in vector]


class GoogleJobEmbeddingProvider:
    def __init__(self, settings: Settings) -> None:
        if settings.embedding_provider.casefold() != "google":
            raise EmbeddingConfigurationError(
                f"Unsupported embedding provider '{settings.embedding_provider}'. Supported: google"
            )
        api_key = ""
        if settings.embedding_api_key:
            api_key = settings.embedding_api_key.get_secret_value()
        if not api_key and settings.llm_api_key:
            api_key = settings.llm_api_key.get_secret_value()
        if not api_key:
            raise EmbeddingConfigurationError(
                "EMBEDDING_API_KEY or LLM_API_KEY must be configured for dense retrieval"
            )
        if not settings.embedding_model:
            raise EmbeddingConfigurationError(
                "EMBEDDING_MODEL must be configured for dense retrieval"
            )

        self._dimensions = settings.embedding_dimensions
        self._model = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=api_key,
            output_dimensionality=settings.embedding_dimensions,
            request_options={"timeout": settings.embedding_timeout_seconds},
        )

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: list[str], titles: list[str]) -> list[list[float]]:
        if len(texts) != len(titles):
            raise ValueError("Each search document must have one title")
        try:
            vectors = self._model.embed_documents(
                texts,
                task_type="RETRIEVAL_DOCUMENT",
                titles=titles,
            )
            return [_normalize(vector) for vector in vectors]
        except EmbeddingServiceError:
            raise
        except Exception as error:
            raise EmbeddingServiceError("Document embedding request failed") from error

    def embed_query(self, text: str) -> list[float]:
        try:
            return _normalize(self._model.embed_query(text, task_type="RETRIEVAL_QUERY"))
        except EmbeddingServiceError:
            raise
        except Exception as error:
            raise EmbeddingServiceError("Query embedding request failed") from error


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    return GoogleJobEmbeddingProvider(settings)
