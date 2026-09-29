"""Select the configured structured-output chat model for application services."""

from dataclasses import dataclass
from typing import Literal

from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from pydantic import BaseModel, SecretStr

from app.config import Settings


class ChatModelConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class StructuredChatModel:
    provider: Literal["gemini", "ollama"]
    runnable: Runnable


def build_structured_chat_model(
    settings: Settings,
    schema: type[BaseModel],
    *,
    api_keys: tuple[SecretStr | None, ...],
    gemini_model: str,
    timeout_seconds: float,
    max_retries: int,
) -> StructuredChatModel:
    """Prefer a Gemini key; otherwise use Ollama when its URL is configured."""
    api_key = ""
    for secret in api_keys:
        if secret:
            api_key = secret.get_secret_value().strip()
        if api_key:
            break
    if api_key:
        if not gemini_model.strip():
            raise ChatModelConfigurationError("A Gemini model name must be configured")
        model = ChatGoogleGenerativeAI(
            model=gemini_model,
            api_key=api_key,
            temperature=0,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )
        return StructuredChatModel(
            provider="gemini",
            runnable=model.with_structured_output(schema, method="json_schema"),
        )

    base_url = (settings.ollama_base_url or "").strip()
    if not base_url:
        raise ChatModelConfigurationError(
            "Configure a Gemini API key or OLLAMA_BASE_URL for local inference"
        )
    if not settings.ollama_model.strip():
        raise ChatModelConfigurationError("OLLAMA_MODEL must be configured")
    model = ChatOllama(
        model=settings.ollama_model,
        base_url=base_url,
        temperature=0,
        num_predict=settings.ollama_num_predict,
        client_kwargs={"timeout": settings.ollama_timeout_seconds},
    )
    return StructuredChatModel(
        provider="ollama",
        runnable=model.with_structured_output(schema, method="json_schema"),
    )
