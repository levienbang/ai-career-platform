"""Shared DeepSeek chat configuration; Google embeddings are configured separately."""

import json
from dataclasses import dataclass
from typing import Literal

from langchain_core.runnables import Runnable
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel

from app.config import Settings


class ChatModelConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class StructuredChatModel:
    provider: Literal["deepseek"]
    model: str
    runnable: Runnable


def structured_output_guidance(settings: Settings, schema: type[BaseModel]) -> str:
    """Return JSON-mode instructions with braces escaped for ChatPromptTemplate."""
    if settings.deepseek_structured_output_method != "json_mode":
        return ""
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    return (
        ("\nReturn one JSON object matching this schema: " + schema_json)
        .replace("{", "{{")
        .replace("}", "}}")
    )


def build_structured_chat_model(
    settings: Settings,
    schema: type[BaseModel],
    *,
    timeout_seconds: float,
    max_retries: int,
    include_raw: bool = False,
) -> StructuredChatModel:
    secret = settings.deepseek_api_key
    api_key = secret.get_secret_value().strip() if secret else ""
    if not api_key:
        raise ChatModelConfigurationError("DEEPSEEK_API_KEY must be configured")
    model_name = settings.deepseek_model.strip()
    if not model_name:
        raise ChatModelConfigurationError("DEEPSEEK_MODEL must be configured")
    chat_model = ChatDeepSeek(
        model=model_name,
        api_key=api_key,
        base_url=settings.deepseek_base_url,
        temperature=0,
        timeout=timeout_seconds,
        max_retries=max_retries,
        extra_body={"thinking": {"type": settings.deepseek_thinking}},
    )
    return StructuredChatModel(
        provider="deepseek",
        model=model_name,
        runnable=chat_model.with_structured_output(
            schema, method=settings.deepseek_structured_output_method, include_raw=include_raw
        ),
    )
