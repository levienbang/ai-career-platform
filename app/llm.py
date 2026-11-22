"""Shared chat-model configuration: DeepSeek, with an optional Qwen backup on Bedrock.

Google embeddings are configured separately.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal

import openai
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.fallbacks import llm_fallbacks

logger = logging.getLogger(__name__)

# Any error from the provider's API goes to the backup: a timeout, 429 or 5xx, but also
# 401/402/403 (a revoked key, no balance left, credit used up), since the other provider
# can still answer. The failure is logged with its type and counted in the admin stats,
# so a misconfigured key still shows up. A missing key fails at build time instead.
FALLBACK_ERRORS: tuple[type[BaseException], ...] = (
    openai.APIError,  # status errors, connection errors and timeouts
    OutputParserException,
    ValidationError,
)
BEDROCK_MAX_TOKENS = 8_000  # every schema here is far smaller


class ChatModelConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class StructuredChatModel:
    provider: Literal["deepseek", "bedrock"]
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
    fallback: bool = True,
) -> StructuredChatModel:
    """Structured-output chat model for every LLM call in the app.

    LLM_PROVIDER picks the primary model (DeepSeek, or Bedrock for benchmarks). With
    DeepSeek as primary and LLM_FALLBACK=bedrock, runtime failures go to Bedrock;
    ``fallback=False`` skips that backup.
    """
    if settings.llm_provider == "bedrock":
        return StructuredChatModel(
            provider="bedrock",
            model=settings.bedrock_model,
            runnable=bedrock_structured_runnable(
                settings, schema, timeout_seconds=timeout_seconds, include_raw=include_raw
            ),
        )
    use_backup = fallback and _bedrock_backup_enabled(settings)
    chat_model = _deepseek_chat_model(
        settings, timeout_seconds=timeout_seconds, max_retries=max_retries, has_backup=use_backup
    )
    runnable: Runnable = chat_model.with_structured_output(
        schema, method=settings.deepseek_structured_output_method, include_raw=include_raw
    )
    if use_backup:
        backup = bedrock_structured_runnable(
            settings,
            schema,
            timeout_seconds=timeout_seconds,
            include_raw=include_raw,
            as_fallback=True,
        )
        runnable = _logged(runnable, "DeepSeek").with_fallbacks(
            [backup], exceptions_to_handle=FALLBACK_ERRORS
        )
    return StructuredChatModel(
        provider="deepseek", model=settings.deepseek_model.strip(), runnable=runnable
    )


def _bedrock_backup_enabled(settings: Settings) -> bool:
    """LLM_FALLBACK=bedrock without a Bedrock key runs DeepSeek alone instead of failing."""
    if settings.llm_fallback != "bedrock":
        return False
    if not settings.aws_bearer_token_bedrock:
        logger.warning("LLM_FALLBACK=bedrock but AWS_BEARER_TOKEN_BEDROCK is empty; no backup")
        return False
    return True


def _deepseek_chat_model(
    settings: Settings, *, timeout_seconds: float, max_retries: int, has_backup: bool
) -> ChatDeepSeek:
    secret = settings.deepseek_api_key
    api_key = secret.get_secret_value().strip() if secret else ""
    if not api_key:
        raise ChatModelConfigurationError("DEEPSEEK_API_KEY must be configured")
    model_name = settings.deepseek_model.strip()
    if not model_name:
        raise ChatModelConfigurationError("DEEPSEEK_MODEL must be configured")
    return ChatDeepSeek(
        model=model_name,
        api_key=api_key,
        base_url=settings.deepseek_base_url,
        temperature=0,
        timeout=timeout_seconds,
        # With a backup, one retry is enough: waiting out more retries on a failing
        # provider only delays the switch.
        max_retries=min(max_retries, 1) if has_backup else max_retries,
        extra_body={"thinking": {"type": settings.deepseek_thinking}},
    )


def _bedrock_chat_model(settings: Settings, *, timeout_seconds: float) -> ChatOpenAI:
    """Bedrock Mantle's OpenAI-compatible API, authenticated by a Bedrock API key."""
    if not settings.aws_bearer_token_bedrock:
        raise ChatModelConfigurationError(
            "Bedrock needs AWS_BEARER_TOKEN_BEDROCK (a Bedrock API key)"
        )
    return ChatOpenAI(
        model=settings.bedrock_model,
        api_key=settings.aws_bearer_token_bedrock,
        base_url=f"https://bedrock-mantle.{settings.bedrock_region}.api.aws/v1",
        temperature=0,
        max_tokens=BEDROCK_MAX_TOKENS,
        timeout=timeout_seconds,
        max_retries=1,
    )


def _logged(primary: Runnable, name: str) -> Runnable:
    """Log why the primary failed (error type only), then let with_fallbacks take over."""

    def run(value: Any, config: Any = None) -> Any:
        try:
            return primary.invoke(value, config)
        except FALLBACK_ERRORS as error:
            logger.warning("%s call failed (%s); trying the backup", name, type(error).__name__)
            raise

    return RunnableLambda(run, name=f"{name}_primary")


def _record_fallback(runnable: Runnable, used: str = "bedrock") -> Runnable:
    def record(result: Any) -> Any:
        llm_fallbacks.record(used)
        logger.warning("Served by the %s backup", used)
        return result

    return runnable | RunnableLambda(record, name="record_fallback")


def build_tool_chat_model(
    settings: Settings,
    tools: list[Any],
    *,
    timeout_seconds: float,
    max_retries: int,
) -> Runnable:
    """Free-text chat model (returns an AIMessage) that may call ``tools``.

    CHAT_PROVIDER picks the primary model for the conversation, separately from the
    agent's own calls: "bedrock" (Qwen) with DeepSeek as the backup, or "deepseek" with
    Qwen as the backup when LLM_FALLBACK=bedrock. An empty ``tools`` list gives a model
    that can only answer.
    """

    def bind(model: Any) -> Runnable:
        return model.bind_tools(tools) if tools else model

    def deepseek(has_backup: bool) -> Runnable:
        return bind(
            _deepseek_chat_model(
                settings,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                has_backup=has_backup,
            )
        )

    def bedrock() -> Runnable:
        return bind(_bedrock_chat_model(settings, timeout_seconds=timeout_seconds))

    if settings.chat_provider == "bedrock" and not settings.aws_bearer_token_bedrock:
        logger.warning(
            "CHAT_PROVIDER=bedrock but AWS_BEARER_TOKEN_BEDROCK is empty; using DeepSeek"
        )
    elif settings.chat_provider == "bedrock":
        if not settings.deepseek_api_key:
            return bedrock()
        backup = _record_fallback(deepseek(has_backup=False), used="deepseek")
        return _logged(bedrock(), "Bedrock").with_fallbacks(
            [backup], exceptions_to_handle=FALLBACK_ERRORS
        )
    if not _bedrock_backup_enabled(settings):
        return deepseek(has_backup=False)
    backup = _record_fallback(bedrock())
    return _logged(deepseek(has_backup=True), "DeepSeek").with_fallbacks(
        [backup], exceptions_to_handle=FALLBACK_ERRORS
    )


def bedrock_structured_runnable(
    settings: Settings,
    schema: type[BaseModel],
    *,
    timeout_seconds: float,
    include_raw: bool = False,
    as_fallback: bool = False,
    client: Any | None = None,
) -> Runnable:
    """An open model on Bedrock (Qwen by default) with the same input and output as DeepSeek.

    It goes through Bedrock Mantle's OpenAI-compatible API with LangChain's ChatOpenAI,
    authenticated by a Bedrock API key. ``client`` injects a fake chat model in tests.
    ``as_fallback`` counts each call it serves in the admin stats.
    """
    chat_model = client or _bedrock_chat_model(settings, timeout_seconds=timeout_seconds)
    runnable = chat_model.with_structured_output(
        schema, method=settings.bedrock_structured_output_method, include_raw=include_raw
    )
    return _record_fallback(runnable) if as_fallback else runnable


class LLMCheckResult(BaseModel):
    ok: bool
    language: str


def check_structured_output(settings: Settings) -> tuple[str, LLMCheckResult]:
    """Send one small structured-output request; returns (model name, parsed result).

    Used by scripts/check_deepseek.py and the admin "check LLM" button. It spends one
    paid request, so nothing calls it automatically.
    """
    from langchain_core.prompts import ChatPromptTemplate

    selected = build_structured_chat_model(
        settings,
        LLMCheckResult,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=0,
        fallback=False,  # the check is about DeepSeek itself
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Return ok=true and language='English'."
                + structured_output_guidance(settings, LLMCheckResult),
            ),
            ("human", "Check structured output."),
        ]
    )
    result = LLMCheckResult.model_validate(selected.runnable.invoke(prompt.invoke({})))
    return selected.model, result
