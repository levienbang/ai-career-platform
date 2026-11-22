"""DeepSeek -> Qwen-on-Bedrock fallback for structured output, with fake chat models."""

import httpx
import openai
import pytest
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

import app.llm as llm
from app.config import Settings
from app.fallbacks import llm_fallbacks


class Route(BaseModel):
    route: str


SQL_ROUTE = Route(route="sql")
PROMPT = ChatPromptTemplate.from_messages([("system", "Pick a route."), ("human", "{q}")])
REQUEST = httpx.Request("POST", "https://api.deepseek.com")


class FakeChatModel:
    """Stands in for the ChatOpenAI pointed at Bedrock Mantle."""

    def __init__(self) -> None:
        self.calls: list = []

    def with_structured_output(self, schema, *, method, include_raw=False):
        def run(value):
            self.calls.append({"schema": schema, "method": method, "messages": value})
            parsed = SQL_ROUTE
            return {"raw": None, "parsed": parsed, "parsing_error": None} if include_raw else parsed

        return RunnableLambda(run)


def settings(**overrides) -> Settings:
    overrides.setdefault("deepseek_api_key", "test-key")
    overrides.setdefault("aws_bearer_token_bedrock", "bedrock-key")
    return Settings(_env_file=None, **overrides)


@pytest.fixture
def deepseek(monkeypatch):
    """Replace ChatDeepSeek; the test sets what its structured runnable does."""
    state = {"behaviour": lambda value: Route(route="search"), "max_retries": None}

    class FakeChatDeepSeek:
        def __init__(self, **kwargs) -> None:
            state["max_retries"] = kwargs["max_retries"]
            self.model_name = kwargs["model"]

        def with_structured_output(self, schema, **kwargs):
            return RunnableLambda(lambda value: state["behaviour"](value))

    monkeypatch.setattr(llm, "ChatDeepSeek", FakeChatDeepSeek)
    return state


@pytest.fixture
def backup(monkeypatch):
    chat = FakeChatModel()
    real = llm.bedrock_structured_runnable

    def build_backup(settings, schema, **kwargs):
        return real(settings, schema, client=chat, **kwargs)

    monkeypatch.setattr(llm, "bedrock_structured_runnable", build_backup)
    llm_fallbacks.reset()
    return chat


def build(config: Settings, **kwargs):
    return llm.build_structured_chat_model(
        config, Route, timeout_seconds=5, max_retries=2, **kwargs
    ).runnable


def raise_(error):
    def run(value):
        raise error

    return run


def test_without_fallback_deepseek_errors_propagate(deepseek, backup) -> None:
    deepseek["behaviour"] = raise_(openai.APITimeoutError(request=REQUEST))

    with pytest.raises(openai.APITimeoutError):
        build(settings()).invoke(PROMPT.invoke({"q": "count jobs"}))
    assert deepseek["max_retries"] == 2
    assert backup.calls == []


@pytest.mark.parametrize(
    "error",
    [
        openai.APITimeoutError(request=REQUEST),
        openai.RateLimitError(
            "slow down", response=httpx.Response(429, request=REQUEST), body=None
        ),
        openai.InternalServerError(
            "down", response=httpx.Response(503, request=REQUEST), body=None
        ),
    ],
)
def test_runtime_failure_is_served_by_bedrock(deepseek, backup, error) -> None:
    deepseek["behaviour"] = raise_(error)

    result = build(settings(llm_fallback="bedrock")).invoke(PROMPT.invoke({"q": "count jobs"}))

    assert result == SQL_ROUTE
    assert deepseek["max_retries"] == 1  # do not wait out every retry before switching
    call = backup.calls[0]
    assert call["schema"] is Route
    assert call["method"] == "function_calling"
    assert [m.content for m in call["messages"].to_messages()] == ["Pick a route.", "count jobs"]
    assert llm_fallbacks.snapshot() == {"bedrock": 1}


def test_healthy_deepseek_never_calls_bedrock(deepseek, backup) -> None:
    result = build(settings(llm_fallback="bedrock")).invoke(PROMPT.invoke({"q": "find jobs"}))

    assert result == Route(route="search")
    assert backup.calls == []
    assert llm_fallbacks.snapshot() == {}


def test_configuration_errors_do_not_fall_back(deepseek, backup) -> None:
    deepseek["behaviour"] = raise_(
        openai.AuthenticationError(
            "bad key", response=httpx.Response(401, request=REQUEST), body=None
        )
    )

    with pytest.raises(openai.AuthenticationError):
        build(settings(llm_fallback="bedrock")).invoke(PROMPT.invoke({"q": "x"}))
    assert backup.calls == []


def test_fallback_can_be_turned_off_per_call(deepseek, backup) -> None:
    deepseek["behaviour"] = raise_(openai.APITimeoutError(request=REQUEST))

    with pytest.raises(openai.APITimeoutError):
        build(settings(llm_fallback="bedrock"), fallback=False).invoke(PROMPT.invoke({"q": "x"}))


def test_include_raw_keeps_the_langchain_shape(deepseek, backup) -> None:
    deepseek["behaviour"] = raise_(openai.APITimeoutError(request=REQUEST))

    result = build(settings(llm_fallback="bedrock"), include_raw=True).invoke(
        PROMPT.invoke({"q": "x"})
    )

    assert result == {"raw": None, "parsed": SQL_ROUTE, "parsing_error": None}


def test_bedrock_as_primary_skips_deepseek(deepseek, backup) -> None:
    deepseek["behaviour"] = raise_(AssertionError("DeepSeek must not run"))

    selected = llm.build_structured_chat_model(
        settings(llm_provider="bedrock"), Route, timeout_seconds=5, max_retries=2
    )

    assert selected.provider == "bedrock"
    assert selected.model == "qwen.qwen3-235b-a22b-2507"
    assert selected.runnable.invoke(PROMPT.invoke({"q": "x"})) == SQL_ROUTE
    assert llm_fallbacks.snapshot() == {}  # a primary call is not a fallback


def test_bedrock_needs_an_api_key() -> None:
    with pytest.raises(llm.ChatModelConfigurationError, match="AWS_BEARER_TOKEN_BEDROCK"):
        llm.bedrock_structured_runnable(
            settings(aws_bearer_token_bedrock=None), Route, timeout_seconds=5
        )


def test_backup_without_a_bedrock_key_runs_deepseek_alone(deepseek, backup) -> None:
    deepseek["behaviour"] = raise_(openai.APITimeoutError(request=REQUEST))
    config = settings(llm_fallback="bedrock", aws_bearer_token_bedrock=None)

    with pytest.raises(openai.APITimeoutError):
        build(config).invoke(PROMPT.invoke({"q": "x"}))
    assert backup.calls == []


def test_qwen_chat_without_a_bedrock_key_uses_deepseek(monkeypatch) -> None:
    deepseek = FakeToolModel(reply="from deepseek")
    tool_models(monkeypatch, deepseek, FakeToolModel(reply="from qwen"))
    config = settings(chat_provider="bedrock", aws_bearer_token_bedrock=None)

    model = llm.build_tool_chat_model(config, ["tool"], timeout_seconds=5, max_retries=1)

    assert model.invoke("hi") == "from deepseek"


def test_bedrock_client_targets_mantle() -> None:
    runnable = llm.bedrock_structured_runnable(
        settings(aws_bearer_token_bedrock="key", bedrock_region="us-west-2"),
        Route,
        timeout_seconds=5,
    )

    chat_model = runnable.first if hasattr(runnable, "first") else runnable.bound
    assert chat_model.openai_api_base == "https://bedrock-mantle.us-west-2.api.aws/v1"
    assert chat_model.model_name == "qwen.qwen3-235b-a22b-2507"


class FakeToolModel:
    """Stands in for a chat model with bind_tools."""

    def __init__(self, reply=None, error=None) -> None:
        self.reply, self.error, self.tools = reply, error, None

    def bind_tools(self, tools):
        self.tools = tools
        return RunnableLambda(self._run)

    def _run(self, value):
        if self.error:
            raise self.error
        return self.reply


def tool_models(monkeypatch, deepseek_model, bedrock_model):
    monkeypatch.setattr(llm, "_deepseek_chat_model", lambda *a, **k: deepseek_model)
    monkeypatch.setattr(llm, "_bedrock_chat_model", lambda *a, **k: bedrock_model)
    llm_fallbacks.reset()


def test_chat_uses_qwen_first_and_falls_back_to_deepseek(monkeypatch) -> None:
    qwen = FakeToolModel(error=openai.APITimeoutError(request=REQUEST))
    deepseek = FakeToolModel(reply="from deepseek")
    tool_models(monkeypatch, deepseek, qwen)

    model = llm.build_tool_chat_model(
        settings(chat_provider="bedrock"), ["tool"], timeout_seconds=5, max_retries=1
    )

    assert model.invoke("hi") == "from deepseek"
    assert qwen.tools == ["tool"] and deepseek.tools == ["tool"]
    assert llm_fallbacks.snapshot() == {"deepseek": 1}


def test_chat_with_deepseek_primary_falls_back_to_qwen(monkeypatch) -> None:
    deepseek = FakeToolModel(error=openai.APITimeoutError(request=REQUEST))
    qwen = FakeToolModel(reply="from qwen")
    tool_models(monkeypatch, deepseek, qwen)

    model = llm.build_tool_chat_model(
        settings(llm_fallback="bedrock"), ["tool"], timeout_seconds=5, max_retries=1
    )

    assert model.invoke("hi") == "from qwen"
    assert llm_fallbacks.snapshot() == {"bedrock": 1}


def test_answer_only_chat_model_binds_no_tools(monkeypatch) -> None:
    qwen = FakeToolModel(reply="x")
    tool_models(monkeypatch, FakeToolModel(reply="y"), qwen)

    model = llm.build_tool_chat_model(
        settings(chat_provider="bedrock", deepseek_api_key=None),
        [],
        timeout_seconds=5,
        max_retries=1,
    )

    assert model is qwen  # no tools, no backup without a DeepSeek key
