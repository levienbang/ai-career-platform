import json

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.graph.router import AgentRouterConfigurationError, LangChainQuestionRouter, RouteDecision
from app.ingestion.extractor import ExtractorConfigurationError, LangChainJobExtractor
from app.llm import ChatModelConfigurationError, build_structured_chat_model
from app.retrieval.reranker import LangChainJobReranker, RerankerConfigurationError, RerankOutput
from app.schemas.cv import CVExtraction
from app.schemas.ingestion import JobBatchExtraction, JobExtraction
from app.services.cv_extractor import CVExtractorError, LangChainCVExtractor
from app.tools.sql_tool import LangChainSQLGenerator, SQLPlan, SQLToolConfigurationError


class FakeChatModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def with_structured_output(self, schema, *, method, include_raw):
        assert method != "json_schema"
        self.schema = schema
        self.method = method
        self.include_raw = include_raw
        return self


@pytest.fixture
def chat_models(monkeypatch):
    calls = []

    def create(**kwargs):
        instance = FakeChatModel(**kwargs)
        calls.append(instance)
        return instance

    monkeypatch.setattr("app.llm.ChatDeepSeek", create)
    return calls


COMPONENTS = [
    (LangChainJobExtractor, JobExtraction, ExtractorConfigurationError),
    (LangChainCVExtractor, CVExtraction, CVExtractorError),
    (LangChainQuestionRouter, RouteDecision, AgentRouterConfigurationError),
    (LangChainSQLGenerator, SQLPlan, SQLToolConfigurationError),
    (LangChainJobReranker, RerankOutput, RerankerConfigurationError),
]


@pytest.mark.parametrize("method", ["function_calling", "json_mode"])
@pytest.mark.parametrize("include_raw", [False, True])
def test_deepseek_constructor_and_output_options(chat_models, method, include_raw):
    settings = Settings(
        deepseek_api_key=" test-key ", deepseek_structured_output_method=method, _env_file=None
    )
    selected = build_structured_chat_model(
        settings,
        JobExtraction,
        timeout_seconds=17,
        max_retries=2,
        include_raw=include_raw,
    )
    assert selected.provider == "deepseek"
    assert selected.model == "deepseek-v4-flash"
    assert chat_models[0].kwargs == dict(
        model=selected.model,
        api_key="test-key",
        base_url="https://api.deepseek.com",
        temperature=0,
        timeout=17,
        max_retries=2,
        extra_body={"thinking": {"type": "disabled"}},
    )
    assert selected.runnable.schema is JobExtraction
    assert selected.runnable.method == method
    assert selected.runnable.include_raw is include_raw


@pytest.mark.parametrize("key", [None, "", "   "])
def test_missing_key(chat_models, key):
    settings = Settings(deepseek_api_key=key, _env_file=None)
    with pytest.raises(ChatModelConfigurationError, match="DEEPSEEK_API_KEY must be configured"):
        build_structured_chat_model(settings, JobExtraction, timeout_seconds=30, max_retries=0)
    assert not chat_models


@pytest.mark.parametrize("component,schema,error", COMPONENTS)
@pytest.mark.parametrize("key", [None, "   "])
def test_component_configuration_errors(chat_models, component, schema, error, key):
    settings = Settings(deepseek_api_key=key, embedding_api_key="embedding-only", _env_file=None)
    with pytest.raises(error, match="DEEPSEEK_API_KEY"):
        component(settings)
    assert not chat_models


@pytest.mark.parametrize("component,schema,error", COMPONENTS)
@pytest.mark.parametrize("method", ["function_calling", "json_mode"])
def test_components_use_deepseek_model_and_schema_prompt(
    chat_models, component, schema, error, method
):
    settings = Settings(
        deepseek_api_key="fake",
        deepseek_model="default-model",
        deepseek_structured_output_method=method,
        _env_file=None,
    )
    instance = component(settings)
    assert instance._model.kwargs["model"] == "default-model"
    assert instance._model.schema is schema
    assert instance._model.method == method
    # Invoke the template: embedded schema braces must not become input variables.
    values = {name: "untrusted input" for name in instance._prompt.input_variables}
    system = instance._prompt.invoke(values).messages[0].content
    marker = "Return one JSON object matching this schema: "
    if method == "json_mode":
        assert json.loads(system.split(marker)[1]) == schema.model_json_schema()
    else:
        assert marker not in system
    if component is LangChainJobExtractor:
        batch_system = instance._batch_prompt.invoke({"records": "[]"}).messages[0].content
        if method == "json_mode":
            assert (
                json.loads(batch_system.split(marker)[1]) == JobBatchExtraction.model_json_schema()
            )
        else:
            assert marker not in batch_system


@pytest.mark.parametrize("method", ["json_schema", "invalid", ""])
def test_settings_reject_unsupported_methods(method):
    with pytest.raises(ValidationError):
        Settings(deepseek_structured_output_method=method, _env_file=None)


@pytest.mark.parametrize("model", ["", "   "])
def test_settings_reject_empty_default_model(model):
    with pytest.raises(ValidationError, match="deepseek_model"):
        Settings(deepseek_model=model, _env_file=None)


@pytest.mark.parametrize("url", ["file:///tmp/model", "ftp://example.com", "not a url", "https://"])
def test_settings_reject_invalid_base_url(url):
    with pytest.raises(ValidationError, match="deepseek_base_url"):
        Settings(deepseek_base_url=url, _env_file=None)


def test_custom_endpoint(chat_models):
    settings = Settings(
        deepseek_api_key="fake", deepseek_base_url="http://localhost:8080/v1", _env_file=None
    )
    selected = build_structured_chat_model(
        settings, JobExtraction, timeout_seconds=30, max_retries=0
    )
    assert selected.model == settings.deepseek_model
    assert chat_models[0].kwargs["base_url"] == "http://localhost:8080/v1"


def test_empty_model_after_fallback_reports_configuration_error(chat_models):
    settings = Settings(deepseek_api_key="fake", _env_file=None).model_copy(
        update={"deepseek_model": ""}
    )
    with pytest.raises(ChatModelConfigurationError, match="DEEPSEEK_MODEL"):
        build_structured_chat_model(settings, JobExtraction, timeout_seconds=30, max_retries=0)


@pytest.mark.parametrize("method", ["function_calling", "json_mode"])
def test_batch_include_raw_salvages_valid_items(chat_models, method, monkeypatch):
    from langchain_core.messages import AIMessage

    from app.schemas.ingestion import RawJobRecord

    payload = {
        "items": [
            {"record_index": 1, "title": "Python job", "description": "Python"},
            {"title": "Missing index", "description": "Python"},
        ]
    }
    raw = (
        AIMessage(
            content="", tool_calls=[{"name": "JobBatchExtraction", "args": payload, "id": "call-1"}]
        )
        if method == "function_calling"
        else AIMessage(
            content=json.dumps(payload), additional_kwargs={"reasoning_content": "Ignore reasoning"}
        )
    )
    # Exercise the shared builder, lazy batch model, provider envelope and final validation.
    monkeypatch.setattr(
        FakeChatModel,
        "invoke",
        lambda self, messages: {
            "raw": raw,
            "parsed": None,
            "parsing_error": ValueError("invalid index"),
        },
        raising=False,
    )
    extractor = LangChainJobExtractor(
        Settings(
            deepseek_api_key="fake",
            deepseek_structured_output_method=method,
            llm_max_retries=0,
            _env_file=None,
        )
    )
    result = extractor.extract_many(
        [
            RawJobRecord(title="Original", description="Python"),
            RawJobRecord(title="Other", description="Python"),
        ]
    )
    assert result[0].title == "Original"
    assert result[1] is None
    assert chat_models[0].include_raw is False
    assert chat_models[1].schema is JobBatchExtraction
    assert chat_models[1].include_raw is True
    assert chat_models[1].method == method
    assert extractor.llm_calls == 1


@pytest.mark.parametrize("thinking", ["enabled", "disabled"])
def test_thinking_mode_is_sent_to_deepseek(chat_models, thinking):
    settings = Settings(deepseek_api_key="k", deepseek_thinking=thinking, _env_file=None)
    build_structured_chat_model(settings, JobExtraction, timeout_seconds=5, max_retries=0)
    assert chat_models[0].kwargs["extra_body"] == {"thinking": {"type": thinking}}


def test_settings_reject_unknown_thinking_mode():
    with pytest.raises(ValidationError):
        Settings(deepseek_thinking="auto", _env_file=None)
