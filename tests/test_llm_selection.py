import pytest

from app.config import Settings
from app.graph.router import LangChainQuestionRouter, RouteDecision
from app.ingestion.extractor import (
    LOCAL_EXTRACTION_GUIDANCE,
    LangChainJobExtractor,
)
from app.llm import ChatModelConfigurationError, build_structured_chat_model
from app.retrieval.reranker import LangChainJobReranker, RerankOutput
from app.schemas.cv import CVExtraction
from app.schemas.ingestion import JobExtraction
from app.services.cv_extractor import LangChainCVExtractor
from app.tools.sql_tool import LangChainSQLGenerator, SQLPlan


class FakeChatModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.schema = None

    def with_structured_output(self, schema, *, method):
        assert method == "json_schema"
        self.schema = schema
        return self


@pytest.fixture
def chat_models(monkeypatch):
    calls = []

    def build(provider):
        def create(**kwargs):
            instance = FakeChatModel(**kwargs)
            calls.append((provider, instance))
            return instance

        return create

    monkeypatch.setattr("app.llm.ChatGoogleGenerativeAI", build("gemini"))
    monkeypatch.setattr("app.llm.ChatOllama", build("ollama"))
    return calls


def test_gemini_key_takes_priority_over_ollama_url(chat_models):
    settings = Settings(
        llm_api_key="gemini-key", ollama_base_url="http://localhost:11434", _env_file=None
    )
    selected = build_structured_chat_model(
        settings,
        JobExtraction,
        api_keys=(settings.llm_api_key,),
        gemini_model=settings.llm_model,
        timeout_seconds=30,
        max_retries=0,
    )

    assert selected.provider == "gemini"
    assert chat_models[0][0] == "gemini"
    assert chat_models[0][1].kwargs["api_key"] == "gemini-key"
    assert chat_models[0][1].schema is JobExtraction


def test_ollama_used_without_chat_key(chat_models):
    settings = Settings(llm_api_key="  ", ollama_base_url="http://localhost:11434", _env_file=None)
    selected = build_structured_chat_model(
        settings,
        JobExtraction,
        api_keys=(settings.llm_api_key,),
        gemini_model=settings.llm_model,
        timeout_seconds=30,
        max_retries=0,
    )

    assert selected.provider == "ollama"
    assert chat_models[0][1].kwargs["model"] == "gemma3:12b"
    assert chat_models[0][1].kwargs["base_url"] == "http://localhost:11434"
    assert chat_models[0][1].schema is JobExtraction


def test_missing_key_and_url_reports_configuration_error(chat_models):
    settings = Settings(llm_api_key=None, ollama_base_url=None, _env_file=None)
    with pytest.raises(ChatModelConfigurationError, match="OLLAMA_BASE_URL"):
        build_structured_chat_model(
            settings,
            JobExtraction,
            api_keys=(settings.llm_api_key,),
            gemini_model=settings.llm_model,
            timeout_seconds=30,
            max_retries=0,
        )
    assert chat_models == []


def test_every_chat_component_uses_local_structured_output_and_guidance(chat_models):
    settings = Settings(
        llm_api_key=None,
        agent_api_key=None,
        reranker_api_key=None,
        cv_api_key=None,
        embedding_api_key="embedding-only-key",
        ollama_base_url="http://localhost:11434",
        _env_file=None,
    )
    components = [
        (LangChainJobExtractor(settings), JobExtraction),
        (LangChainQuestionRouter(settings), RouteDecision),
        (LangChainSQLGenerator(settings), SQLPlan),
        (LangChainJobReranker(settings), RerankOutput),
        (LangChainCVExtractor(settings), CVExtraction),
    ]

    assert [call[0] for call in chat_models] == ["ollama"] * len(components)
    assert [component._model.schema for component, _ in components] == [
        schema for _, schema in components
    ]
    assert LOCAL_EXTRACTION_GUIDANCE in components[0][0]._prompt.messages[0].prompt.template
    local_markers = (
        "Java services using MySQL",
        "Find Python jobs and count",
        "For a count of all jobs",
        "Include every supplied job_id",
        "Built APIs with Python",
    )
    for (component, _), marker in zip(components, local_markers, strict=True):
        assert marker in component._prompt.messages[0].prompt.template


def test_gemini_extractor_keeps_original_prompt(chat_models):
    settings = Settings(
        llm_api_key="gemini-key", ollama_base_url="http://localhost:11434", _env_file=None
    )
    extractor = LangChainJobExtractor(settings)

    assert chat_models[0][0] == "gemini"
    assert LOCAL_EXTRACTION_GUIDANCE not in extractor._prompt.messages[0].prompt.template
