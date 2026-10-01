import json
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.ingestion.extractor import LangChainJobExtractor
from app.schemas.ingestion import JobExtraction, RawJobRecord


def records(count):
    return [RawJobRecord(title=f"Job {i}", description=f"Python work {i}") for i in range(count)]


def item(index, **overrides):
    return {"record_index": index, "title": "Model title", "description": "Python", **overrides}


@pytest.fixture
def extractor_factory(monkeypatch):
    def factory(response, *, retries=0, rpm=0, clock=None, sleep=None):
        calls = []

        def invoke(messages):
            calls.append(messages)
            if callable(response):
                return response(messages)
            return response

        model = SimpleNamespace(invoke=invoke)
        monkeypatch.setattr(
            "app.ingestion.extractor.build_structured_chat_model",
            lambda *_args, **_kwargs: SimpleNamespace(provider="deepseek", runnable=model),
        )
        extractor = LangChainJobExtractor(
            Settings(
                deepseek_api_key="fake",
                llm_max_retries=retries,
                llm_requests_per_minute=rpm,
                _env_file=None,
            ),
            clock=clock,
            sleep=sleep,
        )
        return extractor, calls

    return factory


def test_batch_missing_duplicate_and_out_of_range_indices(extractor_factory):
    extractor, calls = extractor_factory(
        {
            "items": [
                item(1),
                item(2),
                item(2),
                item(7),
                item(0),
                {"title": "Missing index", "description": "Python"},
                item(4),
            ]
        }
    )
    result = extractor.extract_many(records(4))
    assert [r.title if r else None for r in result] == ["Job 0", None, None, "Job 3"]
    assert len(calls) == 1
    assert extractor.extract_many([]) == []
    assert len(calls) == 1


def test_retry_split_and_terminal_failure(extractor_factory):
    def response(messages):
        content = messages.messages[-1].content
        if content.count("<job_record index=") > 1 or "work 2" in content:
            raise ValueError("invalid JSON")
        return {"items": [item(1)]}

    extractor, calls = extractor_factory(response, retries=1)
    result = extractor.extract_many(records(3))
    assert [r.title if r else None for r in result] == ["Job 0", "Job 1", None]
    assert len(calls) == 8  # two failed groups, two good singletons, one failed singleton
    assert extractor.llm_calls == 8


@pytest.mark.parametrize("single", [True, False])
def test_prompt_projection_and_source_guard(extractor_factory, single):
    extracted = item(
        1,
        company="Invented",
        location="Wrong",
        employment_type="Wrong",
        source_url="https://wrong.example/job",
    )
    response = (
        {k: v for k, v in extracted.items() if k != "record_index"}
        if single
        else {"items": [extracted]}
    )
    extractor, calls = extractor_factory(response)
    source = RawJobRecord(
        title="Source title",
        company=None,
        location="Hà Nội",
        employment_type="Toàn thời gian",
        source_url="https://real.example/job",
        description="Python source",
        technical_skills="['Python']",
        salary="PRIVATE_SALARY",
        benefits="PRIVATE_BENEFITS",
        category="PRIVATE_CATEGORY",
        required_skills=["SHOULD_NOT_SEND"],
    )
    result = extractor.extract(source) if single else extractor.extract_many([source])[0]
    assert result.title == source.title
    assert result.company is None
    assert result.location == source.location
    assert result.employment_type == source.employment_type
    assert result.source_url == source.source_url
    prompt = calls[0].messages[-1].content
    assert "Python" in prompt and "technical_skills" in prompt
    assert all(
        token not in prompt
        for token in [
            "PRIVATE_SALARY",
            "PRIVATE_BENEFITS",
            "PRIVATE_CATEGORY",
            "SHOULD_NOT_SEND",
        ]
    )
    if not single:
        assert '<job_record index="1">' in prompt
    source = source.model_copy(update={"location": None})
    result = extractor.extract(source) if single else extractor.extract_many([source])[0]
    assert result.location == "Wrong"


@pytest.mark.parametrize("rpm", [0, 30])
def test_throttle_with_fake_clock(extractor_factory, rpm):
    now = [0.0]
    sleeps = []
    starts = []

    def sleep(duration):
        sleeps.append(duration)
        now[0] += duration

    def response(_messages):
        starts.append(now[0])
        return {"items": [item(1)]}

    extractor, _calls = extractor_factory(response, rpm=rpm, clock=lambda: now[0], sleep=sleep)
    for _ in range(3):
        extractor.extract_many(records(1))
    assert starts == ([0, 2, 4] if rpm else [0, 0, 0])
    assert sleeps == ([2, 2] if rpm else [])


def test_invalid_json_retries(extractor_factory):
    sequence = iter(["{broken", json.dumps({"items": [item(1)]})])
    extractor, calls = extractor_factory(lambda _: next(sequence), retries=1)
    assert isinstance(extractor.extract_many(records(1))[0], JobExtraction)
    assert len(calls) == 2


def test_provider_parsing_error_recovers_good_indices(extractor_factory):
    from langchain_core.messages import AIMessage

    response = {
        "raw": AIMessage(
            content=json.dumps(
                {
                    "items": [
                        item(1),
                        {"title": "Missing index", "description": "Python"},
                        item(0),
                        item(3),
                    ]
                }
            )
        ),
        "parsed": None,
        "parsing_error": ValueError("record_index required"),
    }
    extractor, calls = extractor_factory(response)
    result = extractor.extract_many(records(3))
    assert [r.title if r else None for r in result] == ["Job 0", None, "Job 2"]
    assert len(calls) == 1
