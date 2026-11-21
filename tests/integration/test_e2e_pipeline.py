import os
import re
from dataclasses import replace
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient
from sqlalchemy import select

from app.config import Settings
from app.db.models import Company, Job
from app.db.repositories import JobRepository
from app.db.session import SessionLocal
from app.graph.builder import CareerAgent
from app.graph.router import RouteDecision
from app.ingestion.pipeline import JobIngestionPipeline
from app.retrieval.dense import DenseSearchService
from app.retrieval.hybrid import HybridSearchService
from app.retrieval.keyword import KeywordSearchService
from app.retrieval.qdrant import QdrantJobIndex
from app.retrieval.reranker import RerankedSearchService
from app.tools.search_tool import RerankedJobSearchTool
from tests.conftest import FakeEmbeddingProvider, FakeJobExtractor


@pytest.mark.skipif(os.getenv("RUN_DOCKER_E2E") != "1", reason="Requires PostgreSQL and Qdrant")
def test_import_to_agent_grounded_answer_on_real_services():
    settings = Settings()
    client = QdrantClient(url=settings.qdrant_url, timeout=settings.qdrant_timeout_seconds)
    suffix = uuid4().hex[:12]
    collection = f"e2e_{suffix}"
    source_url = f"https://example.com/e2e/{suffix}"
    company_name = f"E2E Company {suffix}"
    job_id = None

    class FakeRouter:
        def decide(self, question):
            return RouteDecision(route="search", sql_scope="all")

    class UnusedSQL:
        def invoke(self, question, *, job_ids=None):
            raise AssertionError("SQL route should not run")

    class TokenReranker:
        def rerank(self, query, candidates):
            words = set(re.findall(r"\w+", query.casefold()))
            return [
                replace(
                    hit,
                    score=float(len(words & set(re.findall(r"\w+", hit.document_text.casefold())))),
                )
                for hit in sorted(
                    candidates,
                    key=lambda item: (
                        -len(words & set(re.findall(r"\w+", item.document_text.casefold()))),
                        item.job_id,
                    ),
                )
            ]

    try:
        with SessionLocal() as session:
            result = JobIngestionPipeline(session, FakeJobExtractor()).import_records(
                [
                    {
                        "title": "E2E Vision Intern",
                        "company": company_name,
                        "description": f"Analyze image and video with Python. Fixture {suffix}.",
                        "source_url": source_url,
                        "required_skills": ["Python", "Computer Vision"],
                    }
                ]
            )
            assert result.inserted == 1
            job = JobRepository(session).get_by_source_url(source_url)
            assert job is not None
            job_id = job.id
            embedder = FakeEmbeddingProvider()
            QdrantJobIndex(client, embedder, collection).index_job(session, job.id)
            search = RerankedJobSearchTool(
                RerankedSearchService(
                    HybridSearchService(
                        KeywordSearchService(session),
                        DenseSearchService(client, embedder, collection),
                    ),
                    TokenReranker(),
                )
            )
            answer = CareerAgent(FakeRouter(), UnusedSQL(), search, search_limit=20).invoke(
                f"Find image internship {suffix}"
            )
            assert answer.route == "search"
            assert any(item.job_id == job.id for item in answer.search_result.jobs)
            assert "E2E Vision Intern" in answer.answer
            assert answer.errors == []
    finally:
        if client.collection_exists(collection):
            client.delete_collection(collection)
        with SessionLocal.begin() as cleanup:
            if job_id is not None:
                job = cleanup.get(Job, job_id)
                if job is not None:
                    cleanup.delete(job)
            company = cleanup.scalar(select(Company).where(Company.name == company_name))
            if company is not None:
                cleanup.delete(company)
