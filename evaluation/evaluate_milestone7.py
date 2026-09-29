"""Deterministic, no-paid-API evaluation against the prelabelled seed corpus."""

import json
import re
from dataclasses import replace
from pathlib import Path
from statistics import mean
from time import perf_counter

from qdrant_client import QdrantClient

from app.config import Settings
from app.db.repositories import JobRepository
from app.db.session import SessionLocal
from app.graph.builder import CareerAgent
from app.graph.router import RouteDecision
from app.retrieval.dense import DenseSearchService
from app.retrieval.hybrid import HybridSearchService
from app.retrieval.keyword import KeywordSearchService
from app.retrieval.qdrant import QdrantJobIndex
from app.retrieval.reranker import RerankedSearchService
from app.schemas.cv import CVExtraction, CVSkillClaim
from app.services.cv_pdf import CVPDFError, validate_cv_pdf
from app.services.skill_gap import TargetJobError, analyze_skill_gap
from app.tools.search_tool import RerankedJobSearchTool, SearchToolResult
from app.tools.sql_tool import (
    PostgreSQLReadOnlyExecutor,
    SQLSafetyValidator,
    SQLToolResult,
    SQLValidationError,
    _readonly_engine,
)
from evaluation.evaluate_agent import percentile, required_tools_succeeded
from evaluation.evaluate_retrieval import mrr_at_k, recall_at_k

ROOT = Path(__file__).parent
CASES_FILE = ROOT / "milestone7_cases.json"
RETRIEVAL_CASES_FILE = ROOT / "retrieval_cases.json"
AGENT_CASES_FILE = ROOT / "agent_cases.json"
RESULTS_FILE = ROOT / "milestone7_results.json"


class FixtureEmbeddingProvider:
    dimensions = 4

    @staticmethod
    def _vector(value: str) -> list[float]:
        lowered = value.casefold()
        groups = (
            ("vision", "image", "video", "computer"),
            ("backend", "api", "fastapi", "service"),
            ("docker", "deploy", "container", "platform"),
            ("postgres", "data", "database", "sql"),
        )
        vector = [float(sum(word in lowered for word in group)) for group in groups]
        return vector if any(vector) else [0.5] * 4

    def embed_documents(self, texts: list[str], titles: list[str]) -> list[list[float]]:
        return [self._vector(f"{title} {text}") for title, text in zip(titles, texts, strict=True)]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FixtureReranker:
    def rerank(self, query, candidates):
        query_words = set(re.findall(r"\w+", query.casefold()))
        return [
            replace(
                hit,
                score=float(
                    len(query_words & set(re.findall(r"\w+", hit.document_text.casefold())))
                ),
            )
            for hit in sorted(
                candidates,
                key=lambda item: (
                    -len(query_words & set(re.findall(r"\w+", item.document_text.casefold()))),
                    item.job_id,
                ),
            )
        ]


class FixtureRouter:
    def decide(self, question: str) -> RouteDecision:
        text = question.casefold()
        wants_search = any(term in text for term in ("tìm", "công việc nào", "job nào", "find"))
        wants_sql = any(
            term in text
            for term in ("bao nhiêu", "đếm", "thống kê", "phổ biến", "tổng số", "thường xuyên")
        )
        route = "both" if wants_search and wants_sql else "search" if wants_search else "sql"
        return RouteDecision(route=route, sql_scope="all")


class FixtureSQLTool:
    def invoke(self, question: str, *, job_ids: list[int] | None = None) -> SQLToolResult:
        del question, job_ids
        return SQLToolResult(
            sql="SELECT COUNT(*) FROM jobs", columns=["count"], rows=[[1]], row_count=1
        )


class EmptySearchTool:
    def invoke(self, query: str, limit: int = 5) -> SearchToolResult:
        del limit
        return SearchToolResult(query=query, jobs=[])


def _load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _retrieval(session, cases: list[dict]) -> tuple[dict, list[float], int]:
    client = QdrantClient(":memory:")
    embedder = FixtureEmbeddingProvider()
    index = QdrantJobIndex(client, embedder, "evaluation_jobs")
    indexed = index.index_all(session).indexed
    keyword = KeywordSearchService(session)
    dense = DenseSearchService(client, embedder, "evaluation_jobs")
    hybrid = HybridSearchService(keyword, dense, candidate_limit=20)
    reranked = RerankedSearchService(hybrid, FixtureReranker(), candidate_limit=20)
    methods = {"keyword": keyword, "dense": dense, "hybrid": hybrid, "reranked": reranked}
    scores: dict[str, dict[str, list[float]]] = {
        name: {"recall_at_5": [], "mrr_at_10": [], "latency_ms": []} for name in methods
    }
    all_latencies: list[float] = []
    repository = JobRepository(session)
    for case in cases:
        relevant = {job.id for job in repository.get_by_source_urls(case["relevant_source_urls"])}
        if len(relevant) != len(case["relevant_source_urls"]):
            raise RuntimeError("Retrieval ground truth jobs are missing; seed the database")
        for name, method in methods.items():
            started = perf_counter()
            ids = [hit.job_id for hit in method.search(case["query"], limit=10)]
            latency = (perf_counter() - started) * 1000
            scores[name]["recall_at_5"].append(recall_at_k(ids, relevant, 5))
            scores[name]["mrr_at_10"].append(mrr_at_k(ids, relevant, 10))
            scores[name]["latency_ms"].append(latency)
            all_latencies.append(latency)
    metrics = {
        name: {
            "recall_at_5": mean(values["recall_at_5"]),
            "mrr_at_10": mean(values["mrr_at_10"]),
            "p50_latency_ms": percentile(values["latency_ms"], 0.5),
            "p95_latency_ms": percentile(values["latency_ms"], 0.95),
        }
        for name, values in scores.items()
    }
    return metrics, all_latencies, indexed


def _routing(session, cases: list[dict]) -> tuple[dict, list[float], int]:
    client = QdrantClient(":memory:")
    embedder = FixtureEmbeddingProvider()
    QdrantJobIndex(client, embedder, "routing_jobs").index_all(session)
    hybrid = HybridSearchService(
        KeywordSearchService(session), DenseSearchService(client, embedder, "routing_jobs")
    )
    tool = RerankedJobSearchTool(RerankedSearchService(hybrid, FixtureReranker()))
    agent = CareerAgent(FixtureRouter(), FixtureSQLTool(), tool, tool_max_retries=0)
    correct = 0
    completed = 0
    failures = 0
    latencies = []
    for case in cases:
        try:
            result = agent.invoke(case["question"])
            route_correct = result.route == case["expected_route"]
            correct += int(route_correct)
            completed += int(
                route_correct
                and required_tools_succeeded(result, case["expected_route"])
                and result.answer.startswith(("SQL result:", "Search result:"))
            )
            failures += int(bool(result.errors))
            latencies.append(result.latency_ms)
        except Exception:
            failures += 1
            latencies.append(0.0)
    return (
        {
            "intent_accuracy": correct / len(cases),
            "tool_selection_accuracy": correct / len(cases),
            "task_success_rate": completed / len(cases),
            "p50_latency_ms": percentile(latencies, 0.5),
            "p95_latency_ms": percentile(latencies, 0.95),
        },
        latencies,
        failures,
    )


def _sql(session, cases: list[dict], settings: Settings) -> tuple[dict, list[float], int]:
    database_url = settings.readonly_database_url or settings.database_url
    connection_mode = (
        "read-only-role" if settings.readonly_database_url else "app-role-read-only-tx"
    )
    validator = SQLSafetyValidator(max_rows=settings.sql_max_rows)
    executor = PostgreSQLReadOnlyExecutor(
        _readonly_engine(database_url),
        statement_timeout_ms=settings.sql_statement_timeout_ms,
    )
    executable = exact = errors = 0
    latencies = []
    source_urls = [
        job["source_url"]
        for job in json.loads((ROOT.parent / "data" / "seed_data.json").read_text())["jobs"]
        if job["source_url"].startswith("https://example.com/eval/")
    ]
    job_ids = [job.id for job in JobRepository(session).get_by_source_urls(source_urls)]
    if len(job_ids) != len(source_urls):
        raise RuntimeError("SQL ground truth jobs are missing; seed the database")
    case_results = []
    for case in cases:
        started = perf_counter()
        try:
            result = executor.execute(validator.validate(case["sql"], job_ids=job_ids))
            executable += 1
            correct = result.rows == [[case["expected_count"]]]
            exact += int(correct)
            case_results.append({"id": case["id"], "actual_rows": result.rows, "correct": correct})
        except Exception as error:
            errors += 1
            case_results.append({"id": case["id"], "error": type(error).__name__})
        latencies.append((perf_counter() - started) * 1000)
    return (
        {
            "connection_mode": connection_mode,
            "executable_sql_rate": executable / len(cases),
            "exact_result_accuracy": exact / len(cases),
            "cases": case_results,
        },
        latencies,
        errors,
    )


def _cv(session, cases: list[dict]) -> tuple[dict, list[float], int]:
    correct = errors = 0
    latencies = []
    repository = JobRepository(session)
    for case in cases:
        started = perf_counter()
        try:
            jobs = repository.get_by_source_urls([case["target_source_url"]])
            if len(jobs) != 1:
                raise RuntimeError("CV target job is missing")
            profile = CVExtraction(
                skills=[CVSkillClaim(name=name, evidence=name) for name in case["claims"]]
            )
            report = analyze_skill_gap(session, profile, case["text"], [jobs[0].id])
            gap = report.jobs[0]
            correct += int(
                gap.matched_skills == case["expected_matched"]
                and gap.missing_required_skills == case["expected_missing_required"]
                and gap.fit_score == case["expected_fit"]
            )
        except Exception:
            errors += 1
        latencies.append((perf_counter() - started) * 1000)
    return {"correctness": correct / len(cases)}, latencies, errors


def _errors(session, cases: list[dict]) -> tuple[dict, list[float], int]:
    accepted = failures = 0
    unsafe_total = unsafe_rejected = 0
    latencies = []
    case_results = []
    for case in cases:
        started = perf_counter()
        try:
            kind = case["kind"]
            if kind in {"unsafe_sql", "stacked_sql"}:
                unsafe_total += 1
                try:
                    SQLSafetyValidator().validate(case["input"])
                except SQLValidationError:
                    actual = "SQLValidationError"
                    unsafe_rejected += 1
                else:
                    actual = "accepted"
            elif kind == "missing_job":
                try:
                    analyze_skill_gap(session, CVExtraction(), "", [case["input"]])
                except TargetJobError:
                    actual = "TargetJobError"
                else:
                    actual = "accepted"
            elif kind == "invalid_pdf":
                try:
                    validate_cv_pdf(
                        case["input"].encode(),
                        filename="cv.pdf",
                        content_type="application/pdf",
                        max_bytes=100,
                    )
                except CVPDFError:
                    actual = "CVPDFError"
                else:
                    actual = "accepted"
            else:
                actual = (
                    CareerAgent(
                        FixtureRouter(), FixtureSQLTool(), EmptySearchTool(), tool_max_retries=0
                    )
                    .invoke(case["input"])
                    .answer
                )
            accepted += int(actual == case["expected"])
            case_results.append({"id": case["id"], "correct": actual == case["expected"]})
        except Exception as error:
            failures += 1
            case_results.append({"id": case["id"], "error": type(error).__name__})
        latencies.append((perf_counter() - started) * 1000)
    return (
        {
            "correct_rejection_or_fallback_rate": accepted / len(cases),
            "unsafe_query_rejection_rate": unsafe_rejected / unsafe_total,
            "cases": case_results,
        },
        latencies,
        failures,
    )


def evaluate() -> dict[str, object]:
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    retrieval_cases = _load(RETRIEVAL_CASES_FILE)
    agent_cases = _load(AGENT_CASES_FILE)
    counts = {
        "retrieval": len(retrieval_cases),
        "routing": len(agent_cases),
        "sql": len(cases["sql"]),
        "cv": len(cases["cv"]),
        "error": len(cases["error"]),
    }
    latencies: list[float] = []
    errors = 0
    with SessionLocal() as session:
        retrieval, times, indexed = _retrieval(session, retrieval_cases)
        latencies.extend(times)
        routing, times, failures = _routing(session, agent_cases)
        latencies.extend(times)
        errors += failures
        sql, times, failures = _sql(session, cases["sql"], Settings())
        latencies.extend(times)
        errors += failures
        cv, times, failures = _cv(session, cases["cv"])
        latencies.extend(times)
        errors += failures
        error_cases, times, failures = _errors(session, cases["error"])
        latencies.extend(times)
        errors += failures
    sql["unsafe_query_rejection_rate"] = error_cases["unsafe_query_rejection_rate"]
    metrics = {
        "mode": "deterministic-fakes-no-llm",
        "ground_truth_file": str(CASES_FILE.name),
        "case_counts": counts,
        "total_cases": sum(counts.values()),
        "indexed_jobs": indexed,
        "retrieval": retrieval,
        "routing": routing,
        "sql": sql,
        "cv": cv,
        "error_cases": error_cases,
        "system": {
            "p50_latency_ms": percentile(latencies, 0.5),
            "p95_latency_ms": percentile(latencies, 0.95),
            "error_rate": errors / len(latencies),
            "measured_operations": len(latencies),
        },
    }
    RESULTS_FILE.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
