"""Wire the career agent to real tools built from application settings."""

from sqlalchemy.orm import Session

from app.config import Settings
from app.graph.builder import CareerAgent
from app.graph.router import QuestionRouter
from app.retrieval.embeddings import build_embedding_provider
from app.retrieval.factory import build_reranked_search_service
from app.retrieval.qdrant import build_qdrant_client
from app.retrieval.reranker import build_reranker
from app.tools.search_tool import RerankedJobSearchTool, SearchToolResult
from app.tools.sql_tool import SQLToolResult, build_sql_tool
from app.tracing import build_agent_callbacks


class LazySQLTool:
    """Build the SQL tool only when the graph routes to it."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def invoke(self, question: str, *, job_ids: list[int] | None = None) -> SQLToolResult:
        return build_sql_tool(self.settings).invoke(question, job_ids=job_ids)


class LazyRerankedSearchTool:
    """Build the search pipeline only when the graph routes to it."""

    def __init__(self, db: Session, settings: Settings) -> None:
        self.db = db
        self.settings = settings

    def invoke(self, query: str, limit: int = 5) -> SearchToolResult:
        embedder = build_embedding_provider(self.settings)
        client = build_qdrant_client(self.settings.qdrant_url, self.settings.qdrant_timeout_seconds)
        try:
            reranker = build_reranker(self.settings)
            backend = build_reranked_search_service(
                self.db, embedder, client, reranker, self.settings
            )
            return RerankedJobSearchTool(backend).invoke(query, limit)
        finally:
            client.close()


def build_career_agent(db: Session, settings: Settings, router: QuestionRouter) -> CareerAgent:
    return CareerAgent(
        router,
        LazySQLTool(settings),
        LazyRerankedSearchTool(db, settings),
        search_limit=settings.agent_search_limit,
        tool_max_retries=settings.agent_tool_max_retries,
        callbacks=build_agent_callbacks(settings),
    )
