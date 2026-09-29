from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client

from app.config import Settings


def build_agent_callbacks(settings: Settings) -> list[BaseCallbackHandler]:
    key = settings.langsmith_api_key.get_secret_value() if settings.langsmith_api_key else ""
    if not settings.langsmith_tracing or not key:
        return []
    try:
        client = Client(api_key=key)
        return [LangChainTracer(project_name=settings.langsmith_project, client=client)]
    except Exception:
        return []
