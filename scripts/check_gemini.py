import sys
from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.config import Settings


class GeminiSmokeResult(BaseModel):
    status: Literal["ok"]
    message: str = Field(min_length=1)


def main() -> int:
    settings = Settings()
    api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else ""
    model_name = sys.argv[1] if len(sys.argv) > 1 else settings.llm_model
    if not model_name or not api_key:
        print("Missing LLM_MODEL or LLM_API_KEY in .env", file=sys.stderr)
        return 2

    model = ChatGoogleGenerativeAI(
        model=model_name,
        api_key=api_key,
        temperature=0,
        timeout=settings.llm_timeout_seconds,
        max_retries=0,
    ).with_structured_output(GeminiSmokeResult, method="json_schema")

    try:
        result = model.invoke(
            "Return status 'ok' and a short message confirming the Gemini API is reachable."
        )
    except Exception as error:
        print(f"Gemini API failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    validated = GeminiSmokeResult.model_validate(result)
    print(f"model={model_name} result={validated.model_dump_json()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
