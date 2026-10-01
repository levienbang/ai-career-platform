"""One-request structured-output smoke check; never run by the test suite."""

import argparse
import sys
import time

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.config import Settings
from app.llm import build_structured_chat_model, structured_output_guidance


class SmokeResult(BaseModel):
    ok: bool
    language: str


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("json_mode", "function_calling"))
    args = parser.parse_args()
    started = time.monotonic()
    try:
        settings = Settings(
            **({"deepseek_structured_output_method": args.method} if args.method else {})
        )
        selected = build_structured_chat_model(
            settings,
            SmokeResult,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=0,
        )
        print(f"model={selected.model} method={settings.deepseek_structured_output_method}")
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Return ok=true and language='English'."
                    + structured_output_guidance(settings, SmokeResult),
                ),
                ("human", "Check structured output."),
            ]
        )
        result = SmokeResult.model_validate(selected.runnable.invoke(prompt.invoke({})))
        print(f"elapsed_seconds={time.monotonic() - started:.3f} result={result.model_dump_json()}")
        return 0
    except Exception as error:
        # Provider exception text may contain headers or sensitive request details.
        print(
            f"DeepSeek check failed ({type(error).__name__}); check configuration/connectivity.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
