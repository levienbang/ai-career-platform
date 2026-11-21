"""One-request structured-output smoke check; never run by the test suite.

The admin page has the same check as a button (POST /admin/check-llm).
"""

import argparse
import sys
import time

from app.config import Settings
from app.llm import check_structured_output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("json_mode", "function_calling"))
    args = parser.parse_args()
    started = time.monotonic()
    try:
        settings = Settings(
            **({"deepseek_structured_output_method": args.method} if args.method else {})
        )
        model, result = check_structured_output(settings)
        print(f"model={model} method={settings.deepseek_structured_output_method}")
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
