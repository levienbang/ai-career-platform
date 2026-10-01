# Chat provider after Milestone 11
- All structured chat uses `app/llm.py::build_structured_chat_model` with `ChatDeepSeek` (`langchain-deepseek`); one `DEEPSEEK_API_KEY`, default `DEEPSEEK_MODEL=deepseek-v4-flash`, configurable HTTP(S) base URL.
- `LLM_MODEL`, `CV_MODEL`, `AGENT_MODEL`, `RERANKER_MODEL` are optional DeepSeek overrides; empty values use the default. Retained model variables in an older user .env still override defaults; do not migrate or expose the user's secrets.
- Only function_calling and json_mode are accepted. JSON mode adds the actual Pydantic schema to system prompts via template-safe `structured_output_guidance`; job batch uses JobBatchExtraction, single uses JobExtraction.
- Batch include_raw recovery, guards, dedup, fast path, throttle and retry behavior from M10 remain intact.
- Google embedding implementation and EMBEDDING_API_KEY remain independent; chat migration alone needs no re-indexing.
- `scripts/check_deepseek.py` performs one paid request with max_retries=0; do not invoke during pytest. Supports --method json_mode or function_calling.
- Live provider/real jobs/private CV checks were deferred to Claude as explicitly assigned by docs/CURRENT_MILESTONE.md. Offline tests do not establish live availability or quality.
