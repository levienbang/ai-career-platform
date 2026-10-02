# Milestone 12 — Chất lượng skill, log lỗi và build nhanh

Ngày thực hiện: 01/10/2026. Phạm vi theo `docs/CURRENT_MILESTONE.md`.

## Baseline trước khi sửa

- Đã đọc `AGENTS.md`, `docs/PROJECT_SPEC.md`, `docs/CURRENT_MILESTONE.md`.
- Serena không được expose trong phiên này; đọc memory conventions và source có mục tiêu.
- Các thay đổi M9–M11 có sẵn được giữ nguyên; không sửa `../Crawl` hay `docs/PROJECT_SPEC.md`.
- `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .` ban đầu
  không chạy được vì sandbox không cho ghi cache `~/.cache/uv`.
- Chạy lại bằng `UV_CACHE_DIR=/tmp/m12-uv-cache uv run --offline ...`:
  **213 passed, 1 skipped**; Ruff check pass; format check **124 files already formatted**.
- Test dùng fake model/embedding/reranker, SQLite và Qdrant in-memory; không gọi API trả phí.

## Files tạo mới

- `data/skill_aliases.json`: 55 nhóm canonical và alias được kiểm soát, gồm tất cả
  nhóm bắt buộc và `C`, `C++`, `C#`, `R` riêng biệt.
- `app/services/skill_taxonomy.py`: nạp/kiểm tra alias, lập kế hoạch gộp,
  chuyển liên kết và alias, seed taxonomy; transaction do caller quản lý.
- `scripts/merge_skills.py`: CLI mặc định dry-run, ghi DB chỉ khi `--apply`,
  in nhóm gộp/số job_skills bị ảnh hưởng và nhắc re-index.
- `tests/test_milestone12_skills.py`, `tests/test_milestone12_errors.py`: kiểm thử offline.
- `docs/MILESTONE12_REPORT.md`: báo cáo này.

## Files cập nhật

- `app/ingestion/normalizer.py`: `skill_match_key` từ NFKC/casefold, bỏ khoảng
  trắng và `. - _ /`, giữ `+ #`; resolver và tạo extracted skill tái sử dụng khóa đó.
  Evidence vẫn dùng `skill_lookup_key` và biên từ trên text nguồn; một cách viết
  resolve được vẫn phải có bằng chứng literal hoặc canonical/alias đã biết.
  Chỉ tên extracted mới chịu giới hạn 5 từ/50 ký tự.
- `app/ingestion/structured.py`: bảo toàn dấu phẩy trong ngoặc khi đọc chuỗi;
  `Cụm chữ (A, B, ...)` được thay bằng từng phần tử trong ngoặc.
- `scripts/seed.py`: nạp taxonomy gốc rồi alias mới, seed curated và gộp extracted
  trùng trong cùng transaction; cả default và `--taxonomy-only` idempotent.
- `app/services/skill_gap.py`: CV sử dụng cùng resolver và kiểm tra literal evidence.
- `app/ingestion/extractor.py`: warning cho mỗi attempt lỗi và lỗi parsing đã recover;
  component, số records, attempt, exception class và thông điệp tối đa 300 ký tự.
  Che `sk-...`, redaction source fields và request/response bodies; lỗi parse/validate
  dùng diagnostic tổng quát để không log output model. Giữ diagnostic message riêng
  từ provider error body để nhận ra lỗi như `Model Not Exist`. Tổng kết một lần
  cho số record cuối cùng thất bại sau retry/split. ImportResult không đổi.
- `app/main.py`: cấu hình logging WARNING ra stderr để Docker thu nhận.
- `app/retrieval/dense.py`: kiểm tra `points_count == 0` trước khi embed query;
  thông báo `Qdrant collection '<name>' is empty; index jobs first`.
- `app/api/search.py`: SearchIndexNotReadyError trả 503 cho semantic/hybrid/reranked,
  thống nhất với CV; collection cấu hình sai vẫn là 409.
- `Dockerfile`: cài dependencies từ metadata trước khi copy source rồi cài lại
  project bằng `--no-deps`; giữ ARG INSTALL_TARGET và Compose command migration/uvicorn.
- `README.md`: workflow dry-run/apply/seed/re-index và hành vi mới.
- `tests/test_seed.py`, `tests/test_search_api.py`: cập nhật kỳ vọng taxonomy mở rộng
  và HTTP 503. Test M9–M11 không sửa.

Không thêm dependency, environment variable hoặc migration; không đổi provider,
job schema, công thức CV scoring hay shape API response. Không cần sửa `.env.example`.

## Quyết định kỹ thuật

- Alias đồng nghĩa đến từ dữ liệu curated, không giao taxonomy cho LLM.
- Merge nhóm theo match key/canonical/alias, kể cả alias hiện có để chuyển toàn bộ
  ownership an toàn. Chọn canonical chính xác trong alias file, rồi curated,
  rồi số job_skills lớn nhất, rồi id nhỏ nhất.
- Khi đụng unique `(job_id, skill_id, requirement_type)`, giữ một liên kết,
  ưu tiên liên kết survivor nếu cùng loại; required loại preferred thừa.
  Flush các dòng bị xóa trước khi chuyển skill_id để không vi phạm unique constraint.
  Chuyển aliases trước khi xóa skill, giữ tên cũ làm alias.
- CLI apply và seed dùng một transaction; rollback khôi phục các bước đã thực hiện.
  Dry-run chỉ lập kế hoạch và đọc DB. Apply lại không tạo thay đổi.
- PostgreSQL vẫn là nguồn dữ liệu chuẩn; Qdrant chỉ cập nhật khi index rõ ràng.

## Commands vận hành

```bash
docker compose up --build -d
# Swagger: http://localhost:8000/docs

docker compose exec api python -m scripts.merge_skills
# Chỉ chạy sau khi đã xem dry-run:
docker compose exec api python -m scripts.merge_skills --apply
docker compose exec api python -m scripts.seed --taxonomy-only
docker compose exec api python -m scripts.index_jobs

docker compose logs api
# Dev giữ INSTALL_TARGET=.[dev]:
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build -d
```

Chưa chạy merge/seed/index trên DB thật `career_ai`; bước đó và so sánh CV thật
được dành cho Claude theo yêu cầu milestone. Index thật cần Google embedding key.

## Kiểm chứng cuối cùng

- `UV_CACHE_DIR=/tmp/m12-uv-cache uv run --offline pytest`: **249 passed, 1 skipped**, 1.27s.
- Test tập trung M12: **36 passed** (skill + extraction/index errors), 0.34s.
- `UV_CACHE_DIR=/tmp/m12-uv-cache uv run --offline ruff check .`: **All checks passed!**
- `UV_CACHE_DIR=/tmp/m12-uv-cache uv run --offline ruff format --check .`:
  **129 files already formatted**.
- `git diff --check`: pass.
- `python -m scripts.merge_skills --help`, `python -m scripts.seed --help`: pass,
  không kết nối DB để đọc help.
- Test kiểm tra C/C++/C# độc lập, resolver reuse và evidence boundary, alias ownership,
  seed/merge idempotence, dry-run, transaction rollback, survivor priority/tie,
  chuyển aliases và job links, required thắng preferred, CV matching sklearn,
  sentence filtering/length boundaries và import danh sách ngoặc, warning redaction,
  partial-batch summaries, collection unavailable/rỗng và HTTP 503.
- M9–M11 test pass, không sửa file test của các milestone đó.

## Docker build và cache

Docker CLI ban đầu bị chặn socket trong sandbox; truy cập nâng cao đã được
cho phép, Docker Desktop hoạt động. Đo bằng `/usr/bin/time -p docker build --progress=plain ...`:

| Phép đo | Thời gian thực |
|---|---:|
| Dockerfile trước thay đổi (build dependency đầy đủ) | 219.29s |
| Dockerfile mới, dependency layer chưa có cache | 335.36s |
| Dockerfile mới cập nhật source hoàn thiện, cache dependency | 7.16s |
| Chỉ thêm comment vào `app/ingestion/normalizer.py`, cache dependency | **5.85s** |

Build log lần chỉ đổi một file xác nhận:

```text
[ 4/11] RUN pip install --no-cache-dir "."
CACHED
[ 5/11] COPY app ./app
# layer copy source và install project chạy lại
```

Phép đo một-file dùng `/tmp/m12-docker-context`, bản sao các files COPY trong
Dockerfile; chỉ thêm comment trong bản sao, không sửa source workspace để đo.
Thời gian cold build phụ thuộc mạng/cache registry nên không tuyên bố cold build
nhanh hơn. Cải tiến đo được là dependency layer không chạy lại khi chỉ đổi code.

Commands đã chạy:

```bash
docker build --progress=plain -f /tmp/m12-original.Dockerfile -t career-ai:m12-before .
docker build --progress=plain -t career-ai:m12-after .
docker build --progress=plain -t career-ai:m12-after /tmp/m12-docker-context
docker build --progress=plain -t career-ai:m12-cache-check /tmp/m12-docker-context
```

- Image dev build thành công với `--build-arg 'INSTALL_TARGET=.[dev]'`:
  **273.46s** (dependency layer dev chưa có cache).
- `docker run --rm --network none -v
  /Users/levienbang/Documents/ChatGPT/LLM/tests:/app/tests:ro career-ai:m12-dev pytest`:
  **249 passed, 1 skipped**, 1.46s; Docker tắt mạng hoàn toàn trong lúc chạy tests.
- Image production smoke pass: import `app.main`, kiểm tra public OpenAPI có
  `/cv/match`, load đủ 55 nhóm alias và resolve match key; không gọi DB/model.
- `docker compose -f docker-compose.yml -f docker-compose.dev.yml config --quiet`:
  pass, không in environment secrets.
- Image tags giữ lại: `career-ai:m12-after`, `career-ai:m12-dev`; các tag before/cache-check
  được dùng riêng cho đo build. Không khởi động hay thay thế service production.

Command build dev:

```bash
docker build --progress=plain --build-arg 'INSTALL_TARGET=.[dev]' \
  -t career-ai:m12-dev /tmp/m12-docker-context
```

Build logs/timings nằm trong `/tmp/m12-build-{before,after,current,rebuild,dev}.log`;
đây là artifact kiểm chứng tạm, không commit vào repo.

Một smoke command đầu dùng `route.path` trực tiếp trên `app.routes` thất bại vì
FastAPI mới trả `_IncludedRouter`; đây là lỗi script smoke, import app vẫn thành công.
Đổi sang kiểm tra public OpenAPI schema để xác minh endpoint.

## Giới hạn và khái niệm cần hiểu

- Key so khớp xử lý khác cách viết; alias curated xử lý khác tên/đồng nghĩa.
  Hai việc này không thay thế việc kiểm tra evidence chống bịa.
- CV chỉ nhận kỹ năng đã có trong taxonomy và thị trường; không tự tạo skill mới.
- Merge DB không tự re-index Qdrant. Seed trên DB cũ có thể gộp skill và cũng cần re-index.
- Giới hạn tên lọc câu dài; không tách tự động mọi câu liệt kê ngoài mẫu ngoặc được yêu cầu.
- Test offline chứng minh logic và tích hợp nội bộ; không chứng minh chất lượng LLM live
  hay mức tăng skill_score trên CV riêng của user. Không gọi model/embedding API trong lượt này.
- Một test Docker E2E PostgreSQL/Qdrant được skip theo mặc định vì không bật
  `RUN_DOCKER_E2E=1`; không dùng DB production để chạy suite.
- Không thay đổi schema, nên không có migration mới cần apply/verify cho M12.
