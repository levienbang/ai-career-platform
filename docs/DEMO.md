# Demo 2–3 phút (Swagger)

## Chuẩn bị

```bash
docker compose up --build -d
docker compose exec api python -m scripts.seed
docker compose exec api python -m scripts.index_jobs
```

Mở <http://localhost:8000/docs>. Các endpoint dùng embedding hoặc LLM thật cần
API key riêng; buổi demo không có key dùng keyword search, `/ready`, và kết quả
evaluation deterministic. Chỉ dùng dữ liệu job mẫu và CV PDF giả.

## Kịch bản

1. **0:00–0:30 — Hệ thống và dữ liệu:** mở `GET /ready`, cho thấy API, PostgreSQL,
   Qdrant sẵn sàng. Mở `GET /jobs` và một job mẫu; giải thích PostgreSQL lưu job
   và taxonomy, Qdrant lưu vector sau bước index.
2. **0:30–1:10 — Ingestion và tìm kiếm:** mở `POST /search/keyword`, tìm
   `computer vision internship`. Chỉ ra ID, kỹ năng và nguồn job trong kết quả.
   Nếu có key, có thể so sánh dense/hybrid/reranked bằng cùng câu hỏi.
3. **1:10–1:50 — Agent và an toàn:** trình bày `POST /agent/query` và
   `POST /analytics/query` trong Swagger. Nếu không có key, dùng kết quả
   `evaluation/milestone7_results.json` để giải thích route giả và SQL đã được
   xác thực trên PostgreSQL. Minh họa test từ chối `DELETE FROM jobs`.
4. **1:50–2:30 — CV và evaluation:** giải thích `POST /cv/upload` nhận PDF giả,
   evidence quote, alias normalization và gap score. Không gọi endpoint live khi
   không có key cho extractor; chỉ rõ kết quả test dùng fake extractor. Trình bày
   39 case, bốn phương pháp retrieval, latency và failure analysis trong README.
5. **2:30–3:00 — Giới hạn:** nêu rõ chưa đo chất lượng LLM live hay remote trace
   nếu thiếu key, và Qdrant cần index sau khi import.

Không ghi hình CV cá nhân, token hoặc mật khẩu. Video demo là deliverable thủ
công; tài liệu này là kịch bản thực hiện.
