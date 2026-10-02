# Demo (dữ liệu thật + Streamlit)

## Chuẩn bị

API phải chạy với dữ liệu đã import và index (xem README, mục "Real jobs from Crawl
and CV matching" và "Resumable indexing and skill learning").

```bash
docker compose up -d                       # PostgreSQL, Qdrant, API (cổng 8000)
cd demo && uv run streamlit run app.py     # giao diện chat (cổng 8501)
```

Mở <http://localhost:8501>. Swagger vẫn ở <http://localhost:8000/docs>.

## Kịch bản 3 phút

1. **Thống kê (SQL):** gõ *"Có bao nhiêu job software ở Hồ Chí Minh?"* → route `sql`,
   bảng kết quả và câu SQL do LLM sinh (chạy bằng tài khoản chỉ đọc).
2. **Tìm job theo nghĩa:** gõ *"Tìm job AI engineer cho người biết PyTorch"* → route
   `search`, danh sách job xếp hạng hybrid + rerank.
3. **Hỏi tiếp có ngữ cảnh:** gõ *"còn ở Hà Nội thì sao?"* → 5 lượt chat gần nhất được
   ghép vào câu hỏi (xem mục "Query đã gửi").
4. **CV:** kéo một CV PDF vào khung chat → `/cv/match` trả skill nhận diện được và
   top job với điểm final / semantic / skill.
5. **Skill gap:** bấm "Phân tích skill gap" → `/cv/upload` liệt kê skill bắt buộc còn
   thiếu, xếp theo tỉ lệ job yêu cầu.

Mỗi câu hỏi agent tốn 2–3 lượt DeepSeek; mỗi CV tốn 1 lượt DeepSeek và 1 request
embedding. Chỉ dùng CV của chính mình hoặc CV giả.
