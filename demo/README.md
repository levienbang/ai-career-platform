# Career Chat demo (Streamlit)

Giao diện chat mỏng gọi API của repo này. Không chứa logic AI; mọi xử lý nằm ở API.

| Bạn làm | Demo gọi | Hiển thị |
|---|---|---|
| Gõ câu hỏi | `POST /agent/query` | Route agent chọn, bảng kết quả SQL + câu SQL, danh sách job tìm được |
| Kéo CV (PDF) vào khung chat | `POST /cv/match` | Skill nhận diện, bảng job gợi ý với điểm final / semantic / skill, skill khớp/thiếu |
| Bấm "Phân tích skill gap" sau khi gửi CV | `POST /cv/upload` với top N job | Skill bắt buộc còn thiếu, xếp theo tỉ lệ job yêu cầu |

## Chạy

API LLM phải đang chạy (mặc định `http://localhost:8000`, đã index Qdrant).

```bash
cd demo
uv run streamlit run app.py
# hoặc đổi API:
CAREER_API_URL=http://localhost:8000 uv run streamlit run app.py
```

Mở <http://localhost:8501>.

CV chỉ được gửi tới API cấu hình ở sidebar và giữ trong phiên trình duyệt; không lưu
xuống đĩa.
