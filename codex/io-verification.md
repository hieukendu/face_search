# Kiểm thử thay đổi nhập Drive và tải ảnh — 2026-11-10

Nguồn ảnh cố định: [Database](https://drive.google.com/drive/folders/1Fw76RVGWqtaKqCVBnDi3TR3oFqOfHLd_).

## Kết quả đã xác nhận

- Tải hai model từ `globalnebula/insightface-buffalo-l-onnx` trên Hugging Face: `scrfd_10g.onnx` (16.923.827 byte), `arcface_w600k_r50.onnx` (174.383.860 byte).
- Gọi `POST /events` bằng FastAPI TestClient, dùng kết nối Drive và model thật: đọc đủ 38 ảnh, lưu 38 bản gốc, lập chỉ mục 155 khuôn mặt; 7 ảnh không có khuôn mặt, không có lỗi tải/đọc. Cả đợt lưu index/metadata một lần.
- Truy vấn bằng ảnh cắt mặt từ ảnh sự kiện: nhận 31 ảnh. Ảnh tải riêng và từng ảnh trong ZIP đều khớp nội dung gốc. Sau khi nạp lại runtime/index, số kết quả vẫn là 31.
- Nhập lại 38 bản gốc đã tải, dùng model thật và thay phần truyền tải mạng bằng đọc bản local: bỏ qua 31 ảnh đã có embedding, xử lý lại 7 ảnh không có mặt, thêm 0 ảnh/0 vector; index vẫn có 155 vector.
- Kiểm thử UI đầy đủ bằng `python -B tests/ui_drive_smoke.py`: chạy Uvicorn thật và Edge headless, bắt đầu bằng album/index tạm rỗng, bấm nút nhập Drive. Tải đủ **38 ảnh trực tiếp từ Drive**, lập chỉ mục **155 khuôn mặt**, **7 ảnh không có mặt**, **0 lỗi**. Không dùng ảnh tải sẵn hay phản hồi mạng giả lập.
- Sau lượt nhập UI, tìm được **31 ảnh**, tải riêng và ZIP bằng nút trên trang, đối chiếu byte của tất cả ảnh với bản gốc: đạt. Server vẫn trả `/health` trong lúc nhập, yêu cầu nhập đồng thời bị từ chối với HTTP 409; không có lỗi JavaScript. Toàn bộ lượt UI mất **165,3 giây**; server và dữ liệu tạm tự dọn thành công.
- Giao diện đã được kiểm tra và xem ảnh chụp ở chiều rộng 1440 px và 390 px, không tràn ngang.
- Bộ `unittest`: 27 bài, **26 đạt, 1 bỏ qua**. Bao gồm FAISS thật, model thật, 121 kết quả tìm/tải, lỗi Drive/ảnh hỏng, chống trùng, tải ngoài thư mục và dọn tệp tạm. Kiểm thử UI dùng mạng thật được chạy riêng bằng lệnh trên.
- Bài bỏ qua cần bộ 10 selfie `data/queries/smoke_queries/` của bộ kiểm thử cũ; bộ này không có trong checkout.
- `git diff --check` và phân tích cú pháp Python đạt. Đối chiếu với Git xác nhận không thay đổi `face_encoder.py`, `search_engine.py`, `vector_store.py`, ba script pipeline/tuning hoặc `storage/best_config.json`.

## Kết luận kiểm thử và phạm vi

Lỗi DNS của lượt kiểm thử trước không tái hiện: DNS/HTTPS từ cả tiến trình chính và tiến trình con đều hoạt động; lượt nhập Drive từ trình duyệt đã hoàn tất qua server thật. Điểm chưa xác nhận trước đây đã được kiểm tra thành công. Truy vấn bằng ảnh `2X3.png` được kiểm tra với điểm từng kết quả; ngưỡng hiện tại 0,23 cho 18 kết quả, trong đó phần đuôi đạt 0,238–0,329. Đây là chẩn đoán chất lượng, không thay đổi ngưỡng hay pipeline.

Các truy vấn tự đối chiếu kiểm tra chức năng và tính toàn vẹn dữ liệu, không đo độ chính xác nhận dạng trên một bộ dữ liệu có nhãn. Tuning detector, ngưỡng tương đồng 0,23 và pipeline embedding giữ nguyên.

## Chạy lại

```powershell
python -m pip install -r requirements-dev.txt
python -B -m unittest discover -s tests -v
python -B tests/ui_drive_smoke.py
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Model, 38 ảnh gốc và index được giữ lại để sử dụng. Tệp ZIP tải thử, ảnh chụp UI, log và cache tải model được dọn sau kiểm thử; các kiểm thử dùng thư mục tạm tự dọn.
