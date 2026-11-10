# Face Search — Amazing race 2026

Ứng dụng chạy local để nhập toàn bộ ảnh sự kiện từ một thư mục Google Drive cố định và tìm những tấm ảnh có bạn bằng một ảnh selfie. Giao diện **Amazing race 2026** có phần nhập Drive, tìm kiếm, tải từng ảnh gốc và tải tất cả kết quả thành ZIP; album hiển thị 24 ảnh mới nhất.

Pipeline của dự án:

```text
Google Drive → ảnh sự kiện → SCRFD phát hiện khuôn mặt → căn chỉnh → ArcFace tạo embedding
            → FAISS lưu vector + metadata

Ảnh selfie  → phát hiện và căn chỉnh khuôn mặt → ArcFace → tìm trong FAISS
            → lọc theo ngưỡng tương đồng → danh sách ảnh phù hợp
```

**Để sử dụng local, bạn chỉ cần cài thư viện, tải đúng hai model bên dưới và giữ file `storage/best_config.json` đi kèm source.** Bạn có thể chạy trên CPU; không cần GPU hay dữ liệu bổ sung để bắt đầu dùng ảnh của mình.

## 1. Chuẩn bị

- Cài Python **64-bit**. Môi trường Windows đã được kiểm tra với **Python 3.14.5**; ví dụ PowerShell bên dưới dùng Python 3.14.
- Có kết nối Internet khi cài thư viện, tải model và nhập ảnh từ Drive. Sau khi nhập, tìm kiếm và tải bản ảnh đã lưu diễn ra trên máy local.
- Tải hoặc clone source của dự án và mở terminal tại thư mục có `README.md`, `requirements.txt` và `app.py`.

Ví dụ vào thư mục dự án trên Windows; thay đường dẫn cho đúng máy của bạn:

```powershell
cd "D:\Projects\face_search"
```

Cấu trúc các file quan trọng sau khi chuẩn bị:

```text
face_search/
├── app.py
├── README.md
├── requirements.txt
├── .venv/                         # Môi trường Python cục bộ
├── .env                           # Tùy chọn, chứa HF_TOKEN nếu cần
├── models/
│   ├── scrfd_10g.onnx              # Bắt buộc tải
│   └── arcface_w600k_r50.onnx      # Bắt buộc tải
├── data/
│   ├── event_photos/              # Ảnh gốc được nhập từ Drive
│   ├── queries/                   # Ảnh query cho CLI, nếu dùng
├── storage/
│   ├── best_config.json           # Cấu hình có sẵn trong repository
│   ├── faces.faiss                # Tự sinh sau khi nhập Drive/lập chỉ mục
│   └── metadata.json              # Tự sinh cùng faces.faiss
├── src/
└── scripts/
```

Các model, ảnh nhập từ Drive và index phát sinh được bỏ qua bởi Git, nên người mới tải source cần tự tải model. Không cần tạo sẵn `faces.faiss` hoặc `metadata.json`; ứng dụng có thể bắt đầu với album và index rỗng.

## 2. Tạo môi trường và cài thư viện

### Windows — PowerShell

Kiểm tra Python rồi tạo môi trường:

```powershell
py -3.14 --version
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Nếu lệnh `py` chưa có, kiểm tra việc cài Python/launcher. Nếu `python --version` đã trỏ đúng Python bạn muốn dùng, có thể thay lệnh tạo môi trường bằng `python -m venv .venv`.

### Linux/macOS — Bash/Zsh

```bash
python3 --version
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip check
```

Các lệnh dùng trực tiếp Python trong `.venv`, vì vậy **không bắt buộc activate môi trường**. Trên PowerShell dùng `.venv\Scripts\python.exe`; trên Linux/macOS dùng `.venv/bin/python`. Không dùng đường dẫn `./.venv/bin/python` trên Windows.

`requirements.txt` cài các thư viện cho xử lý ảnh, ONNX Runtime CPU, FAISS, FastAPI và công cụ tải dữ liệu. Khi kiểm tra thành công, `pip check` báo `No broken requirements found.`

## 3. Tải hai model bắt buộc

Nguồn model: [globalnebula/insightface-buffalo-l-onnx trên Hugging Face](https://huggingface.co/globalnebula/insightface-buffalo-l-onnx).

| File cần tải | Vai trò | Dung lượng tham khảo | Vị trí cần lưu |
|---|---|---:|---|
| [`scrfd_10g.onnx`](https://huggingface.co/globalnebula/insightface-buffalo-l-onnx/resolve/main/scrfd_10g.onnx) | Phát hiện khuôn mặt, bounding box và 5 landmarks | 16,9 MB | `models/scrfd_10g.onnx` |
| [`arcface_w600k_r50.onnx`](https://huggingface.co/globalnebula/insightface-buffalo-l-onnx/resolve/main/arcface_w600k_r50.onnx) | Tạo vector embedding 512 chiều cho khuôn mặt đã căn chỉnh | 174,4 MB | `models/arcface_w600k_r50.onnx` |

Tổng dung lượng hai file khoảng **191 MB**. Đây là các model đã huấn luyện sẵn; không cần train lại.

### Cách A — Tải trực tiếp bằng trình duyệt

1. Nhấn từng liên kết file trong bảng trên và tải file về máy.
2. Tạo thư mục `models` ở cùng cấp với `app.py` nếu chưa có.
3. Chép cả hai file vào `models/` và giữ nguyên tên.
4. Kiểm tra không bị lồng thành `models/models/`, không có đuôi `.txt`, và không chỉ tải trang HTML hoặc file Git LFS pointer thay cho model thực.

Hai file này hiện tải được công khai; **không bắt buộc có `HF_TOKEN` để tải model hoặc chạy app local**.

### Cách B — Tải bằng Python

Chạy sau khi đã cài `requirements.txt`. Đoạn mã sẽ tải đúng hai file vào `models/`. Nếu có token trong `.env` thì sử dụng token đó; nếu không có thì tải công khai.

**Windows — sao chép nguyên khối dưới đây vào PowerShell:**

```powershell
@'
from pathlib import Path
from dotenv import dotenv_values
from huggingface_hub import hf_hub_download

token = dotenv_values(".env", encoding="utf-8-sig").get("HF_TOKEN") or False
repo = "globalnebula/insightface-buffalo-l-onnx"
Path("models").mkdir(exist_ok=True)

for filename in ("scrfd_10g.onnx", "arcface_w600k_r50.onnx"):
    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir="models",
        token=token,
    )
    print(Path(path).name, Path(path).stat().st_size, "bytes")
'@ | .\.venv\Scripts\python.exe -
```

**Linux/macOS — dùng khối Bash dưới đây:**

```bash
./.venv/bin/python - <<'PY'
from pathlib import Path
from dotenv import dotenv_values
from huggingface_hub import hf_hub_download

token = dotenv_values(".env", encoding="utf-8-sig").get("HF_TOKEN") or False
repo = "globalnebula/insightface-buffalo-l-onnx"
Path("models").mkdir(exist_ok=True)

for filename in ("scrfd_10g.onnx", "arcface_w600k_r50.onnx"):
    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir="models",
        token=token,
    )
    print(Path(path).name, Path(path).stat().st_size, "bytes")
PY
```

Cú pháp `<<'PY'` dành cho Bash, không dùng trực tiếp trong PowerShell. Tham khảo thêm [hướng dẫn tải file của Hugging Face](https://huggingface.co/docs/huggingface_hub/guides/download).

### Tùy chọn: thêm Hugging Face token

Token hữu ích khi cần xác thực, tăng giới hạn tải hoặc truy cập bộ dữ liệu có giới hạn quyền. Để tạo token, đăng nhập [Hugging Face → Access Tokens](https://huggingface.co/settings/tokens), chọn **New token** và cấp quyền đọc phù hợp. Xem [tài liệu User Access Tokens](https://huggingface.co/docs/hub/security-tokens).

Tạo file tên chính xác là `.env`, cùng cấp với `README.md`, lưu dạng UTF-8 và điền:

```dotenv
HF_TOKEN=hf_THAY_BANG_TOKEN_THUC_TE_CUA_BAN
```

Thay phần sau dấu `=` bằng token thật rồi lưu file. `.env` chỉ chứa cấu hình dạng `TEN=GIA_TRI`, không chứa đoạn mã Python tải model. Không đặt tên thành `.env.txt`, không đưa token lên Git hoặc ghi token vào log/tài liệu chia sẻ. File `.env` đã được dự án bỏ qua trong Git.

### Kiểm tra model trước khi chạy

Tại thư mục gốc dự án, kiểm tra cả hai model có thể nạp bằng ONNX Runtime CPU:

**Windows:**

```powershell
.\.venv\Scripts\python.exe -c "import onnxruntime as ort; files=('scrfd_10g.onnx','arcface_w600k_r50.onnx'); [print(f, ort.InferenceSession('models/'+f, providers=['CPUExecutionProvider']).get_providers()) for f in files]"
```

**Linux/macOS:**

```bash
./.venv/bin/python -c "import onnxruntime as ort; files=('scrfd_10g.onnx','arcface_w600k_r50.onnx'); [print(f, ort.InferenceSession('models/'+f, providers=['CPUExecutionProvider']).get_providers()) for f in files]"
```

Nếu thành công, mỗi tên model được in kèm `['CPUExecutionProvider']`, không có traceback. Chỉ kiểm tra file tồn tại hoặc kích thước lớn hơn 0 chưa đủ để xác nhận model nạp được.

## 4. Chạy ứng dụng local

Giữ nguyên `storage/best_config.json` đi kèm repository. App và các script lập chỉ mục/tìm kiếm sẽ tự đọc cấu hình này.

**Windows — PowerShell:**

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

**Linux/macOS:**

```bash
./.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Giữ terminal này mở. Khi thấy `Uvicorn running on http://127.0.0.1:8000`, truy cập:

- [Giao diện ứng dụng](http://127.0.0.1:8000/).
- [Kiểm tra server — `/health`](http://127.0.0.1:8000/health), kết quả mong đợi: `{"status":"ok"}`.
- [Tài liệu API — `/docs`](http://127.0.0.1:8000/docs).

`/health` xác nhận server đang hoạt động; hãy dùng bước kiểm tra model ở trên để xác nhận việc nạp model. Lần nhập Drive hoặc tìm kiếm đầu tiên sẽ khởi tạo model, nên có thể cần chờ thêm.

Album trống ở lần chạy đầu là bình thường. Chưa có ảnh sự kiện thì chưa có ảnh để tìm kiếm.

### Nhập toàn bộ ảnh từ Google Drive

Nguồn duy nhất là thư mục [Database](https://drive.google.com/drive/folders/1Fw76RVGWqtaKqCVBnDi3TR3oFqOfHLd_). Link được cố định trong `src/drive_import.py`; giao diện và API không nhận link thay thế hay ảnh sự kiện từ máy người dùng. Ứng dụng dùng [gdown](https://github.com/wkentaro/gdown) để đọc thư mục chia sẻ, không cần API key Google. Thư mục và ảnh phải cho phép **Bất kỳ ai có đường liên kết → Người xem**, đồng thời cho phép tải xuống.

1. Lưu ảnh JPG/JPEG/PNG vào thư mục Drive trên, có thể đặt trong các thư mục con.
2. Ở **Nhập ảnh từ Drive**, nhấn **Nhập toàn bộ ảnh** và chờ xử lý hoàn tất.
3. Xem số ảnh tìm thấy, số ảnh thêm mới, số khuôn mặt và lỗi của từng ảnh nếu có. Mỗi thời điểm chỉ chạy một đợt nhập.
4. Gallery bên dưới tự cập nhật, hiển thị tối đa 24 ảnh mới nhất, xếp ảnh mới lên trước. Nhấn một ảnh để mở bản gốc.

Ảnh gốc được lưu tại `data/event_photos/`, giữ nguyên nội dung. Toàn bộ đợt nhập dùng encoder, FAISS và cấu hình cũ; index/metadata được lưu một lần sau đợt xử lý. Không cần chạy `build_index.py` sau khi dùng giao diện. Tệp không phải JPG/JPEG/PNG được bỏ qua; lỗi tải/đọc một ảnh được báo riêng và các ảnh khác vẫn tiếp tục. Các tệp tải tạm được tự xóa cả khi có lỗi.

Nhấn nhập lần nữa để lấy ảnh mới trên Drive. Nội dung ảnh đã có và đã lập chỉ mục được bỏ qua, tránh thêm bản sao vector. Ảnh không có khuôn mặt vẫn nằm trong album; lần nhập sau có thể được nhận diện lại vì chưa có embedding. Nhập Drive không xóa ảnh/index cũ khi một ảnh bị gỡ khỏi Drive.

### Tìm ảnh bằng selfie

1. Ở **Tìm ảnh của bạn**, chọn một ảnh rõ mặt, đủ sáng, ưu tiên chụp chính diện.
2. Nhấn **Tìm ảnh** và chờ xử lý.
3. Kết quả xuất hiện trong khu vực **Ảnh có bạn**; nhấn một kết quả để mở ảnh gốc. Có thể đóng khu vực kết quả để tiếp tục xem album.
4. Nhấn **Tải ảnh gốc** bên dưới từng kết quả, hoặc **Tải tất cả ảnh (.zip)** để nhận `anh-cua-ban.zip` chứa thư mục `anh-cua-ban/` với toàn bộ ảnh trả về, không giới hạn 100 ảnh.

Selfie chỉ dùng làm truy vấn, **không được thêm vào album hoặc FAISS index**. Nếu selfie có nhiều khuôn mặt, hệ thống sử dụng khuôn mặt hợp lệ lớn nhất. Không tìm thấy khuôn mặt hoặc không có ảnh phù hợp sẽ có thông báo tương ứng.

ZIP chứa bản ảnh gốc đã lưu, không phải ảnh thumbnail hay ảnh cắt mặt. Tệp ZIP tạm trên server được xóa khi phản hồi kết thúc. Nếu một ảnh đã bị xóa khỏi album, API báo lỗi thay vì âm thầm tải một ZIP thiếu ảnh.

### Dữ liệu được lưu ở đâu?

| Nội dung | Nơi lưu | Khi khởi động lại |
|---|---|---|
| Ảnh gốc đã nhập từ Drive | `data/event_photos/` | Giữ nguyên trên ổ đĩa |
| Các vector khuôn mặt | `storage/faces.faiss` | App tự nạp lại nếu có |
| Đường dẫn ảnh và bounding box | `storage/metadata.json` | Phải đi cùng FAISS index |
| Cấu hình detector/tìm kiếm | `storage/best_config.json` | App tự đọc |

Khi chuyển dữ liệu sự kiện sang máy khác, cần giữ các ảnh và cặp `faces.faiss`/`metadata.json` đồng bộ. Nếu thay đổi đường dẫn ảnh hoặc hệ điều hành, nên lập lại index theo bước 5 để metadata dùng đường dẫn trên máy đích. Đóng trình duyệt hoặc tắt server không xóa ảnh đã nhập.

### Tắt server

Trong terminal đang chạy Uvicorn, nhấn **Ctrl+C** và chờ tiến trình dừng. Đóng tab trình duyệt không tắt server.

Trên Windows, có thể kiểm tra cổng sau khi tắt:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
```

Nếu không có dòng nào ở trạng thái `Listen`, cổng 8000 không còn server lắng nghe.

## 5. Các lệnh CLI tùy chọn

Các ví dụ dưới đây dùng PowerShell và chạy tại thư mục gốc. Trên Linux/macOS, thay `.\.venv\Scripts\python.exe` bằng `./.venv/bin/python`.

### Lập lại index từ ảnh có sẵn

Nếu cần khôi phục index từ ảnh đã lưu tại `data/event_photos/`, chạy:

```powershell
.\.venv\Scripts\python.exe scripts/build_index.py --image-folder data/event_photos
```

Lệnh tạo lại `storage/faces.faiss` và `storage/metadata.json` từ toàn bộ ảnh trong thư mục đã chỉ định, thay cho index cũ. Nên dừng app trước khi build và mở lại app sau khi xong để nạp index mới. Không đưa ảnh query vào thư mục gallery này.

### Tìm kiếm bằng một file query

Đặt ảnh query tại `data/queries/selfie.jpg`, sau đó chạy khi đã có index:

```powershell
.\.venv\Scripts\python.exe scripts/search.py --query data/queries/selfie.jpg
```

Kết quả CLI gồm thứ hạng, điểm tương đồng và đường dẫn ảnh. Ví dụ dùng đường dẫn tương đối, tên file ASCII để dễ chạy trên Windows; giao diện nhận ảnh qua upload và xử lý nội dung file.

## 6. NVIDIA GPU — tùy chọn

CPU là cách chạy mặc định từ `requirements.txt`. Chỉ chuyển sang GPU khi đã có driver NVIDIA và CUDA/cuDNN tương thích với phiên bản ONNX Runtime GPU. Có driver NVIDIA không đồng nghĩa máy đã có đủ thư viện CUDA/cuDNN.

Theo [tài liệu CUDA Execution Provider của ONNX Runtime](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html), chọn bộ thư viện tương thích trước khi cài. Tránh cài đồng thời `onnxruntime` và `onnxruntime-gpu` trong cùng môi trường vì chúng cung cấp cùng module Python.

**Windows:**

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y onnxruntime
.\.venv\Scripts\python.exe -m pip install --upgrade onnxruntime-gpu
```

**Linux:**

```bash
./.venv/bin/python -m pip uninstall -y onnxruntime
./.venv/bin/python -m pip install --upgrade onnxruntime-gpu
```

Wheel GPU vẫn có CPU provider. Code hiện tại ưu tiên CUDA khi provider có sẵn và giữ CPU trong danh sách provider. Nếu thiếu DLL/thư viện CUDA, cần kiểm tra provider thực tế của session; việc `get_available_providers()` liệt kê CUDA chưa chứng minh model đã chạy trên GPU. Có thể tiếp tục dùng bản CPU để chạy local.

Nếu cài lại `requirements.txt`, gói CPU sẽ được yêu cầu cài lại; hãy kiểm tra và chọn lại một biến thể ONNX Runtime phù hợp sau đó.

## 7. API hiện có

| Phương thức | Đường dẫn | Công dụng |
|---|---|---|
| `GET` | `/` | Giao diện album, nhập Drive, tìm kiếm và tải ảnh |
| `GET` | `/health` | Kiểm tra server hoạt động |
| `POST` | `/events` | Nhập toàn bộ ảnh từ thư mục Drive cố định; không cần body |
| `POST` | `/search` | Tìm ảnh bằng một selfie; multipart field `file` |
| `GET` | `/event-photos/<đường-dẫn-ảnh>` | Xem ảnh sự kiện đã lưu |
| `GET` | `/downloads/photo/<đường-dẫn-trong-album>` | Tải một ảnh gốc dưới dạng attachment |
| `POST` | `/downloads/zip` | Tải ZIP; JSON `{"photos": ["data/event_photos/..."]}` lấy từ `matches[].photo_path` |
| `GET` | `/docs` | Tài liệu API tương tác của FastAPI |

## 8. Xử lý lỗi thường gặp

| Hiện tượng | Cách kiểm tra |
|---|---|
| Không tìm thấy `.venv/bin/python` trên Windows | Dùng `.\.venv\Scripts\python.exe`; thư mục `bin` dành cho Linux/macOS. |
| PowerShell báo lỗi ở `<<'PY'` | Dùng khối PowerShell ở bước tải model, không dán khối Bash vào PowerShell. |
| `ModuleNotFoundError` | Cài lại `requirements.txt` bằng Python trong `.venv`, dùng cùng Python đó để chạy app. |
| Pip không tìm thấy gói hoặc tải bị lỗi | Kiểm tra Python 64-bit, kết nối mạng, proxy và cấu hình pip (`PIP_NO_INDEX`, index URL). |
| Báo thiếu/không nạp được model | Kiểm tra đúng tên, vị trí và kích thước hai file; chạy lệnh kiểm tra ONNX Runtime ở bước 3. |
| Báo thiếu `storage/best_config.json` | Lấy lại file cấu hình đi kèm repository khi chép source. |
| HTTP 401/403 khi tải từ Hugging Face | Nếu dùng token, kiểm tra token còn hiệu lực và quyền đọc; với dataset gated, tài khoản phải được cấp quyền truy cập trước. Hai model công khai cũng có thể tải bằng liên kết trực tiếp. |
| Không mở được trang local | Kiểm tra terminal Uvicorn còn chạy, URL và cổng đúng. Sau khi chỉ khởi động lại server, tải lại trang trình duyệt. |
| Cổng 8000 đã được dùng | Dừng tiến trình server cũ, hoặc đổi `--port 8001` và mở `http://127.0.0.1:8001`. |
| Không đọc/tải được thư mục Drive | Kiểm tra link nguồn, Internet, quyền Người xem cho bất kỳ ai có link và quyền tải ảnh. |
| Album có ảnh nhưng tìm không ra | Chờ nhập Drive xử lý xong; kiểm tra ảnh có khuôn mặt rõ và cặp index/metadata còn đồng bộ. Khi cần, khôi phục index theo bước 5. |
| Không tìm thấy khuôn mặt trong selfie | Dùng ảnh đủ sáng, rõ mặt, ít che khuất; thử ảnh chụp chính diện. |
| Không có ảnh phù hợp | Kiểm tra đã tải ảnh sự kiện có người cần tìm; thử selfie khác. |

Phần cài đặt và hướng dẫn trên sử dụng các chức năng hiện có của dự án; không cần thay đổi encoder, vector store, search engine hoặc pipeline để chạy local.

## 9. Kiểm thử

```powershell
python -m pip install -r requirements-dev.txt
python -B -m unittest discover -s tests -v
```

Các kiểm thử nhập/tải dùng thư mục tạm tự dọn và FAISS thật, kiểm tra lỗi Drive, ảnh hỏng, chống trùng, đường dẫn tải, ZIP và luồng tìm/tải hơn 100 ảnh. Kiểm thử model thật chạy khi có hai model và đã nhập ảnh Drive (có ảnh và metadata); kiểm thử cũ với bộ 10 selfie riêng chỉ chạy khi có đủ bộ dữ liệu đó. Lệnh `unittest` trên không gọi Drive để tránh phụ thuộc mạng.

Để kiểm tra đầy đủ nút nhập Drive trên UI bằng trình duyệt thật, sau khi tải hai model:

```powershell
python -B tests/ui_drive_smoke.py
```

Lệnh này chạy Uvicorn và trình duyệt, nhập toàn bộ ảnh từ Drive thật vào album/index tạm rỗng, tìm kiếm, tải ảnh gốc và ZIP rồi đối chiếu nội dung. Đồng thời kiểm tra server vẫn phản hồi và từ chối đợt nhập thứ hai trong lúc đang xử lý. Không dùng ảnh tải sẵn hay giả lập phản hồi Drive. Album/index đang sử dụng được giữ nguyên; dữ liệu và server kiểm thử được tự dọn khi kết thúc.

Trên Windows, kiểm thử dùng Edge đã cài trên máy. Với Linux/macOS, cài trình duyệt kiểm thử bằng `python -m playwright install chromium` trước khi chạy. Kiểm thử này cần Internet và có thể mất vài phút tùy số ảnh/kết nối; thời gian chờ nhập mặc định là 600 giây, có thể đổi bằng `--timeout`.
