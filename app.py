"""Small FastAPI UI/API for adding event photos and searching them."""

from __future__ import annotations

import hashlib
import html
import json
import re
import tempfile
import threading
import zipfile
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from src.drive_import import (
    DRIVE_FOLDER_URL,
    IMAGE_SUFFIXES,
    DriveImportError,
    download_drive_photo,
    list_drive_photos,
)
from src.face_encoder import FaceEncoder
from src.search_engine import SearchEngine
from src.vector_store import VectorStore


PROJECT_ROOT = Path(__file__).resolve().parent
EVENT_PHOTOS_DIR = PROJECT_ROOT / "data" / "event_photos"
MODELS_DIR = PROJECT_ROOT / "models"
STORAGE_DIR = PROJECT_ROOT / "storage"
EVENT_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Face Search")
app.mount(
    "/event-photos",
    StaticFiles(directory=str(EVENT_PHOTOS_DIR)),
    name="event-photos",
)

_runtime = None
_runtime_lock = threading.RLock()
_import_lock = threading.Lock()


def _read_config() -> dict:
    config_path = STORAGE_DIR / "best_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            "storage/best_config.json is missing; run scripts/tune.py before starting the app"
        )
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to read best_config.json: {exc}") from exc
    if not isinstance(config, dict):
        raise RuntimeError("storage/best_config.json must contain a JSON object")
    return config


def _load_runtime():
    global _runtime
    with _runtime_lock:
        if _runtime is not None:
            return _runtime
        config = _read_config()
        detector_path = MODELS_DIR / config.get("detector_model", "scrfd_10g.onnx")
        embedder_path = MODELS_DIR / config.get("embedding_model", "arcface_w600k_r50.onnx")
        det_size = int(config.get("det_size", 640))
        encoder = FaceEncoder(
            str(detector_path),
            str(embedder_path),
            det_size=(det_size, det_size),
            det_threshold=float(config.get("det_threshold", 0.5)),
            min_face_size=float(config.get("min_face_size", 20)),
        )
        store = VectorStore(dimension=int(config.get("embedding_dim", 512)))
        index_path = STORAGE_DIR / "faces.faiss"
        metadata_path = STORAGE_DIR / "metadata.json"
        if index_path.is_file() or metadata_path.is_file():
            store.load(str(index_path), str(metadata_path))
        engine = SearchEngine(
            encoder,
            store,
            similarity_threshold=float(config.get("similarity_threshold", 0.5)),
            top_k=int(config.get("top_k", 100)),
        )
        _runtime = {"encoder": encoder, "store": store, "engine": engine}
        return _runtime


def _decode_upload(contents: bytes):
    if not contents:
        raise ValueError("Uploaded file is empty")
    image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("Uploaded file is not a readable image")
    return image


def _relative_photo_path(path: str) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    try:
        return candidate.resolve().relative_to(EVENT_PHOTOS_DIR.resolve()).as_posix()
    except ValueError:
        return candidate.name


def _photo_is_indexed(store: VectorStore, path: Path) -> bool:
    """Check whether an event photo already has a face in the FAISS index."""
    resolved = path.resolve()
    relative = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    for item in store.metadata.values():
        stored = str(item.get("photo_path", ""))
        if stored == relative:
            return True
        try:
            if (PROJECT_ROOT / stored).resolve() == resolved:
                return True
        except (OSError, ValueError):
            continue
    return False


def _result_for_api(result: dict) -> dict:
    output = dict(result)
    relative = _relative_photo_path(str(result["photo_path"]))
    output["photo_url"] = "/event-photos/" + quote(relative)
    output["download_url"] = "/downloads/photo/" + quote(relative)
    return output


def _resolve_download_photo(photo_path: str | Path) -> Path:
    """Only serve actual photos inside the event album, including symlink checks."""
    try:
        candidate = (PROJECT_ROOT / photo_path).resolve()
        candidate.relative_to(EVENT_PHOTOS_DIR.resolve())
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Đường dẫn ảnh không hợp lệ.") from exc
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Ảnh không còn trong album.")
    if candidate.suffix.lower() not in IMAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail="Tệp không phải ảnh được hỗ trợ.")
    return candidate


class PhotoDownloads(BaseModel):
    photos: list[str] = Field(min_length=1)


@app.get("/downloads/photo/{photo_path:path}")
def download_photo(photo_path: str):
    path = _resolve_download_photo(EVENT_PHOTOS_DIR / photo_path)
    return FileResponse(path, filename=path.name)


def _archive_chunks(archive):
    try:
        while chunk := archive.read(64 * 1024):
            yield chunk
    finally:
        archive.close()


@app.post("/downloads/zip")
def download_photos(request: PhotoDownloads):
    # Validate the complete selection before creating an archive. Do not silently
    # return an incomplete ZIP when an original has been removed.
    paths = list(dict.fromkeys(_resolve_download_photo(path) for path in request.photos))
    archive = tempfile.TemporaryFile()
    try:
        # JPEG/PNG are already compressed. ZIP_STORED avoids recompressing them.
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
            for path in paths:
                relative = path.relative_to(EVENT_PHOTOS_DIR.resolve()).as_posix()
                bundle.write(path, arcname="anh-cua-ban/" + relative)
        archive.seek(0)
    except Exception:
        archive.close()
        raise
    return StreamingResponse(
        _archive_chunks(archive),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="anh-cua-ban.zip"'},
        background=BackgroundTask(archive.close),
    )


@app.get("/", response_class=HTMLResponse)
def home():
    """Render the event album, Drive import, search, and result downloads."""
    photos = []
    for path in EVENT_PHOTOS_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            try:
                photos.append((path.stat().st_mtime_ns, path))
            except FileNotFoundError:
                continue
    photos.sort(key=lambda item: (item[0], item[1].as_posix()), reverse=True)
    cards = []
    for position, (_, path) in enumerate(photos[:24]):
        url = "/event-photos/" + quote(path.relative_to(EVENT_PHOTOS_DIR).as_posix())
        title = html.escape(path.stem.rsplit("__", 1)[0], quote=True)
        badge = '<span class="photo-badge">Mới nhất</span>' if position == 0 else ""
        loading = "eager" if position < 4 else "lazy"
        cards.append(
            f'<a class="photo-card" href="{url}" target="_blank" rel="noopener noreferrer" '
            f'aria-label="Xem ảnh {title}"><div class="photo-image">'
            f'<img src="{url}" alt="{title}" loading="{loading}" decoding="async">'
            f'{badge}<span class="photo-open" aria-hidden="true">↗</span></div>'
            f'<span class="photo-caption" title="{title}">{title}</span></a>'
        )
    album = "".join(cards) or (
        '<div class="empty-album"><svg class="icon" aria-hidden="true">'
        '<use href="#icon-photos"></use></svg><h3>Hành trình bắt đầu từ đây</h3>'
        '<p>Nhập ảnh từ thư mục Drive để cùng lưu lại khoảnh khắc của sự kiện.</p></div>'
    )
    page = """<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#1659e5">
<title>Amazing race 2026 · Album sự kiện</title>
<style>
:root{color-scheme:light;--blue:#1659e5;--blue-dark:#1046bc;--ink:#102544;--muted:#69788e;--line:#e2e9f3;--soft:#edf4ff;--white:#fff}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:24px}body{margin:0;background:#f6f8fc;color:var(--ink);font-family:"Segoe UI",Arial,sans-serif;font-size:15px;line-height:1.55}button,input{font:inherit}a{color:inherit}button,a,label,input{-webkit-tap-highlight-color:transparent}button{cursor:pointer}button:disabled{cursor:wait;opacity:.65}a:focus-visible,button:focus-visible{outline:3px solid #77a8ff;outline-offset:4px}[hidden]{display:none!important}.icon{width:22px;height:22px;fill:none;stroke:currentColor;stroke-width:1.7;stroke-linecap:round;stroke-linejoin:round;flex-shrink:0}.sprite{position:absolute;width:0;height:0;overflow:hidden}.wrap{max-width:1184px;margin:0 auto;padding:0 32px}
.topbar{background:var(--white);border-bottom:1px solid var(--line)}.nav{min-height:78px;display:flex;align-items:center;justify-content:space-between;gap:20px}.brand{display:flex;align-items:center;gap:12px;text-decoration:none}.brand-mark{display:grid;place-items:center;width:42px;height:42px;background:var(--blue);color:white;border-radius:12px}.brand-mark .icon{width:28px;height:28px;stroke-width:2.2}.brand-name{font-weight:750;font-size:18px;letter-spacing:-.5px}.brand-name span{font-weight:450;color:var(--muted)}.nav-link{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--blue);font-weight:600;text-decoration:none}.nav-link .icon{width:17px;height:17px}
.hero{padding:48px 0 31px;display:flex;align-items:flex-end;justify-content:space-between;gap:24px}.eyebrow{margin:0 0 10px;color:var(--blue);font-size:11px;font-weight:750;letter-spacing:2px;text-transform:uppercase}.eyebrow:before{content:"";display:inline-block;vertical-align:middle;width:23px;height:2px;background:var(--blue);margin-right:10px}h1{margin:0;font-size:clamp(33px,4.4vw,54px);font-weight:750;line-height:1.15;letter-spacing:-2px}h1 span{color:var(--blue)}.hero-copy{margin:15px 0 0;color:var(--muted);font-size:15px;max-width:560px}.album-stamp{display:flex;align-items:center;gap:13px;padding:13px 17px;border:1px solid var(--line);border-radius:13px;background:var(--white);flex-shrink:0}.album-stamp>.icon{color:var(--blue);width:27px;height:27px}.stamp-count{font-size:20px;font-weight:750;line-height:1.2}.stamp-label{font-size:11px;color:var(--muted);margin-top:2px}
.actions{display:grid;grid-template-columns:1fr 1fr;gap:20px}.action-card{background:var(--white);border:1px solid var(--line);border-radius:18px;padding:25px 25px 23px;box-shadow:0 5px 16px #18356803}.card-top{display:flex;align-items:flex-start;gap:13px;margin-bottom:22px}.card-icon{width:43px;height:43px;display:grid;place-items:center;background:var(--soft);color:var(--blue);border-radius:12px;flex-shrink:0}.card-step{margin:0 0 2px;font-size:10px;font-weight:700;letter-spacing:1.3px;color:var(--muted)}h2{font-size:21px;font-weight:700;letter-spacing:-.5px;line-height:1.3;margin:0}.card-description{margin:5px 0 0;color:var(--muted);font-size:13px}.file-row{display:flex;gap:10px;align-items:stretch}.file-control{position:relative;display:flex;align-items:center;gap:10px;padding:11px 13px;min-width:0;flex:1;border:1px solid #d9e3f1;border-radius:10px;background:#fbfcfe;cursor:pointer;overflow:hidden;transition:border-color .15s,background .15s}.file-control:hover{border-color:#9cbbf7;background:var(--soft)}.file-control:focus-within{outline:3px solid #77a8ff;outline-offset:3px}.file-control>.icon{color:#7a93ba;width:20px;height:20px}.file-copy{min-width:0}.file-name{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:12px;font-weight:600;max-width:100%}.file-hint{display:block;font-size:10px;color:var(--muted);margin-top:1px}.file-input{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:pointer}.primary-button{display:flex;align-items:center;justify-content:center;gap:7px;flex-shrink:0;border:1px solid var(--blue);border-radius:10px;padding:12px 15px;color:white;background:var(--blue);font-size:12px;font-weight:650;transition:background .15s,transform .15s}.primary-button:hover:not(:disabled){background:var(--blue-dark)}.primary-button .icon{width:17px;height:17px}.card-note{display:flex;align-items:center;gap:6px;font-size:10px;color:#7a879a;margin:12px 0 0}.card-note .icon{width:13px;height:13px}.status{margin:14px 0 0;padding:11px 13px;border-radius:9px;font-size:12px;white-space:pre-line;overflow-wrap:anywhere}.status:empty{display:none}.status.pending{color:#285592;background:#edf4ff}.status.success{color:#216345;background:#edf8f1}.status.error{color:#ac3434;background:#fff0f0}.status.warning{color:#885b16;background:#fff8e8}
.album-section{margin-top:40px}.section-heading{display:flex;align-items:center;justify-content:space-between;gap:15px;margin-bottom:18px}.section-title{display:flex;align-items:center;gap:10px}.section-title h2{font-size:22px}.count-pill{font-size:11px;font-weight:650;background:#e9f0fe;color:var(--blue);padding:3px 9px;border-radius:20px;white-space:nowrap}.sort-note{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--muted)}.sort-note .icon{width:14px;height:14px}.section-description{font-size:12px;color:var(--muted);margin:5px 0 0}.photo-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:20px 16px}.photo-card{display:block;min-width:0;text-decoration:none;background:var(--white);border:1px solid var(--line);border-radius:13px;overflow:hidden;transition:transform .18s,box-shadow .18s}.photo-card:hover{transform:translateY(-3px);box-shadow:0 8px 24px #12366c12}.photo-image{position:relative;aspect-ratio:4/3;background:#e8eef7;overflow:hidden}.photo-image img{display:block;width:100%;height:100%;object-fit:cover;transition:transform .25s}.photo-card:hover img{transform:scale(1.035)}.photo-caption{display:block;margin:0;padding:10px 12px;font-size:11px;color:#65768e;text-overflow:ellipsis;white-space:nowrap;overflow:hidden}.photo-badge{position:absolute;top:10px;left:10px;color:var(--blue);background:#ffffffed;padding:4px 9px;border-radius:6px;font-size:9px;font-weight:750}.photo-open{position:absolute;right:10px;bottom:10px;display:grid;place-items:center;width:28px;height:28px;border-radius:50%;background:#ffffffed;color:var(--blue);font-size:16px;opacity:0;transition:opacity .18s}.photo-card:hover .photo-open,.photo-card:focus-visible .photo-open{opacity:1}.empty-album{grid-column:1/-1;text-align:center;padding:55px 25px;border:1px dashed #c7d6ec;border-radius:15px;background:var(--white);color:var(--muted)}.empty-album>.icon{width:36px;height:36px;color:#86a9e9}.empty-album h3{font-size:17px;color:var(--ink);margin:12px 0 5px}.empty-album p{margin:0;font-size:13px}.results-section{padding:25px;background:var(--white);border:1px solid #cdddfb;border-radius:18px;margin-top:25px;scroll-margin-top:24px}.results-section .section-heading{margin-bottom:0}.results-section .photo-grid:not(:empty){margin-top:20px}.results-section .status{margin-top:12px}.results-section .close-results{border:1px solid var(--line);background:white;border-radius:8px;padding:7px 11px;font-size:11px;color:var(--muted)}.footer{margin-top:43px;padding:23px 0 27px;border-top:1px solid var(--line);display:flex;justify-content:space-between;gap:16px;color:#8290a3;font-size:10px}.footer strong{font-weight:650;color:#5d7293}
@media(min-width:1600px){.wrap{max-width:1280px}.hero{padding-top:58px}}@media(max-width:900px){.wrap{padding:0 24px}.actions{gap:14px}.action-card{padding:20px}.file-row{flex-direction:column}.primary-button{padding:11px}.photo-grid{grid-template-columns:repeat(3,minmax(0,1fr))}.hero{padding-top:36px}.brand-name{font-size:16px}}@media(max-width:600px){.wrap{padding:0 18px}.nav{min-height:68px}.brand-mark{width:35px;height:35px;border-radius:10px}.brand-name{font-size:14px}.brand-name span{display:none}.nav-link{font-size:11px}.hero{padding:31px 0 24px;align-items:flex-start}.hero-copy{font-size:13px;max-width:290px}h1{font-size:35px;letter-spacing:-1.5px}.album-stamp{display:none}.actions{grid-template-columns:1fr;gap:14px}.action-card{padding:20px}.card-top{margin-bottom:17px}.file-row{flex-direction:row}.primary-button{padding:11px 13px}.file-name{font-size:11px}.photo-grid{grid-template-columns:repeat(2,minmax(0,1fr));gap:13px 10px}.album-section{margin-top:29px}.section-title h2{font-size:20px}.section-heading{align-items:flex-start}.sort-note{font-size:10px;margin-top:4px}.sort-note .icon{display:none}.section-description{font-size:11px}.photo-caption{font-size:10px;padding:9px}.results-section{padding:18px}.footer{margin-top:30px;font-size:9px;flex-direction:column;gap:4px}.photo-open{opacity:1;width:23px;height:23px;font-size:13px}.photo-badge{top:7px;left:7px;font-size:8px;padding:3px 7px}}@media(max-width:350px){.file-row{flex-direction:column}.sort-note{display:none}}@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}*,*:before,*:after{transition:none!important;animation:none!important}}
.drive-source{display:flex;align-items:center;gap:10px;min-width:0;flex:1;padding:11px 13px;border:1px solid #d9e3f1;border-radius:10px;background:#fbfcfe;color:var(--blue);font-size:12px;text-decoration:none}.drive-source .file-copy{overflow:hidden}.download-panel{margin-top:18px;padding:16px;background:var(--soft);border-radius:12px}.download-panel p{margin:0 0 12px;font-size:12px;color:var(--muted)}.download-panel h3{margin:0 0 4px;font-size:16px}.result-card>a{display:block;text-decoration:none}.photo-download{display:block;padding:9px 12px;border-top:1px solid var(--line);font-size:12px;font-weight:600;color:var(--blue);text-align:center;text-decoration:none}.photo-download:hover{background:var(--soft)}
</style>
<style>.photo-score{display:block;padding:0 12px 8px;font-size:10px;color:var(--muted);text-align:center}</style>
</head>
<body>
<svg class="sprite" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><defs>
<symbol id="icon-race" viewBox="0 0 24 24"><path d="m5 6 6 6-6 6m7-12 6 6-6 6"/></symbol>
<symbol id="icon-upload" viewBox="0 0 24 24"><path d="M12 16V3m-5 5 5-5 5 5M4 15v5a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-5"/></symbol>
<symbol id="icon-download" viewBox="0 0 24 24"><path d="M12 3v13m-5-5 5 5 5-5M4 16v4a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-4"/></symbol>
<symbol id="icon-photos" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="m21 15-5-5L5 21M3 16l4-4 3 3"/></symbol>
<symbol id="icon-search" viewBox="0 0 24 24"><circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/></symbol>
<symbol id="icon-person" viewBox="0 0 24 24"><circle cx="12" cy="8" r="4"/><path d="M4 21v-2a8 8 0 0 1 16 0v2"/></symbol>
<symbol id="icon-info" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v.1"/></symbol>
<symbol id="icon-clock" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></symbol>
</defs></svg>
<header class="topbar"><div class="wrap nav">
 <a class="brand" href="#top" aria-label="Amazing race 2026 — đầu trang"><span class="brand-mark"><svg class="icon" aria-hidden="true"><use href="#icon-race"></use></svg></span><span class="brand-name">amazing race<span> / 2026</span></span></a>
 <a class="nav-link" href="#latest-photos"><svg class="icon" aria-hidden="true"><use href="#icon-photos"></use></svg>Album sự kiện</a>
</div></header>
<main class="wrap" id="top">
 <div class="hero">
  <div><p class="eyebrow">Cùng nhau lưu giữ hành trình</p><h1>Amazing race <span>2026</span></h1><p class="hero-copy">Mỗi khoảnh khắc, một phần hành trình.<br>Tải ảnh sự kiện và tìm lại những tấm hình có bạn.</p></div>
  <div class="album-stamp"><svg class="icon" aria-hidden="true"><use href="#icon-photos"></use></svg><div><div class="stamp-count" id="album-total">__TOTAL_PHOTOS__</div><div class="stamp-label">khoảnh khắc trong album</div></div></div>
 </div>
 <div class="actions">
  <section class="action-card" aria-labelledby="upload-title">
   <div class="card-top"><div class="card-icon"><svg class="icon" aria-hidden="true"><use href="#icon-upload"></use></svg></div><div><p class="card-step">01 / CHIA SẺ KHOẢNH KHẮC</p><h2 id="upload-title">Nhập ảnh từ Drive</h2><p class="card-description">Đọc toàn bộ ảnh trong thư mục sự kiện đã liên kết.</p></div></div>
   <form id="event-form"><div class="file-row">
    <a class="drive-source" href="__DRIVE_FOLDER_URL__" target="_blank" rel="noopener noreferrer"><svg class="icon" aria-hidden="true"><use href="#icon-photos"></use></svg><span class="file-copy"><span class="file-name">Database · Google Drive ↗</span><span class="file-hint">JPG, JPEG, PNG · Bao gồm thư mục con</span></span></a>
    <button class="primary-button" id="event-upload-button" type="submit"><svg class="icon" aria-hidden="true"><use href="#icon-upload"></use></svg><span>Nhập toàn bộ ảnh</span></button>
   </div></form>
   <p class="card-note"><svg class="icon" aria-hidden="true"><use href="#icon-info"></use></svg>Chỉ đọc từ thư mục Drive này. Ảnh đã lập chỉ mục sẽ được bỏ qua.</p>
   <p id="event-status" class="status" role="status" aria-live="polite"></p>
  </section>
  <section class="action-card" aria-labelledby="search-title">
   <div class="card-top"><div class="card-icon"><svg class="icon" aria-hidden="true"><use href="#icon-person"></use></svg></div><div><p class="card-step">02 / TÌM LẠI CHÍNH MÌNH</p><h2 id="search-title">Tìm ảnh của bạn</h2><p class="card-description">Một tấm selfie, tìm lại những kỷ niệm có bạn.</p></div></div>
   <form id="search-form"><div class="file-row">
    <label class="file-control" for="query-file"><svg class="icon" aria-hidden="true"><use href="#icon-person"></use></svg><span class="file-copy"><span class="file-name" id="query-file-name">Chọn ảnh selfie của bạn</span><span class="file-hint">JPG, JPEG, PNG · Một ảnh rõ mặt</span></span><input class="file-input" id="query-file" type="file" name="file" accept="image/jpeg,image/png" required aria-label="Chọn ảnh selfie để tìm ảnh của bạn"></label>
    <button class="primary-button" id="search-button" type="submit"><svg class="icon" aria-hidden="true"><use href="#icon-search"></use></svg><span>Tìm ảnh</span></button>
   </div></form>
   <p class="card-note"><svg class="icon" aria-hidden="true"><use href="#icon-info"></use></svg>Selfie chỉ dùng để tìm kiếm, không được thêm vào album.</p>
  </section>
 </div>
 <section class="results-section" id="search-results" aria-labelledby="results-title" hidden>
  <div class="section-heading"><div class="section-title"><h2 id="results-title">Ảnh có bạn</h2><span id="result-count" class="count-pill" hidden></span></div><button class="close-results" id="close-results" type="button">Đóng kết quả</button></div>
  <p id="search-status" class="status" role="status" aria-live="polite"></p>
  <section class="download-panel" id="download-panel" aria-labelledby="download-title" hidden><h3 id="download-title">Tải ảnh của bạn</h3><p>Lưu từng ảnh bằng nút bên dưới ảnh, hoặc tải tất cả kết quả trong một tệp ZIP.</p><button class="primary-button" id="download-all" type="button"><svg class="icon" aria-hidden="true"><use href="#icon-download"></use></svg><span>Tải tất cả ảnh (.zip)</span></button><p id="download-status" class="status" role="status" aria-live="polite"></p></section>
  <div id="results" class="photo-grid"></div>
 </section>
 <section class="album-section" id="latest-photos" aria-labelledby="latest-title">
  <div class="section-heading"><div><div class="section-title"><h2 id="latest-title">Mới tải lên</h2><span class="count-pill">__TOTAL_PHOTOS__ ảnh</span></div><p class="section-description">Những khoảnh khắc mới nhất từ hành trình của chúng ta.</p></div><span class="sort-note"><svg class="icon" aria-hidden="true"><use href="#icon-clock"></use></svg>Mới nhất trước</span></div>
  <div class="photo-grid">__PHOTO_CARDS__</div>
  __ALBUM_NOTE__
 </section>
 <footer class="footer"><strong>Amazing race 2026</strong><span>Mỗi khoảnh khắc đều đáng nhớ. Cảm ơn bạn đã cùng chia sẻ.</span></footer>
</main>
<script>
const eventForm = document.querySelector('#event-form');
const eventStatus = document.querySelector('#event-status');
const eventUploadButton = document.querySelector('#event-upload-button');
const queryInput = document.querySelector('#query-file');
const searchForm = document.querySelector('#search-form');
const searchButton = document.querySelector('#search-button');
const searchStatus = document.querySelector('#search-status');
const resultsSection = document.querySelector('#search-results');
const resultCount = document.querySelector('#result-count');
const downloadPanel = document.querySelector('#download-panel');
const downloadAllButton = document.querySelector('#download-all');
const downloadStatus = document.querySelector('#download-status');
let currentMatches = [];
queryInput.addEventListener('change', () => {
 document.querySelector('#query-file-name').textContent = queryInput.files.length ? queryInput.files[0].name : 'Chọn ảnh selfie của bạn';
});
function setStatus(element, state, message) {
 element.className = 'status ' + state;
 element.textContent = message;
}
async function readResponse(response) {
 const data = await response.json().catch(() => null);
 if (!response.ok) throw new Error(typeof data?.detail === 'string' ? data.detail : `Yêu cầu chưa thành công (HTTP ${response.status}). Vui lòng thử lại.`);
 if (!data) throw new Error('Không đọc được phản hồi. Vui lòng thử lại.');
 return data;
}
async function refreshAlbum() {
 try {
  const response = await fetch('/', {cache: 'no-store'});
  if (!response.ok) throw new Error('Album unavailable');
  const page = new DOMParser().parseFromString(await response.text(), 'text/html');
  const album = page.querySelector('#latest-photos');
  if (!album) throw new Error('Album unavailable');
  document.querySelector('#latest-photos').replaceWith(album);
  document.querySelector('#album-total').textContent = page.querySelector('#album-total').textContent;
 } catch {
  eventStatus.textContent += '\\nTải lại trang để xem album mới nhất.';
 }
}
eventForm.addEventListener('submit', async event => {
 event.preventDefault();
 if (eventUploadButton.disabled) return;
 eventUploadButton.disabled = true;
 eventForm.setAttribute('aria-busy', 'true');
 eventUploadButton.querySelector('span').textContent = 'Đang nhập…';
 setStatus(eventStatus, 'pending', 'Đang đọc toàn bộ thư mục Drive và xử lý ảnh… Vui lòng giữ trang mở.');
 try {
  const data = await readResponse(await fetch('/events', {method: 'POST'}));
  const lines = [data.message || `Đã đọc ${data.photos_found} ảnh từ Drive. Thêm ${data.photos_uploaded} ảnh vào album.`, `Khuôn mặt đã nhận diện: ${data.faces_indexed}.`];
  if (data.photos_reindexed) lines.push(`Ảnh cũ đã xử lý lại: ${data.photos_reindexed}.`);
  if (data.photos_with_no_faces) lines.push(`Ảnh không có khuôn mặt: ${data.photos_with_no_faces}.`);
  if (data.duplicates_skipped) lines.push(`Ảnh đã có trong album: ${data.duplicates_skipped} (đã giữ nguyên).`);
  const failed = data.failed_photos || [];
  if (failed.length) lines.push(`Ảnh chưa xử lý được: ${failed.length}.`, ...failed.map(item => `${item.file}: ${item.error}`));
  setStatus(eventStatus, failed.length ? 'warning' : 'success', lines.join('\\n'));
  await refreshAlbum();
 } catch (error) {
  setStatus(eventStatus, 'error', `Chưa hoàn tất nhập ảnh: ${error.message}`);
 } finally {
  eventUploadButton.disabled = false;
  eventForm.removeAttribute('aria-busy');
  eventUploadButton.querySelector('span').textContent = 'Nhập toàn bộ ảnh';
 }
});
function resultCard(result, index) {
 const card = document.createElement('article');
 card.className = 'photo-card result-card';
 const preview = document.createElement('a');
 preview.href = result.photo_url;
 preview.target = '_blank';
 preview.rel = 'noopener noreferrer';
 const imageBox = document.createElement('div');
 imageBox.className = 'photo-image';
 const image = document.createElement('img');
 image.src = result.photo_url;
 image.alt = `Ảnh có bạn — ${index + 1}`;
 image.loading = 'lazy';
 image.decoding = 'async';
 imageBox.append(image);
 const caption = document.createElement('span');
 caption.className = 'photo-caption';
 caption.textContent = `Khoảnh khắc ${String(index + 1).padStart(2, '0')}`;
 const score = document.createElement('span');
 score.className = 'photo-score';
 score.textContent = `Độ khớp: ${Number(result.score).toFixed(3)}`;
 preview.append(imageBox, caption);
 const download = document.createElement('a');
 download.className = 'photo-download';
 download.href = result.download_url;
 download.setAttribute('download', '');
 download.textContent = '↓ Tải ảnh gốc';
 card.append(preview, score, download);
 return card;
}
searchForm.addEventListener('submit', async event => {
 event.preventDefault();
 if (searchButton.disabled) return;
 const payload = new FormData(searchForm);
 searchButton.disabled = true;
 queryInput.disabled = true;
 searchForm.setAttribute('aria-busy', 'true');
 searchButton.querySelector('span').textContent = 'Đang tìm…';
 resultsSection.hidden = false;
 resultCount.hidden = true;
 currentMatches = [];
 downloadPanel.hidden = true;
 downloadStatus.textContent = '';
 document.querySelector('#results').replaceChildren();
 setStatus(searchStatus, 'pending', 'Đang tìm những khoảnh khắc có bạn…');
 try {
  const data = await readResponse(await fetch('/search', {method: 'POST', body: payload}));
  const matches = data.matches || [];
  currentMatches = matches;
  downloadPanel.hidden = matches.length === 0;
  setStatus(searchStatus, matches.length ? 'success' : 'warning', data.message || `Tìm thấy ${matches.length} ảnh có bạn.`);
  resultCount.textContent = `${matches.length} ảnh`;
  resultCount.hidden = false;
  document.querySelector('#results').replaceChildren(...matches.map(resultCard));
 } catch (error) {
  setStatus(searchStatus, 'error', error.message);
 } finally {
  searchButton.disabled = false;
  queryInput.disabled = false;
  searchForm.removeAttribute('aria-busy');
  searchButton.querySelector('span').textContent = 'Tìm ảnh';
 }
});
downloadAllButton.addEventListener('click', async () => {
 if (!currentMatches.length || downloadAllButton.disabled) return;
 const photos = currentMatches.map(result => result.photo_path);
 downloadAllButton.disabled = true;
 setStatus(downloadStatus, 'pending', `Đang chuẩn bị ZIP chứa ${photos.length} ảnh gốc…`);
 try {
  const response = await fetch('/downloads/zip', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({photos})});
  if (!response.ok) await readResponse(response);
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement('a');
  link.href = url;
  link.download = 'anh-cua-ban.zip';
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
  setStatus(downloadStatus, 'success', `Đã chuẩn bị ${photos.length} ảnh. Trình duyệt đang tải tệp ZIP.`);
 } catch (error) {
  setStatus(downloadStatus, 'error', `Chưa tải được ZIP: ${error.message}`);
 } finally {
  downloadAllButton.disabled = false;
 }
});
document.querySelector('#close-results').addEventListener('click', () => { resultsSection.hidden = true; });
</script>
</body></html>"""
    return HTMLResponse(
        page.replace("__TOTAL_PHOTOS__", str(len(photos)))
        .replace("__DRIVE_FOLDER_URL__", DRIVE_FOLDER_URL)
        .replace(
            "__ALBUM_NOTE__",
            '<p class="section-description" style="margin-top:16px">Hiển thị 24 ảnh mới nhất trong album.</p>'
            if len(photos) > 24 else "",
        )
        .replace("__PHOTO_CARDS__", album)
    )


@app.get("/health")
def health():
    """
    Test server hoạt động.
    """
    return {"status": "ok"}


@app.post("/events")
def add_event_photos():
    """Import the fixed Drive folder in one operation; accept no upload input."""
    if not _import_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Đang nhập ảnh từ Drive. Vui lòng chờ hoàn tất.")
    try:
        return _import_drive_photos()
    finally:
        _import_lock.release()


def _import_drive_photos():
    try:
        photos = list_drive_photos()
    except DriveImportError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    report = {
        "source": DRIVE_FOLDER_URL,
        "photos_found": len(photos),
        "photos_uploaded": 0,
        "duplicates_skipped": 0,
        "photos_reindexed": 0,
        "photos_with_no_faces": 0,
        "faces_indexed": 0,
        "failed_photos": [],
    }
    if not photos:
        report["message"] = "Thư mục Drive chưa có ảnh JPG, JPEG hoặc PNG."
        return report
    try:
        runtime = _load_runtime()
    except (FileNotFoundError, RuntimeError, ImportError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    for photo in photos:
        try:
            contents = download_drive_photo(photo)
            digest = hashlib.sha256(contents).hexdigest()[:12]
            suffix = Path(photo.name).suffix.lower()
            # Drive allows names that are invalid on Windows. Keep a readable,
            # bounded name and use the content hash to distinguish originals.
            stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(photo.name).stem)
            stem = stem.strip(" .")[:100] or "photo"
            destination = EVENT_PHOTOS_DIR / f"{stem}__{digest}{suffix}"
            existing = next(iter(EVENT_PHOTOS_DIR.glob(f"*__{digest}{suffix}")), None)
            with _runtime_lock:
                if existing is not None and _photo_is_indexed(runtime["store"], existing):
                    report["duplicates_skipped"] += 1
                    continue
                # Decode/encode before saving so corrupt downloads leave no
                # broken image in the album. The face pipeline is unchanged.
                image = _decode_upload(contents)
                faces = runtime["encoder"].encode_image(image)
                if existing is not None:
                    destination = existing
                    report["photos_reindexed"] += 1
                else:
                    destination.write_bytes(contents)
                    report["photos_uploaded"] += 1
                for face in faces:
                    runtime["store"].add_face(
                        face["embedding"],
                        str(destination.relative_to(PROJECT_ROOT)),
                        face["bbox"],
                    )
                    report["faces_indexed"] += 1
                if not faces:
                    report["photos_with_no_faces"] += 1
        except Exception as exc:
            report["failed_photos"].append({"file": photo.name, "error": str(exc)})
    # Persist once for the complete import, including successful photos when
    # another Drive file cannot be downloaded or decoded.
    with _runtime_lock:
        runtime["store"].save(
            str(STORAGE_DIR / "faces.faiss"),
            str(STORAGE_DIR / "metadata.json"),
        )
    return report


@app.post("/search")
def search_photo(file: UploadFile):
    """
    User upload selfie query.

    Pipeline:

        uploaded image
        -> SearchEngine.search()
        -> matching photos

    Trả JSON cho frontend.
    """
    try:
        contents = file.file.read()
        image = _decode_upload(contents)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        runtime = _load_runtime()
    except (FileNotFoundError, RuntimeError, ImportError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        with _runtime_lock:
            results = runtime["engine"].search(image)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Không tìm thấy khuôn mặt rõ trong ảnh. Hãy thử một ảnh selfie chính diện và đủ sáng hơn.",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    matches = [_result_for_api(result) for result in results]
    response = {
        "matches": matches,
        "count": len(matches),
    }
    if not matches:
        response["message"] = "Chưa tìm thấy ảnh phù hợp trong event này. Hãy thử selfie khác với góc mặt hoặc ánh sáng khác."
    else:
        response["message"] = f"Đã tìm thấy {len(matches)} ảnh phù hợp."
    return response
