"""Small FastAPI UI/API for adding event photos and searching them."""

from __future__ import annotations

import hashlib
import html
import json
import threading
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

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


def _result_for_api(result: dict) -> dict:
    output = dict(result)
    relative = _relative_photo_path(str(result["photo_path"]))
    output["photo_url"] = "/event-photos/" + quote(relative)
    return output


@app.get("/", response_class=HTMLResponse)
def home():
    """Render the two small MVP user flows."""
    return HTMLResponse(
        """<!doctype html>
<html lang="vi">
<head><meta charset="utf-8"><title>Face Search</title>
<style>body{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem}section{border:1px solid #ddd;border-radius:12px;padding:1rem;margin:1rem 0}button{padding:.6rem 1rem}#results img{max-width:180px;max-height:140px;object-fit:cover;margin:.4rem;border-radius:8px}</style>
</head>
<body>
<h1>Face Search</h1>
<section><h2>Add Event Photos</h2>
<form id="event-form"><input type="file" name="files" accept="image/jpeg,image/png" multiple required><button>Upload</button></form>
<pre id="event-status"></pre></section>
<section><h2>Find My Photos</h2>
<form id="search-form"><input type="file" name="file" accept="image/jpeg,image/png" required><button>Search</button></form>
<p id="search-status"></p><div id="results"></div></section>
<script>
const eventForm=document.querySelector('#event-form');
eventForm.addEventListener('submit',async e=>{e.preventDefault();const r=await fetch('/events',{method:'POST',body:new FormData(eventForm)});document.querySelector('#event-status').textContent=JSON.stringify(await r.json(),null,2);});
const searchForm=document.querySelector('#search-form');
searchForm.addEventListener('submit',async e=>{e.preventDefault();const r=await fetch('/search',{method:'POST',body:new FormData(searchForm)});const data=await r.json();document.querySelector('#search-status').textContent=data.message||data.detail||('Matches: '+(data.count||0));const box=document.querySelector('#results');box.replaceChildren(...(data.matches||[]).map(x=>{const img=document.createElement('img');img.src=x.photo_url;img.alt=x.photo_path;img.title=x.photo_path;return img;}));});
</script></body></html>"""
    )


@app.get("/health")
def health():
    """
    Test server hoạt động.
    """
    return {"status": "ok"}


@app.post("/events")
async def add_event_photos(files: list[UploadFile] = File(...)):
    """Save and index one or more event photos immediately."""
    try:
        runtime = _load_runtime()
    except (FileNotFoundError, RuntimeError, ImportError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    uploaded = 0
    duplicates = 0
    failed = []
    no_faces = 0
    faces_indexed = 0
    with _runtime_lock:
        for upload in files:
            filename = Path(upload.filename or "upload.jpg").name
            suffix = Path(filename).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png"}:
                failed.append({"file": filename, "error": "Only JPG, JPEG, and PNG are supported"})
                continue
            try:
                contents = await upload.read()
                digest = hashlib.sha256(contents).hexdigest()[:12]
                stem = Path(filename).stem or "photo"
                destination = EVENT_PHOTOS_DIR / f"{stem}__{digest}{suffix}"
                if destination.is_file() or any(EVENT_PHOTOS_DIR.glob(f"*__{digest}{suffix}")):
                    duplicates += 1
                    continue
                destination.write_bytes(contents)
                uploaded += 1
                image = _decode_upload(contents)
                faces = runtime["encoder"].encode_image(image)
                if not faces:
                    no_faces += 1
                    continue
                for face in faces:
                    runtime["store"].add_face(
                        face["embedding"],
                        str(destination.relative_to(PROJECT_ROOT)),
                        face["bbox"],
                    )
                    faces_indexed += 1
            except Exception as exc:
                failed.append({"file": filename, "error": str(exc)})
        runtime["store"].save(
            str(STORAGE_DIR / "faces.faiss"),
            str(STORAGE_DIR / "metadata.json"),
        )
    return {
        "photos_uploaded": uploaded,
        "duplicates_skipped": duplicates,
        "photos_with_no_faces": no_faces,
        "faces_indexed": faces_indexed,
        "failed_photos": failed,
    }


@app.post("/search")
async def search_photo(file: UploadFile):
    """
    User upload selfie query.

    Pipeline:

        uploaded image
        -> SearchEngine.search()
        -> matching photos

    Trả JSON cho frontend.
    """
    try:
        contents = await file.read()
        image = _decode_upload(contents)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        runtime = _load_runtime()
    except (FileNotFoundError, RuntimeError, ImportError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
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
