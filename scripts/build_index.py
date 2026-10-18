"""Build a persistent FAISS index from event photos."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.face_encoder import FaceEncoder
from src.vector_store import VectorStore


def load_image(path):
    """
    Đọc ảnh từ disk.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"Unable to read image: {path}")
    return image


def get_image_paths(folder):
    """
    Lấy toàn bộ:

        .jpg
        .jpeg
        .png

    trong event folder.
    """
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {folder}")
    extensions = {".jpg", ".jpeg", ".png"}
    return sorted(
        [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions],
        key=lambda path: str(path).lower(),
    )


def build_index(
    image_folder,
    encoder,
    vector_store
):
    """
    Offline indexing pipeline.

    Với mỗi ảnh:

        image
        -> encoder.encode_image()
        -> lấy tất cả faces
        -> vector_store.add_face()

    Sau khi xong:
        save FAISS index + metadata.
    """
    started = time.perf_counter()
    image_paths = get_image_paths(image_folder)
    processed = 0
    failed = 0
    no_faces = 0
    faces_indexed = 0
    failures = []

    for path in image_paths:
        try:
            image = load_image(path)
            faces = encoder.encode_image(image)
            processed += 1
            if not faces:
                no_faces += 1
                continue
            for face in faces:
                vector_store.add_face(
                    face["embedding"],
                    str(path),
                    face["bbox"],
                )
                faces_indexed += 1
        except Exception as exc:
            failed += 1
            failures.append({"path": str(path), "error": str(exc)})

    return {
        "images_processed": processed,
        "images_failed": failed,
        "images_with_no_faces": no_faces,
        "faces_indexed": faces_indexed,
        "indexing_time_seconds": time.perf_counter() - started,
        "failures": failures,
    }


def _load_config(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return payload


def main():
    """
    1. Load SCRFD + ArcFace
    2. Tạo VectorStore
    3. Load event_photos/
    4. Encode tất cả faces
    5. Build FAISS
    6. Save storage/
    """
    parser = argparse.ArgumentParser(description="Build the event-photo face index")
    parser.add_argument("--image-folder", default="data/event_photos")
    parser.add_argument("--detector-path", default="models/scrfd_10g.onnx")
    parser.add_argument("--embedder-path", default="models/arcface_w600k_r50.onnx")
    parser.add_argument("--index-path", default="storage/faces.faiss")
    parser.add_argument("--metadata-path", default="storage/metadata.json")
    parser.add_argument("--config", default="storage/best_config.json")
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    det_size = int(config.get("det_size", 640))
    encoder = FaceEncoder(
        args.detector_path,
        args.embedder_path,
        det_size=(det_size, det_size),
        det_threshold=float(config.get("det_threshold", 0.5)),
        min_face_size=float(config.get("min_face_size", 20)),
    )
    store = VectorStore(dimension=int(config.get("embedding_dim", 512)))
    stats = build_index(args.image_folder, encoder, store)
    store.save(args.index_path, args.metadata_path)

    print(f"images processed: {stats['images_processed']}")
    print(f"images failed: {stats['images_failed']}")
    print(f"images with no faces: {stats['images_with_no_faces']}")
    print(f"faces indexed: {stats['faces_indexed']}")
    print(f"indexing time seconds: {stats['indexing_time_seconds']:.3f}")
    print(f"FAISS vectors: {len(store)}")
    print(f"metadata entries: {len(store.metadata)}")
    for failure in stats["failures"]:
        print(f"failed: {failure['path']}: {failure['error']}", file=sys.stderr)


if __name__ == "__main__":
    main()
