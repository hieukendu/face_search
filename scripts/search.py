"""Run a selfie query against the persisted event-photo index."""

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
from src.search_engine import SearchEngine
from src.vector_store import VectorStore


def load_query_image(path):
    """
    Đọc selfie query.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"Unable to read query image: {path}")
    return image


def print_results(results):
    """
    In kết quả:

    0.93  IMG_123.jpg
    0.89  IMG_891.jpg
    0.86  IMG_182.jpg
    """
    for rank, result in enumerate(results or [], start=1):
        print(f"{rank}. {float(result['score']):.4f}  {result['photo_path']}")


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


def _query_paths(query: str | None, query_folder: str):
    if query:
        return [Path(query)]
    folder = Path(query_folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Query folder does not exist: {query_folder}")
    extensions = {".jpg", ".jpeg", ".png"}
    return sorted(
        [path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in extensions],
        key=lambda path: str(path).lower(),
    )


def main():
    """
    1. Load SCRFD + ArcFace
    2. Load FAISS index
    3. Load metadata
    4. Khởi tạo SearchEngine
    5. Load selfie
    6. Search
    7. Print results
    """
    parser = argparse.ArgumentParser(description="Search event photos with a selfie")
    parser.add_argument("--query")
    parser.add_argument("--query-folder", default="data/queries/smoke_queries")
    parser.add_argument("--detector-path", default="models/scrfd_10g.onnx")
    parser.add_argument("--embedder-path", default="models/arcface_w600k_r50.onnx")
    parser.add_argument("--index-path", default="storage/faces.faiss")
    parser.add_argument("--metadata-path", default="storage/metadata.json")
    parser.add_argument("--config", default="storage/best_config.json")
    parser.add_argument("--top-k", type=int, default=None)
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
    store.load(args.index_path, args.metadata_path)
    engine = SearchEngine(
        encoder,
        store,
        similarity_threshold=float(config.get("similarity_threshold", 0.5)),
        top_k=int(args.top_k if args.top_k is not None else config.get("top_k", 100)),
    )

    query_paths = _query_paths(args.query, args.query_folder)
    if not query_paths:
        print("No query images found.")
        return

    for query_path in query_paths:
        print(f"query: {query_path}")
        started = time.perf_counter()
        try:
            results = engine.search(load_query_image(query_path))
        except (ValueError, RuntimeError) as exc:
            print(f"query error: {exc}")
            print(f"query time seconds: {time.perf_counter() - started:.3f}")
            continue
        print_results(results)
        if not results:
            print("No matching photos.")
        print(f"matches: {len(results)}")
        print(f"query time seconds: {time.perf_counter() - started:.3f}")


if __name__ == "__main__":
    main()
