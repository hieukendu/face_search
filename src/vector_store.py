"""Persistent FAISS storage for normalized face embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover - exercised when setup is incomplete
    faiss = None


class VectorStore:
    def __init__(
        self,
        dimension: int = 512
    ):
        """
        Khởi tạo FAISS IndexFlatIP.

        Vì embedding đã L2-normalize:

            inner product
            =
            cosine similarity
        """
        if faiss is None:
            raise ImportError("faiss-cpu is required. Install dependencies from requirements.txt.")
        self.dimension = int(dimension)
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata: dict[int, dict] = {}

    def _normalize_embeddings(self, embeddings) -> np.ndarray:
        array = np.asarray(embeddings, dtype=np.float32)
        if array.size == 0:
            return np.empty((0, self.dimension), dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != self.dimension:
            raise ValueError(
                f"Embeddings must have shape (N, {self.dimension}), got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("Embeddings must contain only finite values")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValueError("Embeddings must have non-zero L2 norm")
        array = array / norms
        return np.ascontiguousarray(array, dtype=np.float32)

    @staticmethod
    def _metadata_item(item):
        if isinstance(item, dict):
            if "photo_path" not in item:
                raise ValueError("metadata item must contain photo_path")
            photo_path = item["photo_path"]
            bbox = item.get("bbox")
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            photo_path, bbox = item[0], item[1]
        else:
            raise ValueError("metadata items must be dicts or (photo_path, bbox) pairs")
        if photo_path is None:
            raise ValueError("photo_path must not be empty")
        if bbox is None:
            normalized_bbox = None
        else:
            normalized_bbox = [float(value) for value in bbox]
            if len(normalized_bbox) != 4:
                raise ValueError("bbox must contain four coordinates")
        return {"photo_path": str(photo_path), "bbox": normalized_bbox}

    def add_face(
        self,
        embedding: np.ndarray,
        photo_path: str,
        bbox
    ):
        """
        Add một embedding vào FAISS.

        Đồng thời lưu metadata:

            vector ID
            -> photo_path
            -> bbox

        Ví dụ:

            FAISS ID 51
            -> IMG_0012.jpg
            -> [x1,y1,x2,y2]
        """
        normalized = self._normalize_embeddings(embedding)
        if normalized.shape[0] != 1:
            raise ValueError("add_face expects one embedding vector")
        metadata_item = self._metadata_item({"photo_path": photo_path, "bbox": bbox})
        vector_id = int(self.index.ntotal)
        self.index.add(normalized)
        self.metadata[vector_id] = metadata_item
        return vector_id

    def add_faces(
        self,
        embeddings,
        metadata
    ):
        """
        Add nhiều face cùng lúc.

        Dùng khi build index cho cả event.

        Nhanh hơn gọi add_face() hàng nghìn lần.
        """
        normalized = self._normalize_embeddings(embeddings)
        if isinstance(metadata, dict):
            ordered_metadata = [metadata[key] for key in sorted(metadata, key=lambda value: int(value))]
        else:
            ordered_metadata = list(metadata)
        if len(normalized) != len(ordered_metadata):
            raise ValueError("embeddings and metadata must have the same length")
        if not ordered_metadata:
            return []

        first_id = int(self.index.ntotal)
        self.index.add(normalized)
        ids = []
        for offset, item in enumerate(ordered_metadata):
            vector_id = first_id + offset
            self.metadata[vector_id] = self._metadata_item(item)
            ids.append(vector_id)
        return ids

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 100,
        score_threshold: float | None = None,
    ):
        """
        Search query embedding trong FAISS.

        FAISS tính similarity giữa query
        với embeddings trong index.

        When ``score_threshold`` is provided, use FAISS range search and
        return every vector whose inner-product score is at least that
        threshold. ``top_k`` is intentionally ignored in that mode; the
        production search path must not truncate valid matches.

        Without a threshold, preserve the legacy top-k search behavior for
        callers that use VectorStore directly.

        Output ví dụ:

        [
            {
                "id": 124,
                "score": 0.91,
                "photo_path": "...",
                "bbox": [...]
            },
            ...
        ]
        """
        if self.index.ntotal == 0:
            return []
        normalized = self._normalize_embeddings(query_embedding)
        if normalized.shape[0] != 1:
            raise ValueError("search expects one query embedding")

        if score_threshold is not None:
            threshold = float(score_threshold)
            if not np.isfinite(threshold):
                raise ValueError("score_threshold must be finite")
            # FAISS range search uses a strict distance comparison. Lower the
            # float32 radius by one representable step so a score exactly on
            # the public >= threshold boundary is not lost.
            radius = float(
                np.nextafter(
                    np.float32(threshold),
                    np.float32(-np.inf),
                )
            )
            limits, scores, ids = self.index.range_search(normalized, radius)
            start = int(limits[0])
            end = int(limits[1])
            scores = scores[start:end]
            ids = ids[start:end]
        else:
            if int(top_k) <= 0:
                return []
            limit = min(int(top_k), int(self.index.ntotal))
            scores, ids = self.index.search(normalized, limit)
            scores = scores[0]
            ids = ids[0]

        results = []
        for score, vector_id in zip(scores, ids):
            vector_id = int(vector_id)
            if vector_id < 0:
                continue
            item = self.metadata.get(vector_id)
            if item is None:
                raise RuntimeError(f"Missing metadata for FAISS vector ID {vector_id}")
            results.append(
                {
                    "id": vector_id,
                    "score": float(score),
                    "photo_path": item["photo_path"],
                    "bbox": item.get("bbox"),
                }
            )
        return results

    def save(
        self,
        index_path: str,
        metadata_path: str
    ):
        """
        Save:

            FAISS index
                -> faces.faiss

            metadata
                -> metadata.json
        """
        index_file = Path(index_path)
        metadata_file = Path(metadata_path)
        index_file.parent.mkdir(parents=True, exist_ok=True)
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        if int(self.index.ntotal) != len(self.metadata):
            raise RuntimeError("FAISS count and metadata count do not match")
        faiss.write_index(self.index, str(index_file))
        serializable = {
            str(vector_id): {
                "photo_path": item["photo_path"],
                "bbox": item.get("bbox"),
            }
            for vector_id, item in sorted(self.metadata.items())
        }
        with metadata_file.open("w", encoding="utf-8") as handle:
            json.dump(serializable, handle, ensure_ascii=False, indent=2)

    def load(
        self,
        index_path: str,
        metadata_path: str
    ):
        """
        Load FAISS index và metadata
        đã build trước đó.
        """
        if faiss is None:
            raise ImportError("faiss-cpu is required. Install dependencies from requirements.txt.")
        index_file = Path(index_path)
        metadata_file = Path(metadata_path)
        if not index_file.is_file():
            raise FileNotFoundError(f"FAISS index does not exist: {index_path}")
        if not metadata_file.is_file():
            raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")
        loaded_index = faiss.read_index(str(index_file))
        if int(loaded_index.d) != self.dimension:
            raise ValueError(
                f"Index dimension {loaded_index.d} does not match store dimension {self.dimension}"
            )
        try:
            with metadata_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to read metadata: {exc}") from exc

        if isinstance(payload, dict) and "metadata" in payload:
            payload = payload["metadata"]
        if not isinstance(payload, dict):
            raise ValueError("Metadata must be a JSON object keyed by FAISS ID")
        parsed_metadata = {}
        for key, item in payload.items():
            try:
                vector_id = int(key)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid FAISS metadata ID: {key}") from exc
            parsed_metadata[vector_id] = self._metadata_item(item)

        if int(loaded_index.ntotal) != len(parsed_metadata):
            raise RuntimeError("FAISS count and metadata count do not match")
        expected_ids = set(range(int(loaded_index.ntotal)))
        if set(parsed_metadata) != expected_ids:
            raise RuntimeError("FAISS metadata IDs are not contiguous")
        self.index = loaded_index
        self.metadata = parsed_metadata
        return self

    def __len__(self):
        """
        Trả về số face embeddings
        hiện đang nằm trong index.
        """
        return int(self.index.ntotal)
