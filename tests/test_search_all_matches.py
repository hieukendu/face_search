"""Tests for threshold-complete production retrieval."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.search_engine import SearchEngine
from src.vector_store import VectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _StubEncoder:
    def __init__(self, embedding):
        self.embedding = np.asarray(embedding, dtype=np.float32)

    def encode_query(self, _image):
        return self.embedding


class SearchAllMatchesTest(unittest.TestCase):
    def test_threshold_search_is_not_capped_by_top_k_and_deduplicates(self):
        dimension = 8
        query = np.zeros(dimension, dtype=np.float32)
        query[0] = 1.0
        store = VectorStore(dimension)

        for index in range(120):
            store.add_face(query, f"photo-{index:03d}.jpg", [0, 0, 10, 10])
        for index in range(10):
            store.add_face(query, "photo-000.jpg", [index, 0, index + 10, 10])

        # This score is exactly the public threshold before float32 rounding.
        boundary = np.asarray(
            [0.23, np.sqrt(1.0 - 0.23**2), 0, 0, 0, 0, 0, 0],
            dtype=np.float32,
        )
        store.add_face(boundary, "boundary.jpg", [0, 0, 10, 10])
        store.add_face(np.asarray([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32), "below.jpg", [0, 0, 10, 10])

        self.assertEqual(store.index.ntotal, len(store.metadata))

        # top_k=1 must not truncate a threshold-aware range search.
        candidates = store.search(query, top_k=1, score_threshold=0.23)
        self.assertEqual(len(candidates), 131)
        self.assertNotIn("below.jpg", {item["photo_path"] for item in candidates})

        engine = SearchEngine(
            _StubEncoder(query),
            store,
            similarity_threshold=0.23,
            top_k=100,
        )
        results = engine.search(None)
        self.assertGreater(len(results), 100)
        self.assertEqual(len(results), 121)
        self.assertEqual(len(results), len({item["photo_path"] for item in results}))
        self.assertTrue(all(float(item["score"]) >= 0.23 for item in results))
        self.assertTrue(
            all(
                results[index]["score"] >= results[index + 1]["score"]
                for index in range(len(results) - 1)
            )
        )
        self.assertIn("boundary.jpg", {item["photo_path"] for item in results})

        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory) / "faces.faiss"
            metadata_path = Path(directory) / "metadata.json"
            store.save(str(index_path), str(metadata_path))
            reloaded = VectorStore(dimension).load(str(index_path), str(metadata_path))
            self.assertEqual(reloaded.index.ntotal, len(reloaded.metadata))
            reloaded_results = SearchEngine(
                _StubEncoder(query),
                reloaded,
                similarity_threshold=0.23,
                top_k=100,
            ).search(None)
            self.assertEqual(len(reloaded_results), 121)

    @unittest.skipUnless(
        (PROJECT_ROOT / "models" / "scrfd_10g.onnx").is_file()
        and (PROJECT_ROOT / "models" / "arcface_w600k_r50.onnx").is_file()
        and (PROJECT_ROOT / "storage" / "faces.faiss").is_file(),
        "production models/index are not available",
    )
    def test_existing_smoke_queries_still_search_without_top_k_truncation(self):
        config = json.loads((PROJECT_ROOT / "storage" / "best_config.json").read_text())
        from src.face_encoder import FaceEncoder

        encoder = FaceEncoder(
            str(PROJECT_ROOT / "models" / config["detector_model"]),
            str(PROJECT_ROOT / "models" / config["embedding_model"]),
            det_size=(int(config["det_size"]), int(config["det_size"])),
            det_threshold=float(config["det_threshold"]),
            min_face_size=float(config["min_face_size"]),
        )
        store = VectorStore(int(config["embedding_dim"])).load(
            str(PROJECT_ROOT / "storage" / "faces.faiss"),
            str(PROJECT_ROOT / "storage" / "metadata.json"),
        )
        self.assertEqual(store.index.ntotal, len(store.metadata))
        engine = SearchEngine(
            encoder,
            store,
            similarity_threshold=float(config["similarity_threshold"]),
            top_k=int(config["top_k"]),
        )

        query_paths = sorted((PROJECT_ROOT / "data" / "queries" / "smoke_queries").glob("*.jpg"))
        self.assertEqual(len(query_paths), 10)
        queries_with_matches = 0
        for query_path in query_paths:
            image = cv2.imread(str(query_path), cv2.IMREAD_COLOR)
            results = engine.search(image)
            if results:
                queries_with_matches += 1
            self.assertTrue(all(item["score"] >= 0.23 for item in results))
            self.assertEqual(len(results), len({item["photo_path"] for item in results}))
            self.assertTrue(
                all(
                    results[index]["score"] >= results[index + 1]["score"]
                    for index in range(len(results) - 1)
                )
            )
        self.assertGreaterEqual(queries_with_matches, 9)


if __name__ == "__main__":
    unittest.main()
