"""Local integration test with the downloaded SCRFD/ArcFace models and real photos.

Drive transport is replaced with an already downloaded original. All decoding,
face detection, embeddings, FAISS, search, persistence, and ZIP bytes are real.
"""

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import app
from src.drive_import import DrivePhoto


ROOT = Path(__file__).resolve().parents[1]


def _find_real_photo():
    metadata_path = ROOT / "storage" / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for item in metadata.values():
            path = ROOT / item["photo_path"]
            if path.is_file():
                return path
    return None


REAL_PHOTO = _find_real_photo()


@unittest.skipUnless(
    REAL_PHOTO is not None
    and (ROOT / "models" / "scrfd_10g.onnx").is_file()
    and (ROOT / "models" / "arcface_w600k_r50.onnx").is_file(),
    "download both models and import Drive photos to run the real-model integration test",
)
class RealModelsIOTest(unittest.TestCase):
    def test_real_embedding_search_reload_and_original_downloads(self):
        original = REAL_PHOTO.read_bytes()
        config_bytes = (ROOT / "storage" / "best_config.json").read_bytes()
        with tempfile.TemporaryDirectory(prefix="face-search-real-test-") as directory:
            root = Path(directory)
            album = root / "data" / "event_photos"
            album.mkdir(parents=True)
            storage = root / "storage"
            storage.mkdir()
            (storage / "best_config.json").write_bytes(config_bytes)
            with (
                patch.object(app, "PROJECT_ROOT", root),
                patch.object(app, "EVENT_PHOTOS_DIR", album),
                patch.object(app, "STORAGE_DIR", storage),
                patch.object(app, "_runtime", None),
                patch.object(app, "list_drive_photos", return_value=[DrivePhoto("real-fixture", "real.png")]),
                patch.object(app, "download_drive_photo", return_value=original),
                TestClient(app.app) as client,
            ):
                imported = client.post("/events")
                self.assertEqual(imported.status_code, 200)
                self.assertEqual(imported.json()["failed_photos"], [])
                self.assertGreater(imported.json()["faces_indexed"], 0)
                vectors = len(app._runtime["store"])
                # This is the real event image as query, so the indexed original
                # must be returned. This checks functionality, not labeled accuracy.
                result = client.post("/search", files={"file": ("query.png", original, "image/png")})
                self.assertEqual(result.status_code, 200)
                matches = result.json()["matches"]
                self.assertEqual(len(matches), 1)
                self.assertGreaterEqual(matches[0]["score"], .23)
                self.assertEqual(len(app._runtime["store"]), vectors)
                single = client.get(matches[0]["download_url"])
                self.assertEqual(single.content, original)
                zipped = client.post("/downloads/zip", json={"photos": [m["photo_path"] for m in matches]})
                self.assertEqual(zipped.status_code, 200)
                with zipfile.ZipFile(io.BytesIO(zipped.content)) as archive:
                    self.assertEqual(len(archive.namelist()), 1)
                    self.assertEqual(archive.read(archive.namelist()[0]), original)
                # A fresh runtime must load the same saved vectors/config.
                app._runtime = None
                after_restart = client.post("/search", files={"file": ("query.png", original, "image/png")})
                self.assertEqual(after_restart.json()["count"], 1)
                self.assertEqual(len(app._runtime["store"]), vectors)
                repeated = client.post("/events").json()
                self.assertEqual(repeated["duplicates_skipped"], 1)
                self.assertEqual(repeated["faces_indexed"], 0)
                self.assertEqual(len(app._runtime["store"]), vectors)
                self.assertEqual((storage / "best_config.json").read_bytes(), config_bytes)


if __name__ == "__main__":
    unittest.main()
