"""Exercise input/output endpoints with real FAISS and isolated local storage."""

import hashlib
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

import app
from src.drive_import import DRIVE_FOLDER_URL, DriveImportError, DrivePhoto
from src.search_engine import SearchEngine
from src.vector_store import VectorStore


def photo_bytes(value=100):
    ok, encoded = cv2.imencode(".png", np.full((32, 32, 3), value, dtype=np.uint8))
    assert ok
    return encoded.tobytes()


class Encoder:
    vector = np.asarray([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    def encode_image(self, image):
        if not image.any():
            return []
        return [
            {"embedding": self.vector, "bbox": [0, 0, 12, 12]},
            {"embedding": self.vector, "bbox": [12, 12, 24, 24]},
        ]

    def encode_query(self, image):
        if not image.any():
            raise ValueError("no face")
        return self.vector


class AppIOTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="face-search-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.album = self.root / "data" / "event_photos"
        self.album.mkdir(parents=True)
        self.storage = self.root / "storage"
        self.encoder = Encoder()
        self.store = VectorStore(8)
        self.runtime = {
            "encoder": self.encoder,
            "store": self.store,
            "engine": SearchEngine(self.encoder, self.store, similarity_threshold=.23, top_k=1),
        }
        for name, value in (
            ("PROJECT_ROOT", self.root), ("EVENT_PHOTOS_DIR", self.album),
            ("STORAGE_DIR", self.storage), ("_runtime", self.runtime),
        ):
            patcher = patch.object(app, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = TestClient(app.app)
        self.addCleanup(self.client.close)

    def import_photos(self, contents):
        photos = [DrivePhoto(str(i), name) for i, (name, _) in enumerate(contents)]
        by_id = {photo.file_id: value for photo, (_, value) in zip(photos, contents)}

        def download(photo):
            value = by_id[photo.file_id]
            if isinstance(value, Exception):
                raise value
            return value

        with patch.object(app, "list_drive_photos", return_value=photos):
            with patch.object(app, "download_drive_photo", side_effect=download):
                return self.client.post("/events")

    def indexed_photo(self, name, contents=None):
        path = self.album / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents if contents is not None else photo_bytes())
        self.store.add_face(self.encoder.vector, str(path.relative_to(self.root)), [0, 0, 10, 10])
        return path

    def test_import_saves_all_faces_once_and_survives_reload(self):
        with patch.object(self.store, "save", wraps=self.store.save) as save:
            response = self.import_photos([("one.png", photo_bytes(100)), ("two.png", photo_bytes(110))])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["photos_found"], 2)
        self.assertEqual(response.json()["photos_uploaded"], 2)
        self.assertEqual(response.json()["faces_indexed"], 4)
        self.assertEqual(response.json()["source"], DRIVE_FOLDER_URL)
        save.assert_called_once()
        reloaded = VectorStore(8).load(str(self.storage / "faces.faiss"), str(self.storage / "metadata.json"))
        self.assertEqual(len(reloaded), 4)
        self.assertEqual(len(list(self.album.iterdir())), 2)

    def test_reimport_and_content_duplicates_do_not_add_vectors(self):
        data = [("one.png", photo_bytes()), ("renamed.png", photo_bytes())]
        first = self.import_photos(data).json()
        second = self.import_photos(data).json()
        self.assertEqual(first["photos_uploaded"], 1)
        self.assertEqual(first["duplicates_skipped"], 1)
        self.assertEqual(second["photos_uploaded"], 0)
        self.assertEqual(second["duplicates_skipped"], 2)
        self.assertEqual(len(self.store), 2)

    def test_no_faces_corrupt_and_failed_download_do_not_stop_other_photos(self):
        response = self.import_photos([
            ("good.png", photo_bytes()), ("blank.png", photo_bytes(0)),
            ("corrupt.png", b"invalid"), ("unreadable.png", DriveImportError("denied")),
        ])
        data = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["photos_uploaded"], 2)
        self.assertEqual(data["photos_with_no_faces"], 1)
        self.assertEqual(data["faces_indexed"], 2)
        self.assertEqual(len(data["failed_photos"]), 2)
        self.assertEqual(len(list(self.album.iterdir())), 2)
        self.assertFalse(any("corrupt" in p.name for p in self.album.iterdir()))

    def test_recovers_a_photo_saved_without_index_metadata(self):
        contents = photo_bytes()
        digest = hashlib.sha256(contents).hexdigest()[:12]
        (self.album / f"previous__{digest}.png").write_bytes(contents)
        data = self.import_photos([("new-name.png", contents)]).json()
        self.assertEqual(data["photos_reindexed"], 1)
        self.assertEqual(data["photos_uploaded"], 0)
        self.assertEqual(len(self.store), 2)

    def test_remote_filename_is_sanitized_for_windows(self):
        response = self.import_photos([('unsafe:<>"?|name.png', photo_bytes())])
        self.assertEqual(response.json()["failed_photos"], [])
        self.assertEqual(len(list(self.album.iterdir())), 1)
        self.assertFalse(any(c in next(self.album.iterdir()).name for c in '<>:"|?*'))

    def test_empty_folder_does_not_require_models(self):
        with patch.object(app, "_load_runtime", side_effect=AssertionError("must not load")):
            response = self.import_photos([])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["photos_found"], 0)
        self.assertIn("message", response.json())

    def test_drive_error_is_reported_and_import_lock_is_released(self):
        with patch.object(app, "list_drive_photos", side_effect=DriveImportError("no access")):
            response = self.client.post("/events")
        self.assertEqual(response.status_code, 502)
        self.assertEqual(self.import_photos([]).status_code, 200)

    def test_missing_models_is_reported_without_saving_photos(self):
        with patch.object(app, "_load_runtime", side_effect=FileNotFoundError("model missing")):
            response = self.import_photos([("one.png", photo_bytes())])
        self.assertEqual(response.status_code, 503)
        self.assertEqual(list(self.album.iterdir()), [])

    def test_concurrent_import_is_rejected(self):
        app._import_lock.acquire()
        try:
            self.assertEqual(self.client.post("/events").status_code, 409)
            self.assertEqual(self.client.get("/health").status_code, 200)
        finally:
            app._import_lock.release()

    def test_event_endpoint_has_no_upload_or_source_override(self):
        schema = self.client.get("/openapi.json").json()
        self.assertNotIn("requestBody", schema["paths"]["/events"]["post"])
        with patch.object(app, "list_drive_photos", return_value=[]) as listing:
            response = self.client.post("/events", json={"url": "https://example.com/other"})
        listing.assert_called_once_with()
        self.assertEqual(response.json()["source"], DRIVE_FOLDER_URL)

    def test_search_downloads_all_results_including_more_than_100_photos(self):
        for i in range(121):
            self.indexed_photo(f"photo-{i}.png", photo_bytes(i + 1))
        # Multiple matching faces in one original must still give one download.
        self.store.add_face(self.encoder.vector, str(Path("data") / "event_photos" / "photo-0.png"), [10, 10, 20, 20])
        response = self.client.post("/search", files={"file": ("selfie.png", photo_bytes(), "image/png")})
        data = response.json()
        self.assertEqual(data["count"], 121)
        self.assertEqual(len(self.store), 122)  # Query itself was not indexed.
        matches = data["matches"]
        original = self.root / matches[0]["photo_path"]
        single = self.client.get(matches[0]["download_url"])
        self.assertEqual(single.content, original.read_bytes())
        self.assertIn("attachment", single.headers["content-disposition"])
        zipped = self.client.post("/downloads/zip", json={"photos": [m["photo_path"] for m in matches]})
        self.assertEqual(zipped.status_code, 200)
        self.assertEqual(zipped.headers["content-type"], "application/zip")
        with zipfile.ZipFile(io.BytesIO(zipped.content)) as bundle:
            self.assertEqual(len(bundle.namelist()), 121)
            self.assertIsNone(bundle.testzip())
            for match in matches:
                path = self.root / match["photo_path"]
                self.assertEqual(bundle.read("anh-cua-ban/" + path.name), path.read_bytes())

    def test_zip_deduplicates_and_preserves_subfolders_unicode_and_names(self):
        first = self.indexed_photo("nhóm một/ảnh.png", photo_bytes(100))
        second = self.indexed_photo("nhóm hai/ảnh.png", photo_bytes(110))
        paths = [str(first.relative_to(self.root)), str(second.relative_to(self.root)), str(first)]
        response = self.client.post("/downloads/zip", json={"photos": paths})
        self.assertEqual(response.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
            self.assertEqual(set(bundle.namelist()), {"anh-cua-ban/nhóm một/ảnh.png", "anh-cua-ban/nhóm hai/ảnh.png"})

    def test_zip_temporary_file_is_closed_after_response(self):
        path = self.indexed_photo("photo.png")
        archive = tempfile.TemporaryFile()
        with patch.object(app.tempfile, "TemporaryFile", return_value=archive):
            response = self.client.post("/downloads/zip", json={"photos": [str(path)]})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(archive.closed)

    def test_zip_temporary_file_is_closed_on_build_error(self):
        path = self.indexed_photo("photo.png")
        archive = tempfile.TemporaryFile()
        with patch.object(app.tempfile, "TemporaryFile", return_value=archive):
            with patch.object(app.zipfile.ZipFile, "write", side_effect=OSError("disk error")):
                with self.assertRaises(OSError):
                    self.client.post("/downloads/zip", json={"photos": [str(path)]})
        self.assertTrue(archive.closed)

    def test_zip_rejects_missing_and_empty_selection(self):
        self.assertEqual(self.client.post("/downloads/zip", json={"photos": []}).status_code, 422)
        self.assertEqual(self.client.post("/downloads/zip", json={"photos": ["data/event_photos/missing.png"]}).status_code, 404)

    def test_download_cannot_read_outside_album_or_non_images(self):
        (self.root / "secret.png").write_bytes(b"secret")
        (self.album / "notes.txt").write_text("not an image")
        for path in ("secret.png", "data/event_photos/../../secret.png", "data/event_photos/notes.txt", "\x00"):
            with self.subTest(path=path):
                response = self.client.post("/downloads/zip", json={"photos": [path]})
                self.assertEqual(response.status_code, 400)
        response = self.client.get("/downloads/photo/%2E%2E/%2E%2E/secret.png")
        self.assertEqual(response.status_code, 400)

    def test_search_empty_and_unreadable_selfie(self):
        self.assertEqual(self.client.post("/search", files={"file": ("empty.png", b"", "image/png")}).status_code, 400)
        self.assertEqual(self.client.post("/search", files={"file": ("invalid.png", b"invalid", "image/png")}).status_code, 400)
        self.assertEqual(self.client.post("/search", files={"file": ("blank.png", photo_bytes(0), "image/png")}).status_code, 400)
        response = self.client.post("/search", files={"file": ("selfie.png", photo_bytes(), "image/png")})
        self.assertEqual(response.json()["matches"], [])

    def test_ui_has_fixed_drive_source_selfie_and_both_download_options(self):
        page = self.client.get("/").text
        self.assertIn(DRIVE_FOLDER_URL, page)
        self.assertNotIn('id="event-files"', page)
        self.assertIn('id="query-file"', page)
        self.assertIn('id="download-all"', page)
        self.assertIn("result.download_url", page)
        self.assertIn("photo-score", page)
        self.assertNotIn("__DRIVE_FOLDER_URL__", page)


if __name__ == "__main__":
    unittest.main()
