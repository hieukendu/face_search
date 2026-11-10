"""Drive listing, filtering, and temporary-download cleanup."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.drive_import import (
    DRIVE_FOLDER_URL,
    DriveImportError,
    DrivePhoto,
    download_drive_photo,
    list_drive_photos,
)


class DriveImportTest(unittest.TestCase):
    def test_lists_all_photos_in_subfolders_without_a_50_or_100_file_cap(self):
        files = [SimpleNamespace(id=str(i), path=f"sub/ảnh {i}.PNG") for i in range(151)]
        files.extend([
            SimpleNamespace(id="0", path="same-photo.PNG"),
            SimpleNamespace(id="text", path="notes.txt"),
            SimpleNamespace(id="pdf", path="document.pdf"),
        ])
        with patch("src.drive_import.gdown.download_folder", return_value=files) as listing:
            photos = list_drive_photos()
        self.assertEqual(len(photos), 151)
        self.assertEqual(photos[0].name, "ảnh 0.PNG")
        listing.assert_called_once_with(
            url=DRIVE_FOLDER_URL, skip_download=True, quiet=True, use_cookies=False,
        )

    def test_empty_folder(self):
        with patch("src.drive_import.gdown.download_folder", return_value=[]):
            self.assertEqual(list_drive_photos(), [])

    def test_unreadable_folder_is_an_error_not_an_empty_success(self):
        for failure in (None, RuntimeError("permission denied")):
            with self.subTest(failure=failure):
                options = {"side_effect": failure} if isinstance(failure, Exception) else {"return_value": failure}
                with patch("src.drive_import.gdown.download_folder", **options):
                    with self.assertRaises(DriveImportError):
                        list_drive_photos()

    def test_download_keeps_original_bytes_and_removes_temporary_files(self):
        destinations = []

        def download(**kwargs):
            destination = Path(kwargs["output"])
            destinations.append(destination)
            self.assertEqual(destination.name, "photo.png")
            self.assertFalse(kwargs["use_cookies"])
            destination.write_bytes(b"original bytes")
            return str(destination)

        with patch("src.drive_import.gdown.download", side_effect=download):
            data = download_drive_photo(DrivePhoto("file-id", "../../unsafe.png"))
        self.assertEqual(data, b"original bytes")
        self.assertFalse(destinations[0].parent.exists())

    def test_download_failure_also_removes_partial_files(self):
        destinations = []

        def download(**kwargs):
            destination = Path(kwargs["output"])
            destinations.append(destination)
            (destination.parent / "partial.part").write_bytes(b"partial")
            raise OSError("network interrupted")

        with patch("src.drive_import.gdown.download", side_effect=download):
            with self.assertRaises(DriveImportError):
                download_drive_photo(DrivePhoto("file-id", "photo.png"))
        self.assertFalse(destinations[0].parent.exists())

    def test_missing_download_content_is_an_error(self):
        with patch("src.drive_import.gdown.download", return_value=None):
            with self.assertRaises(DriveImportError):
                download_drive_photo(DrivePhoto("file-id", "photo.png"))


if __name__ == "__main__":
    unittest.main()
