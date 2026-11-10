"""Read event photos from the one configured, shared Google Drive folder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import gdown


DRIVE_FOLDER_ID = "1Fw76RVGWqtaKqCVBnDi3TR3oFqOfHLd_"
DRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class DriveImportError(RuntimeError):
    """The configured folder or one of its photos could not be read."""


@dataclass(frozen=True)
class DrivePhoto:
    file_id: str
    name: str


def list_drive_photos() -> list[DrivePhoto]:
    """List all supported photos, including subfolders, without downloading."""
    try:
        files = gdown.download_folder(
            url=DRIVE_FOLDER_URL,
            skip_download=True,
            quiet=True,
            use_cookies=False,
        )
        if files is None:
            raise DriveImportError("Google Drive không trả về danh sách tệp.")
        photos = {}
        for file in files:
            name = file.path.replace("\\", "/").rsplit("/", 1)[-1]
            if Path(name).suffix.lower() in IMAGE_SUFFIXES:
                photos.setdefault(file.id, DrivePhoto(file.id, name))
        return list(photos.values())
    except Exception as exc:
        raise DriveImportError(
            "Không đọc được thư mục Google Drive. Kiểm tra kết nối và quyền "
            "chia sẻ thư mục cùng các ảnh: Bất kỳ ai có đường liên kết → Người xem."
        ) from exc


def download_drive_photo(photo: DrivePhoto) -> bytes:
    """Download an original into an isolated temp folder and always clean up."""
    try:
        with TemporaryDirectory(prefix="face-search-drive-") as directory:
            # Never use a remote filename as a local destination.
            destination = Path(directory) / ("photo" + Path(photo.name).suffix.lower())
            downloaded = gdown.download(
                id=photo.file_id,
                output=str(destination),
                quiet=True,
                use_cookies=False,
            )
            if downloaded is None or not destination.is_file():
                raise DriveImportError("Google Drive không trả về nội dung ảnh.")
            return destination.read_bytes()
    except Exception as exc:
        raise DriveImportError(
            f"Không tải được ảnh {photo.name}. Kiểm tra quyền tải xuống và thử lại."
        ) from exc
