"""Opt-in browser test using the real Drive folder, ONNX models, and FAISS.

Run from the repository root: python -B tests/ui_drive_smoke.py
Uses isolated temporary photos/indexes and never modifies the existing album.
Requires Playwright and an installed Edge (default on Windows) or Chromium.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.parse import unquote

import requests
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = """
import sys
from pathlib import Path
import uvicorn
from fastapi.staticfiles import StaticFiles
import app

root = Path(sys.argv[1])
app.PROJECT_ROOT = root
app.EVENT_PHOTOS_DIR = root / "data" / "event_photos"
app.STORAGE_DIR = root / "storage"
for route in app.app.routes:
    if getattr(route, "name", None) == "event-photos":
        route.app = StaticFiles(directory=str(app.EVENT_PHOTOS_DIR))
uvicorn.run(app.app, host="127.0.0.1", port=int(sys.argv[2]), log_level="warning")
"""


def run(channel: str | None, timeout: int) -> dict:
    for model in ("scrfd_10g.onnx", "arcface_w600k_r50.onnx"):
        if not (ROOT / "models" / model).is_file():
            raise RuntimeError(f"Download models/{model} before running this test")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="face-search-ui-drive-") as directory:
        test_root = Path(directory)
        album = test_root / "data" / "event_photos"
        album.mkdir(parents=True)
        storage = test_root / "storage"
        storage.mkdir()
        shutil.copyfile(ROOT / "storage" / "best_config.json", storage / "best_config.json")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        print(f"Temporary test data: {test_root}", flush=True)
        with (test_root / "server.log").open("w", encoding="utf-8") as log:
            server = subprocess.Popen(
                [sys.executable, "-B", "-c", BOOTSTRAP, str(test_root), str(port)],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                deadline = time.monotonic() + 20
                while True:
                    if server.poll() is not None:
                        raise RuntimeError("Test server exited before startup")
                    try:
                        if requests.get(url + "/health", timeout=1).status_code == 200:
                            break
                    except requests.RequestException:
                        pass
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Test server did not start")
                    time.sleep(.1)

                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(channel=channel, headless=True)
                    try:
                        context = browser.new_context(accept_downloads=True)
                        page = context.new_page()
                        errors = []
                        page.on("pageerror", lambda error: errors.append(str(error)))
                        page.goto(url, wait_until="networkidle")
                        assert page.locator("input[type=file]").count() == 1
                        assert page.locator("#album-total").inner_text() == "0"
                        assert not list(album.iterdir())
                        print("Clicking the real Drive import button with an empty album...", flush=True)
                        with page.expect_response(
                            lambda response: response.url == url + "/events"
                            and response.request.method == "POST",
                            timeout=timeout * 1000,
                        ) as importing:
                            page.locator("#event-upload-button").click()
                            assert page.locator("#event-upload-button").is_disabled()
                            # A long import must leave the server responsive and
                            # reject a second batch instead of adding duplicates.
                            assert requests.get(url + "/health", timeout=5).status_code == 200
                            assert requests.post(url + "/events", timeout=5).status_code == 409
                        response = importing.value
                        imported = response.json()
                        assert response.status == 200, imported
                        assert not imported["failed_photos"], imported
                        assert imported["photos_found"] > 0, imported
                        assert imported["photos_uploaded"] == imported["photos_found"], imported
                        assert imported["faces_indexed"] > 0, imported
                        page.wait_for_function("!document.querySelector('#event-upload-button').disabled")
                        page.wait_for_function(
                            "count => document.querySelector('#album-total').textContent === String(count)",
                            arg=imported["photos_uploaded"],
                        )
                        assert "success" in page.locator("#event-status").get_attribute("class")
                        metadata = json.loads((storage / "metadata.json").read_text(encoding="utf-8"))
                        assert len(metadata) == imported["faces_indexed"]
                        assert len(list(album.iterdir())) == imported["photos_uploaded"]
                        print("Real UI import: " + json.dumps(imported, ensure_ascii=True), flush=True)

                        query = (test_root / next(iter(metadata.values()))["photo_path"]).read_bytes()
                        page.locator("#query-file").set_input_files(
                            {"name": "query.png", "mimeType": "image/png", "buffer": query}
                        )
                        page.locator("#search-button").click()
                        page.wait_for_selector(".photo-download", timeout=60000)
                        count = page.locator(".photo-download").count()
                        assert count > 0 and page.locator("#download-panel").is_visible()
                        first_url = page.locator(".photo-download").first.get_attribute("href")
                        expected = album / unquote(first_url.split("/downloads/photo/", 1)[1])
                        with page.expect_download() as downloading:
                            page.locator(".photo-download").first.click()
                        single = test_root / "single.png"
                        downloading.value.save_as(single)
                        assert single.read_bytes() == expected.read_bytes()
                        with page.expect_download(timeout=60000) as downloading_zip:
                            page.locator("#download-all").click()
                        zipped = test_root / "results.zip"
                        downloading_zip.value.save_as(zipped)
                        with zipfile.ZipFile(zipped) as bundle:
                            assert len(bundle.namelist()) == count
                            assert bundle.testzip() is None
                            for name in bundle.namelist():
                                assert name.startswith("anh-cua-ban/")
                                original = album / name.removeprefix("anh-cua-ban/")
                                assert bundle.read(name) == original.read_bytes()
                        assert not errors, errors
                        assert json.loads((storage / "metadata.json").read_text(encoding="utf-8")) == metadata
                        result = {
                            "import": imported,
                            "search_matches": count,
                            "single_download": "byte-for-byte PASS",
                            "zip_download": "all originals byte-for-byte PASS",
                            "concurrent_import": "409 PASS",
                            "javascript_errors": errors,
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                        }
                    finally:
                        browser.close()
            except BaseException:
                print((test_root / "server.log").read_text(encoding="utf-8", errors="replace")[-6000:])
                raise
            finally:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)
    assert not Path(directory).exists()
    result["temporary_data_removed"] = True
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="msedge" if os.name == "nt" else None)
    parser.add_argument("--timeout", type=int, default=600, help="Import timeout in seconds")
    arguments = parser.parse_args()
    if arguments.timeout <= 0:
        parser.error("--timeout must be a positive number of seconds")
    print(json.dumps(run(arguments.channel, arguments.timeout), ensure_ascii=True, indent=2))
