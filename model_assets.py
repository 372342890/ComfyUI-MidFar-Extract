"""Pinned model download; no network/model work happens during plugin import."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import urllib.request

HUNYUAN = {
    "repository": "Comfy-Org/Hunyuan-3D-2.0",
    "revision": "6da110fc4bbaa23cc4fe7d20129c54c2aac28f80",
    "filename": "hunyuan3d-dit-v2_fp16.safetensors",
    "remote_path": "split_files/hunyuan3d-dit-v2_fp16.safetensors",
    "version": "2.0",
    "bytes": 4928151562,
    "sha256": "360bc281fc956d4acac0c3d36d5ec0ebf8cdddbf4b8892e894d12419388d479b",
}
HUNYUAN["url"] = ("https://huggingface.co/" + HUNYUAN["repository"] +
                  "/resolve/" + HUNYUAN["revision"] + "/" + HUNYUAN["remote_path"])


def _verified(path, expected_size, expected_sha):
    if path.stat().st_size != expected_size:
        raise ValueError("Checkpoint length verification failed")
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
    if sha.hexdigest() != expected_sha:
        raise ValueError("Checkpoint SHA256 verification failed")


def verified_download(url, destination, expected_size, expected_sha, progress_path=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    progress_path = Path(progress_path) if progress_path else None

    def progress(status, received):
        if progress_path:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            record = {"status": status, "received": received,
                      "expected": expected_size, "filename": destination.name,
                      "time": time.time()}
            temp = progress_path.with_suffix(".tmp")
            temp.write_text(json.dumps(record), encoding="utf-8")
            temp.replace(progress_path)

    if destination.exists():
        progress("verifying-existing", destination.stat().st_size)
        _verified(destination, expected_size, expected_sha)
        progress("verified", expected_size)
        return destination
    part = destination.with_suffix(destination.suffix + ".part")
    received = 0
    progress("downloading", received)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "MidFar-3D/1.0"})
        sha, last = hashlib.sha256(), 0
        with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as out:
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                received += len(block)
                if received > expected_size:
                    raise ValueError("Checkpoint exceeds pinned length")
                out.write(block)
                sha.update(block)
                if time.monotonic() - last > 2:
                    progress("downloading", received)
                    last = time.monotonic()
        if received != expected_size or sha.hexdigest() != expected_sha:
            raise ValueError("Downloaded checkpoint length/SHA256 verification failed")
        part.replace(destination)
        progress("verified", received)
        return destination
    except Exception as error:
        progress("failed", received)
        # Redirect URLs can contain access tokens; never include them in exceptions.
        raise ValueError("Pinned checkpoint transfer failed: " + type(error).__name__) from None


def parallel_verified_download(url, destination, expected_size, expected_sha,
                               progress_path=None, workers=8, chunk_size=64*1024*1024):
    """Range-transfer a pinned large file; never promote an unverified part."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return verified_download(url, destination, expected_size, expected_sha, progress_path)
    part = destination.with_suffix(destination.suffix+".part")
    progress_path = Path(progress_path) if progress_path else None
    received, last_update = 0, 0
    lock = threading.Lock()

    def progress(status):
        if progress_path:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            record = {"status": status, "received": received, "expected": expected_size,
                      "filename": destination.name, "workers": workers, "time": time.time()}
            temp = progress_path.with_suffix(".tmp")
            temp.write_text(json.dumps(record), encoding="utf-8")
            temp.replace(progress_path)

    progress("downloading")
    with part.open("wb") as out:
        out.truncate(expected_size)

    def transfer(bounds):
        nonlocal received, last_update
        first, last = bounds
        request = urllib.request.Request(url, headers={
            "User-Agent": "MidFar-3D/1.0", "Range": f"bytes={first}-{last}"})
        with urllib.request.urlopen(request, timeout=120) as response, part.open("r+b") as out:
            if (response.status != 206 or
                    response.headers.get("Content-Range") != f"bytes {first}-{last}/{expected_size}"):
                raise ValueError("Model host did not honor the verified byte range")
            out.seek(first)
            count = 0
            while True:
                block = response.read(4*1024*1024)
                if not block:
                    break
                count += len(block)
                if count > last-first+1:
                    raise ValueError("Model range length exceeded")
                out.write(block)
                with lock:
                    received += len(block)
                    if time.monotonic()-last_update > 2:
                        progress("downloading")
                        last_update = time.monotonic()
            if count != last-first+1:
                raise ValueError("Model range length mismatch")

    bounds = [(first, min(first+chunk_size, expected_size)-1)
              for first in range(0, expected_size, chunk_size)]
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in pool.map(transfer, bounds):
                pass
        progress("verifying")
        _verified(part, expected_size, expected_sha)
        part.replace(destination)
        progress("verified")
        return destination
    except Exception as error:
        progress("failed")
        raise ValueError("Parallel pinned checkpoint transfer failed: " +
                         type(error).__name__) from None


def ensure_hunyuan_checkpoint(models_directory, progress_path=None):
    models_directory = Path(models_directory)
    destination = models_directory/"checkpoints"/"midfar"/HUNYUAN["filename"]
    parallel_verified_download(HUNYUAN["url"], destination, HUNYUAN["bytes"],
                               HUNYUAN["sha256"], progress_path)
    return "midfar/" + HUNYUAN["filename"]
