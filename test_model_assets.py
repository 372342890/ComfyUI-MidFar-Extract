import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import http.server
import threading


class WeightTests(unittest.TestCase):
    def download(self, source, target, **kwargs):
        import model_assets
        data = source.read_bytes()
        return model_assets.verified_download(
            source.as_uri(), target, len(data),
            hashlib.sha256(data).hexdigest(), **kwargs)

    def test_verified_local_transfer_and_progress(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, target, progress = root/"source", root/"models"/"weight", root/"progress.json"
            source.write_bytes(b"actual weight fixture" * 100)
            self.assertEqual(self.download(source, target, progress_path=progress), target)
            self.assertEqual(source.read_bytes(), target.read_bytes())
            record = json.loads(progress.read_text())
            self.assertEqual(record["status"], "verified")
            self.assertEqual(record["received"], target.stat().st_size)

    def test_verified_existing_file_is_reused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, target = root/"source", root/"target"
            source.write_bytes(b"fixture")
            self.download(source, target)
            source.unlink()
            import model_assets
            self.assertEqual(model_assets.verified_download(
                source.as_uri(), target, 7, hashlib.sha256(b"fixture").hexdigest()), target)

    def test_corrupt_existing_file_is_not_silently_used_or_deleted(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, target = root/"source", root/"target"
            source.write_bytes(b"correct")
            target.write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                self.download(source, target)
            self.assertEqual(target.read_bytes(), b"corrupt")

    def test_wrong_checksum_never_promotes_partial_download(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, target = root/"source", root/"target"
            source.write_bytes(b"fixture")
            import model_assets
            with self.assertRaises(ValueError):
                model_assets.verified_download(source.as_uri(), target, 7, "0"*64)
            self.assertFalse(target.exists())

    def test_wrong_length_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, target = root/"source", root/"target"
            source.write_bytes(b"fixture")
            import model_assets
            with self.assertRaises(ValueError):
                model_assets.verified_download(source.as_uri(), target, 8,
                                               hashlib.sha256(b"fixture").hexdigest())
            self.assertFalse(target.exists())

    def test_fallback_download_promotes_only_verified_source(self):
        import model_assets
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            first, second, target = root/"unverified", root/"verified", root/"model"
            first.write_bytes(b"corrupt")
            second.write_bytes(b"correct")
            model_assets.verified_download_sources(
                [first.as_uri(), second.as_uri()], target, 7,
                hashlib.sha256(b"correct").hexdigest())
            self.assertEqual(target.read_bytes(), b"correct")

    def test_fallback_sources_do_not_delete_corrupt_existing_model(self):
        import model_assets
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, target = root/"source", root/"model"
            source.write_bytes(b"correct")
            target.write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                model_assets.verified_download_sources(
                    [source.as_uri()], target, 7, hashlib.sha256(b"correct").hexdigest())
            self.assertEqual(target.read_bytes(), b"corrupt")

    def test_parallel_ranges_preserve_real_bytes_and_verify_checksum(self):
        import model_assets
        payload = b"actual range fixture" * 100000
        requests = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                first, last = map(int, self.headers["Range"][6:].split("-"))
                requests.append((first, last))
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {first}-{last}/{len(payload)}")
                self.send_header("Content-Length", str(last-first+1))
                self.end_headers()
                self.wfile.write(payload[first:last+1])

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                target = Path(d)/"weight"
                model_assets.parallel_verified_download(
                    f"http://127.0.0.1:{server.server_port}/weight", target,
                    len(payload), hashlib.sha256(payload).hexdigest(),
                    workers=4, chunk_size=256*1024)
                self.assertEqual(target.read_bytes(), payload)
                self.assertGreater(len(requests), 4)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
