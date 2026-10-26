import hashlib
import json
from pathlib import Path
import tempfile
import unittest


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


if __name__ == "__main__":
    unittest.main()
