import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "fetch_hf_stream.py"


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows

    def shuffle(self, seed=42, buffer_size=1000):
        return self

    def __iter__(self):
        return iter(self.rows)


class FakeHfApi:
    def dataset_info(self, dataset, revision):
        return types.SimpleNamespace(sha="a" * 40)


class HuggingFaceStreamTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("ghost_fetch_hf_stream", SCRIPT)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_bounded_stream_and_resume_preserve_provenance(self):
        rows = [
            {
                "text": (f"Document français numéro {i}. " * 12),
                "url": f"https://example.org/{i}", "source_id": f"id-{i}",
                "language_score": 0.99, "quality_score": 0.8,
                "doc_id": i, "date": "2026-01-01", "dump": "test",
            }
            for i in range(8)
        ]
        fake_datasets = types.SimpleNamespace(
            load_dataset=lambda *args, **kwargs: FakeDataset(rows)
        )
        fake_hub = types.SimpleNamespace(HfApi=FakeHfApi)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "sample.jsonl"
            for resume, limit in ((False, 3), (True, 5)):
                argv = [
                    "fetch_hf_stream.py", "--output", str(output),
                    "--max-documents", str(limit), "--max-total-chars", "100000",
                    "--min-chars", "50", "--min-quality-score", "0.3",
                ]
                if resume:
                    argv.append("--resume")
                with patch.dict(sys.modules, {"datasets": fake_datasets, "huggingface_hub": fake_hub}), patch.object(sys, "argv", argv):
                    self.assertEqual(self.module.main(), 0)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 5)
            self.assertEqual([item["metadata"]["doc_id"] for item in records], list(range(5)))
            self.assertEqual([item["_stream_index"] for item in records], list(range(5)))
            self.assertTrue(all(item["license"] == "ODC-By-1.0" for item in records))
            self.assertTrue(all(item["language"] == "fr" for item in records))
            manifest = json.loads(Path(str(output) + ".source.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["resolved_revision"], "a" * 40)
            self.assertEqual(manifest["documents"], 5)
            self.assertEqual(manifest["status"], "sample_complete")
            self.assertFalse(manifest["full_dataset_downloaded"])
            self.assertTrue(manifest["jsonl_sha256"])

    def test_resume_rejects_changed_hugging_face_revision(self):
        rows = [{"text": "Un document français assez long pour le test. " * 8,
                 "language_score": 0.99, "quality_score": 0.9}]
        fake_datasets = types.SimpleNamespace(load_dataset=lambda *a, **k: FakeDataset(rows))
        class ChangedApi:
            def dataset_info(self, dataset, revision):
                return types.SimpleNamespace(sha="b" * 40)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "sample.jsonl"
            argv = ["fetch_hf_stream.py", "--output", str(output), "--min-chars", "20",
                    "--max-documents", "1"]
            with patch.dict(sys.modules, {"datasets": fake_datasets, "huggingface_hub": types.SimpleNamespace(HfApi=FakeHfApi)}), patch.object(sys, "argv", argv):
                self.assertEqual(self.module.main(), 0)
            argv.append("--resume")
            with patch.dict(sys.modules, {"datasets": fake_datasets, "huggingface_hub": types.SimpleNamespace(HfApi=ChangedApi)}), patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    self.module.main()
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
