import json
import os
import tempfile
import unittest
from pathlib import Path

from ghost.data import (DatasetBuilder, TokenShardDataset, iter_split, write_token_shards,
                        ImageTextDatasetBuilder, iter_image_split,
                        InstructionDatasetBuilder, iter_instruction_split,
                        ImageInstructionDatasetBuilder, iter_image_instruction_split)
from ghost.text import Tokenizer


class DataPipelineTests(unittest.TestCase):
    def test_document_splits_dedup_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "corpus.jsonl"
            rows = [
                {"text": "Le robot apprend à lire les données et à raisonner.", "source": "manual:1", "license": "CC-BY-4.0"},
                {"text": "Le robot apprend à lire les données et à raisonner.", "source": "manual:2", "license": "CC-BY-4.0"},
                {"text": "The model learns from examples and predicts the next token.", "source": "manual:3", "license": "CC0-1.0"},
                {"text": "La rétropropagation calcule les gradients du réseau neuronal.", "source": "manual:4", "license": "CC-BY-4.0"},
                {"text": "Vision transforms image pixels into useful numerical features.", "source": "manual:5", "license": "CC0-1.0"},
            ]
            source.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows), encoding="utf-8")
            output = Path(tmp) / "dataset"
            report = DatasetBuilder(seed=7, val_fraction=0.2, test_fraction=0.2,
                                    near_duplicate_threshold=0).build([source], output)
            self.assertEqual(report.input_documents, 5)
            self.assertEqual(report.kept_documents, 4)
            self.assertEqual(report.exact_duplicates, 1)
            all_docs = list(iter_split(output, "train")) + list(iter_split(output, "validation")) + list(iter_split(output, "test"))
            self.assertEqual(len(all_docs), 4)
            self.assertTrue(all(d.sha256 and d.source and d.license for d in all_docs))
            self.assertTrue((output / "dataset.json").is_file())

    def test_token_shards_cross_boundary_and_mmap(self):
        with tempfile.TemporaryDirectory() as tmp:
            tokenizer = Tokenizer()
            tokenizer.train(["ghost learns from a compact text corpus", "memory stores validated facts"], vocab_size=270)
            tokenizer.add_special_tokens(["<|eos|>"])
            docs = [
                type("Doc", (), {"text": "ghost learns from text"})(),
                type("Doc", (), {"text": "memory stores facts"})(),
            ]
            manifest = write_token_shards(docs, tokenizer, tmp, eos_id=tokenizer.token_to_id("<|eos|>"), tokens_per_shard=5)
            self.assertGreater(len(manifest["shards"]), 1)
            with TokenShardDataset(tmp) as dataset:
                expected = tokenizer.encode(docs[0].text) + [tokenizer.token_to_id("<|eos|>")] + tokenizer.encode(docs[1].text) + [tokenizer.token_to_id("<|eos|>")]
                self.assertEqual(len(dataset), len(expected))
                self.assertEqual(dataset.read_window(2, 7), expected[2:9])
                self.assertEqual(dataset.sample_window(3, __import__("random").Random(1)).__len__(), 3)

    def test_image_text_manifest_grouping_and_path_safety(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            (images / "a.png").write_bytes(b"image-a")
            manifest = root / "images.jsonl"
            rows = [
                {"image": "a.png", "text": "Un chat sur une chaise", "license": "CC-BY-4.0"},
                {"image": "a.png", "text": "Un chat assis", "license": "CC-BY-4.0"},
                {"image": "a.png", "text": "Un chat sur une chaise", "license": "CC-BY-4.0"},
            ]
            manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
            output = root / "image-data"
            report = ImageTextDatasetBuilder(images, val_fraction=0.25, test_fraction=0.25).build(manifest, output)
            self.assertEqual(report.input_pairs, 3)
            self.assertEqual(report.kept_pairs, 2)
            self.assertEqual(report.exact_duplicates, 1)
            records = list(iter_image_split(output, "train")) + list(iter_image_split(output, "validation")) + list(iter_image_split(output, "test"))
            self.assertEqual(len(records), 2)
            self.assertEqual(len({record.split for record in records}), 1)

            bad = root / "bad.jsonl"
            bad.write_text(json.dumps({"image": "../secret.png", "text": "unsafe"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                ImageTextDatasetBuilder(images).build(bad, root / "bad-out")

    def test_image_instruction_splits_group_same_image_and_deduplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            (images / "screen.jpg").write_bytes(b"screen-bytes")
            manifest = root / "agent.jsonl"
            rows = [
                {"image": "screen.jpg", "prompt": "Observe la page.",
                 "completion": '{"type":"tool","tool":"open_page"}', "license": "CC0-1.0"},
                {"image": "screen.jpg", "prompt": "Observe la page.",
                 "completion": '{"type":"tool","tool":"open_page"}', "license": "CC0-1.0"},
                {"image": "screen.jpg", "prompt": "Lis le titre.",
                 "completion": '{"type":"final","answer":"Documentation"}', "license": "CC0-1.0"},
            ]
            manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            output = root / "vlm-sft"
            report = ImageInstructionDatasetBuilder(images, val_fraction=0.25, test_fraction=0.25).build(manifest, output)
            self.assertEqual(report.kept_examples, 2)
            self.assertEqual(report.duplicates, 1)
            examples = [item for split in ("train", "validation", "test")
                        for item in iter_image_instruction_split(output, split)]
            self.assertEqual(len(examples), 2)
            self.assertEqual(len({item.split for item in examples}), 1)
            bad = root / "escape.jsonl"
            bad.write_text(json.dumps({"image":"../private.jpg", "prompt":"p", "completion":"c"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                ImageInstructionDatasetBuilder(images).build(bad, root / "bad-vlm")

    def test_instruction_pairs_are_deduplicated_and_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "instructions.jsonl"
            rows = [
                {"prompt": "Choisis une action.", "completion": '{"type":"final","answer":"OK"}', "license": "CC0"},
                {"prompt": "Choisis une action.", "completion": '{"type":"final","answer":"OK"}', "license": "CC0"},
                {"prompt": "Trouve une page.", "completion": '{"type":"tool","tool":"fetch_page","arguments":{"url":"https://example.org"}}'},
            ]
            source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            output = root / "instruction-data"
            report = InstructionDatasetBuilder(val_fraction=0.2, test_fraction=0.2).build(source, output)
            self.assertEqual(report.duplicates, 1)
            combined = list(iter_instruction_split(output, "train")) + list(iter_instruction_split(output, "validation")) + list(iter_instruction_split(output, "test"))
            self.assertEqual(len(combined), 2)

    def test_unknown_input_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.txt"
            with self.assertRaises(FileNotFoundError):
                DatasetBuilder().build([path], Path(tmp) / "out")


if __name__ == "__main__":
    unittest.main()
