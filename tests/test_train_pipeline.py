import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TrainingPipelineTests(unittest.TestCase):
    def test_training_checkpoint_generation_report_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            command = [
                sys.executable, str(ROOT / "examples" / "train_ghost.py"),
                str(ROOT / "examples" / "data" / "tiny_text_demo.jsonl"),
                "--output", str(run_dir), "--vocab-size", "280",
                "--min-frequency", "2", "--d-model", "8", "--heads", "2",
                "--layers", "1", "--context", "32", "--epochs", "1",
                "--steps-per-epoch", "2", "--checkpoint-every-steps", "1",
                "--tokens-per-shard", "2000", "--seed", "41", "--resume",
            ]
            env = dict(os.environ)
            env["PYTHONPATH"] = str(ROOT / "python")
            first = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            self.assertEqual(first.returncode, 0, msg=first.stdout + "\n" + first.stderr)
            checkpoint_path = run_dir / "checkpoint.json"
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["format"], "ghost-training-checkpoint-v2")
            self.assertEqual(checkpoint["completed_epochs"], 1)
            self.assertGreater(checkpoint["global_step"], 0)
            self.assertTrue((run_dir / "generation_sample.txt").is_file())

            report_path = run_dir / "report.txt"
            report = subprocess.run([
                sys.executable, str(ROOT / "examples" / "write_training_report.py"),
                "--run-dir", str(run_dir), "--output", str(report_path),
            ], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(report.returncode, 0, msg=report.stdout + "\n" + report.stderr)
            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn("RAPPORT DU PREMIER PRÉ-ENTRAÎNEMENT", report_text)
            self.assertIn("generation_sample.txt", report_text)

            before_step = checkpoint["global_step"]
            resumed = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            self.assertEqual(resumed.returncode, 0, msg=resumed.stdout + "\n" + resumed.stderr)
            after = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(after["global_step"], before_step)


if __name__ == "__main__":
    unittest.main()
