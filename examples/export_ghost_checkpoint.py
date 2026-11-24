#!/usr/bin/env python3
"""Reconstruit model.json depuis le checkpoint le plus récent (état d'entraînement exact)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from ghost.llm import GPTModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    config_path = run_dir / "run_config.json"
    checkpoint_path = run_dir / "checkpoint.json"
    tokenizer_path = run_dir / "tokenizer.json"
    if not all(path.is_file() for path in (config_path, checkpoint_path, tokenizer_path)):
        parser.error("run-dir doit contenir run_config.json, checkpoint.json et tokenizer.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))["config"]
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint.get("config") != config:
        raise SystemExit("Le checkpoint ne correspond pas à run_config.json")
    model = GPTModel(**config)
    try:
        model.load_state_dict(checkpoint["model"])
        model.save(str(run_dir / "model.json"))
    finally:
        model.free()
    print(f"Modèle exporté depuis le checkpoint : {run_dir / 'model.json'}")
    print(f"global_step={checkpoint.get('global_step')} tokens_seen={checkpoint.get('tokens_seen')}")


if __name__ == "__main__":
    main()
