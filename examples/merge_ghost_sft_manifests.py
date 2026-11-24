#!/usr/bin/env python3
"""Fusionne et mélange des manifests SFT sans perdre leurs sources/licences."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    seen = set()
    counts = {}
    for path in args.inputs:
        count = 0
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"JSONL invalide {path}:{line_number}: {exc}") from exc
                if not isinstance(row, dict) or not row.get("prompt") or not row.get("completion"):
                    continue
                key = hashlib.sha256((row["prompt"] + "\0" + row["completion"]).encode("utf-8")).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
                count += 1
        counts[str(path)] = count

    if not rows:
        raise SystemExit("Aucun exemple SFT valide à fusionner.")
    random.Random(args.seed).shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Manifest SFT fusionné : {len(rows)} exemples ; détail={json.dumps(counts, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
