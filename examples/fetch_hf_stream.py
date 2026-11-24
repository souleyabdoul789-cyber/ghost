#!/usr/bin/env python3
"""Extrait un sous-corpus français depuis Hugging Face en streaming.

Le flux est borné, sa révision est résolue en SHA immuable, et la provenance
est écrite avant l'extraction puis finalisée à côté du JSONL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

DEFAULT_DATASET = "Brainquiver/general-web-fr-202608"
DEFAULT_CONFIG = "fineweb2-hq-fra_Latn"
LICENSE = "ODC-By-1.0"
DATASET_CARD = f"https://huggingface.co/datasets/{DEFAULT_DATASET}"
UPSTREAM_DATASET = "epfml/FineWeb2-HQ"
UPSTREAM_REVISION = "c0c06e94fd3a44ae9e802b2b0fc533817601eb5e"
ATTRIBUTION = (
    "Brainquiver general-web-fr-202608, derived from EPFL FineWeb2-HQ and Common Crawl; "
    "ODC-By 1.0. Preserve attribution and review Common Crawl terms of use."
)
STABLE_MANIFEST_KEYS = (
    "format", "dataset", "config", "split", "requested_revision", "resolved_revision",
    "license", "upstream_dataset", "upstream_revision", "filters", "seed", "shuffle_buffer",
)


def atomic_json(path: Path, value: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_dataset_revision(dataset: str, requested_revision: str) -> str:
    """Resolve main/tag to the exact HF commit SHA so resume is reproducible."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub absent; installez le supplément data avec `pip install '.[data]'`."
        ) from exc
    info = HfApi().dataset_info(dataset, revision=requested_revision)
    resolved = getattr(info, "sha", None)
    if not resolved or len(resolved) < 40:
        raise RuntimeError(f"Hugging Face n'a pas fourni de SHA immuable pour {dataset}@{requested_revision}.")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", default="main", help="Branche/tag/commit HF; résolu en SHA avant lecture")
    parser.add_argument("--output", default="data/fineweb2_fr_pilot.jsonl")
    parser.add_argument("--max-documents", type=int, default=10000)
    parser.add_argument("--max-total-chars", type=int, default=20_000_000)
    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument("--max-chars-per-document", type=int, default=12000)
    parser.add_argument("--min-language-score", type=float, default=0.95)
    parser.add_argument("--min-quality-score", type=float, default=0.30)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.max_documents < 1 or args.max_total_chars < 1:
        parser.error("Les limites max-documents et max-total-chars doivent être positives.")
    if args.min_chars < 1 or args.max_chars_per_document < args.min_chars:
        parser.error("Bornes de longueur invalides.")
    if args.shuffle_buffer < 1:
        parser.error("shuffle-buffer doit être positif.")
    if not 0 <= args.min_language_score <= 1 or not 0 <= args.min_quality_score <= 1:
        parser.error("Les seuils doivent être compris entre 0 et 1.")

    try:
        from datasets import load_dataset
    except ImportError:
        print("Dépendance absente : python -m pip install -U '.[data]'", file=sys.stderr)
        return 2

    try:
        resolved_revision = resolve_dataset_revision(args.dataset, args.revision)
    except Exception as exc:
        print(f"Impossible de résoudre la révision du dataset : {exc}", file=sys.stderr)
        return 3

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    source_manifest_path = output.with_suffix(output.suffix + ".source.json")
    mode = "a" if args.resume and output.exists() else "w"

    source_manifest = {
        "format": "ghost-hf-source-v2",
        "status": "in_progress",
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "dataset_card": DATASET_CARD if args.dataset == DEFAULT_DATASET else f"https://huggingface.co/datasets/{args.dataset}",
        "license": LICENSE if args.dataset == DEFAULT_DATASET else "REVIEW_REQUIRED",
        "upstream_dataset": UPSTREAM_DATASET if args.dataset == DEFAULT_DATASET else None,
        "upstream_revision": UPSTREAM_REVISION if args.dataset == DEFAULT_DATASET else None,
        "attribution": ATTRIBUTION if args.dataset == DEFAULT_DATASET else "Review dataset card before use.",
        "streaming": True,
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "filters": {
            "min_chars": args.min_chars,
            "max_chars_per_document": args.max_chars_per_document,
            "min_language_score": args.min_language_score,
            "min_quality_score": args.min_quality_score,
        },
        "max_documents_limit": args.max_documents,
        "max_total_chars_limit": args.max_total_chars,
    }

    already = 0
    chars_already = 0
    last_stream_index = -1
    if mode == "a":
        if not source_manifest_path.is_file():
            raise SystemExit(
                "Reprise refusée : manifeste source absent. Utilisez un nouveau chemin de sortie "
                "plutôt que de mélanger un corpus dont les filtres sont inconnus."
            )
        existing_source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        mismatches = [key for key in STABLE_MANIFEST_KEYS if existing_source.get(key) != source_manifest.get(key)]
        if mismatches:
            raise SystemExit("Reprise refusée : provenance/filtres différents : " + ", ".join(mismatches))
        with output.open("r", encoding="utf-8") as existing_stream:
            for line_number, line in enumerate(existing_stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    metadata = record.get("metadata", {})
                    expected = {
                        "dataset": args.dataset, "config": args.config,
                        "requested_revision": args.revision,
                        "resolved_revision": resolved_revision, "split": args.split,
                        "license": source_manifest["license"],
                    }
                    if any(metadata.get(key) != value for key, value in expected.items()):
                        raise SystemExit(
                            "Reprise refusée : métadonnées de document différentes du manifeste courant."
                        )
                    already += 1
                    chars_already += len(record.get("text", ""))
                    last_stream_index = int(record["_stream_index"])
                except (json.JSONDecodeError, AttributeError, KeyError, TypeError, ValueError) as exc:
                    raise SystemExit(f"JSONL existant invalide à la ligne {line_number}: {exc}") from exc

    if mode == "a" and (already >= args.max_documents or chars_already >= args.max_total_chars):
        source_manifest.update({
            "status": "sample_complete", "full_dataset_downloaded": False,
            "documents": already, "characters": chars_already,
            "jsonl_sha256": sha256_file(output),
            "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        atomic_json(source_manifest_path, source_manifest)
        print(f"Limite déjà atteinte : documents={already}, caractères={chars_already}")
        return 0

    # Write provenance before streaming: if the job is interrupted, resume can verify filters.
    atomic_json(source_manifest_path, source_manifest)
    print(f"Dataset: {args.dataset} / {args.config} [{args.split}]")
    print(f"Révision HF immuable: {resolved_revision}")
    print(f"Sortie: {output}; limite: {args.max_documents:,} documents / {args.max_total_chars:,} caractères")
    ds = load_dataset(
        args.dataset, args.config, split=args.split,
        streaming=True, revision=resolved_revision,
    )
    ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    kept = already
    total_chars = chars_already
    rejected = {"length": 0, "language": 0, "quality": 0, "empty": 0}
    started = time.monotonic()
    with output.open(mode, encoding="utf-8") as sink:
        for stream_index, row in enumerate(ds):
            if stream_index <= last_stream_index:
                continue
            text = str(row.get("text") or "").strip()
            if not text:
                rejected["empty"] += 1
                continue
            if len(text) < args.min_chars or len(text) > args.max_chars_per_document:
                rejected["length"] += 1
                continue
            language_score = float(row.get("language_score") or 0.0)
            quality_score = float(row.get("quality_score") or 0.0)
            if language_score < args.min_language_score:
                rejected["language"] += 1
                continue
            if quality_score < args.min_quality_score:
                rejected["quality"] += 1
                continue
            if total_chars + len(text) > args.max_total_chars:
                break

            record = {
                "text": text,
                "source": str(row.get("url") or row.get("source_id") or args.dataset),
                "title": str(row.get("source_id") or "FineWeb2-HQ French document"),
                "license": source_manifest["license"],
                "language": "fr",
                "_stream_index": stream_index,
                "metadata": {
                    **{key: source_manifest[key] for key in (
                        "dataset", "config", "requested_revision", "resolved_revision", "split", "license"
                    )},
                    "doc_id": row.get("doc_id"),
                    "source_id": row.get("source_id"),
                    "date": row.get("date"),
                    "dump": row.get("dump"),
                    "language_score": language_score,
                    "quality_score": quality_score,
                    "url": row.get("url"),
                },
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
            total_chars += len(text)
            if kept % 500 == 0:
                sink.flush()
                elapsed = max(time.monotonic() - started, 1e-9)
                print(f"documents={kept:,} chars={total_chars:,} docs/s={(kept-already)/elapsed:.1f}")
            if kept >= args.max_documents or total_chars >= args.max_total_chars:
                break
        sink.flush()
        os.fsync(sink.fileno())

    source_manifest.update({
        "status": "sample_complete",
        "full_dataset_downloaded": False,
        "documents": kept,
        "characters": total_chars,
        "jsonl_sha256": sha256_file(output),
        "rejected_this_run": rejected,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    atomic_json(source_manifest_path, source_manifest)
    print(json.dumps({
        "dataset": args.dataset,
        "config": args.config,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "split": args.split,
        "license": source_manifest["license"],
        "output": str(output),
        "source_manifest": str(source_manifest_path),
        "documents": kept,
        "characters": total_chars,
        "rejected_this_run": rejected,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
