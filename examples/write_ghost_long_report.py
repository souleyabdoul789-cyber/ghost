#!/usr/bin/env python3
"""Écrit un rapport consolidé des segments de pré-entraînement ghost."""
from __future__ import annotations
import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_json(path: Path, default=None):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def load_jsonl(path: Path):
    records = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"JSONL invalide {path}:{line_number}: {exc}") from exc
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config = load_json(run_dir / "run_config.json", {}) or {}
    checkpoint = load_json(run_dir / "checkpoint.json", {}) or {}
    status = load_json(run_dir / "run_status.json", {}) or {}
    dataset = load_json(run_dir / "dataset" / "dataset.json", {}) or {}
    source = load_json(args.source_manifest.resolve(), {}) or {}
    metrics = load_jsonl(run_dir / "metrics.jsonl")
    completed_rounds = [m for m in metrics if m.get("event") == "training_round"]
    segments = [m for m in metrics if m.get("event") == "training_segment"]
    initial = next((m.get("validation_loss") for m in metrics
                    if m.get("event") == "initial_evaluation"), None)
    latest_validation = status.get("validation_loss")
    if latest_validation is None and segments:
        latest_validation = segments[-1].get("validation_loss")
    if latest_validation is None and completed_rounds:
        latest_validation = completed_rounds[-1].get("validation_loss")

    training_seconds = sum(float(item.get("segment_seconds", 0.0)) for item in segments)
    if not segments:
        training_seconds = sum(float(item.get("elapsed_seconds", 0.0)) for item in completed_rounds)
    total_tokens = int(status.get("tokens_seen", checkpoint.get("tokens_seen", 0)))
    total_steps = int(status.get("global_step", checkpoint.get("global_step", 0)))
    split_counts = dataset.get("report", {}).get("split_counts", {})
    model_config = config.get("config", {})
    lines = [
        "GHOST 0.7.0 — RAPPORT DE PRÉ-ENTRAÎNEMENT SEGMENTÉ",
        "=" * 58,
        f"Date UTC : {datetime.now(timezone.utc).isoformat()}",
        f"Plateforme : {platform.platform()}",
        f"Python : {sys.version.split()[0]}",
        "",
        "CORPUS ET PROVENANCE",
        "--------------------",
        f"Dataset : {source.get('dataset', 'non renseigné')}",
        f"Configuration : {source.get('config', 'non renseignée')}",
        f"Révision résolue : {source.get('resolved_revision', 'non renseignée')}",
        f"Licence déclarée : {source.get('license', 'non renseignée')}",
        f"Attribution : {source.get('attribution', 'non renseignée')}",
        f"Documents extraits : {source.get('documents', 'inconnu')}",
        f"Caractères extraits : {source.get('characters', 'inconnu')}",
        f"Documents gardés après nettoyage/déduplication : {dataset.get('report', {}).get('kept_documents', 'inconnu')}",
        f"Train/validation/test : {split_counts}",
        "",
        "CONFIGURATION DU MODÈLE",
        "------------------------",
        f"Vocabulaire : {model_config.get('vocab_size', 'inconnu')}",
        f"Dimension : {model_config.get('d_model', 'inconnu')}",
        f"Couches : {model_config.get('n_layers', 'inconnu')}",
        f"Têtes : {model_config.get('n_heads', 'inconnu')}",
        f"Contexte : {model_config.get('max_seq_len', 'inconnu')} tokens",
        "Optimiseur : AdamW",
        "",
        "PROGRESSION",
        "-----------",
        f"Segments achevés : {len(segments)}",
        f"Rounds entièrement achevés : {len(completed_rounds)}",
        f"Étapes d'optimisation : {total_steps}",
        f"Tokens vus : {total_tokens}",
        f"Temps cumulé d'entraînement déclaré : {training_seconds / 3600:.2f} heures",
        f"Loss de validation initiale : {initial}",
        f"Dernière loss de validation : {latest_validation}",
        f"Statut : {status}",
        "",
        "GÉNÉRATION DE CONTRÔLE",
        "-----------------------",
    ]
    sample = run_dir / "generation_sample.txt"
    lines.append(sample.read_text(encoding="utf-8", errors="replace") if sample.is_file()
                 else "Aucun échantillon de génération disponible.")
    lines += ["", "Le nombre de tokens vus est une mesure d'exposition, pas un score de compréhension.",
              "Le rapport ne prouve pas à lui seul une capacité générale ou une autonomie fiable."]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Rapport écrit : {args.output}")


if __name__ == "__main__":
    main()
