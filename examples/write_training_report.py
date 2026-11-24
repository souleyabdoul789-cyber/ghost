#!/usr/bin/env python3
"""Produit un rapport texte autonome à partir des artefacts d'un run Ghost."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys


def load_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--source-manifest", default="")
    parser.add_argument("--output", default="ghost-first-training-report.txt")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    report_path = Path(args.output).resolve()
    metrics = load_jsonl(run_dir / "metrics.jsonl")
    run_config_path = run_dir / "run_config.json"
    corpus_meta_path = run_dir / "dataset" / "dataset.json"
    source_manifest_path = Path(args.source_manifest).resolve() if args.source_manifest else None
    run_config = json.loads(run_config_path.read_text(encoding="utf-8")) if run_config_path.exists() else {}
    corpus_meta = json.loads(corpus_meta_path.read_text(encoding="utf-8")) if corpus_meta_path.exists() else {}
    source_meta = json.loads(source_manifest_path.read_text(encoding="utf-8")) if source_manifest_path and source_manifest_path.exists() else {}

    initial = next((m.get("validation_loss") for m in metrics if m.get("event") == "initial_evaluation"), None)
    rounds = [m for m in metrics if m.get("event") == "training_round"]
    final = rounds[-1].get("validation_loss") if rounds else initial
    best = min((m.get("validation_loss") for m in metrics if isinstance(m.get("validation_loss"), (int, float))), default=None)
    tokens_seen = rounds[-1].get("tokens_seen", 0) if rounds else 0
    total_train_seconds = sum(float(m.get("elapsed_seconds", 0)) for m in rounds)
    avg_tps = (sum(int(m.get("round_tokens", 0)) for m in rounds) / total_train_seconds) if total_train_seconds > 0 else 0.0
    corpus_report = corpus_meta.get("report", {})
    split_counts = corpus_report.get("split_counts", {})
    model_files = ["model.json", "tokenizer.json", "checkpoint.json", "generation_sample.txt"]

    lines = [
        "GHOST — RAPPORT DU PREMIER PRÉ-ENTRAÎNEMENT",
        "=" * 48,
        f"Run directory : {run_dir}",
        f"Date UTC      : {__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}",
        f"Python        : {sys.version.split()[0]}",
        f"Plateforme    : {platform.platform()}",
        f"CPU logiques  : {os.cpu_count() or 'inconnu'}",
        "",
        "SOURCE DE DONNÉES",
        "-----------------",
        f"Dataset       : {source_meta.get('dataset', 'non renseigné')}",
        f"Configuration : {source_meta.get('config', 'non renseignée')}",
        f"Révision HF   : {source_meta.get('resolved_revision', 'non renseignée')}",
        f"État échantillon : {source_meta.get('status', 'non renseigné')}",
        f"Dataset complet téléchargé : {source_meta.get('full_dataset_downloaded', False)}",
        f"Licence       : {source_meta.get('license', 'à vérifier')}",
        f"Attribution   : {source_meta.get('attribution', 'à renseigner')}",
        f"Documents     : {source_meta.get('documents', 'inconnu')}",
        f"Caractères    : {source_meta.get('characters', 'inconnu')}",
        f"Corpus retenu : {corpus_report.get('kept_documents', 'inconnu')} documents",
        f"Train/val/test: {split_counts}",
        f"Doublons      : exacts={corpus_report.get('exact_duplicates', 'inconnu')}, quasi={corpus_report.get('near_duplicates', 'inconnu')}",
        "",
        "CONFIGURATION DU MODÈLE",
        "------------------------",
        f"Architecture  : {run_config.get('config', {}).get('n_layers', 'n/a')} couche(s) Transformer causales",
        f"D_model       : {run_config.get('config', {}).get('d_model', 'n/a')}",
        f"Têtes         : {run_config.get('config', {}).get('n_heads', 'n/a')}",
        f"Contexte      : {run_config.get('config', {}).get('max_seq_len', 'n/a')} tokens",
        f"Vocabulaire   : {run_config.get('config', {}).get('vocab_size', 'n/a')} tokens",
        f"Optimiseur    : AdamW",
        "",
        "RÉSULTATS D'ENTRAÎNEMENT",
        "------------------------",
        f"Étapes globales : {rounds[-1].get('global_step', 0) if rounds else 0}",
        f"Tokens vus      : {tokens_seen}",
        f"Durée train     : {total_train_seconds:.2f} s",
        f"Débit moyen     : {avg_tps:.1f} tokens/s",
        f"Loss validation initiale : {initial if initial is not None else 'n/a'}",
        f"Loss validation finale   : {final if final is not None else 'n/a'}",
        f"Meilleure loss validation: {best if best is not None else 'n/a'}",
        f"Nombre de rounds         : {len(rounds)}",
    ]
    if isinstance(initial, (int, float)) and isinstance(final, (int, float)):
        delta = final - initial
        lines.append(f"Évolution validation     : {delta:+.6f} ({'amélioration' if delta < 0 else 'pas d’amélioration'})")
    else:
        lines.append("Évolution validation     : non mesurable (évaluation manquante)")
    lines += ["", "FICHIERS ET EMPREINTES SHA-256", "------------------------------"]
    for name in model_files:
        path = run_dir / name
        if path.is_file():
            lines.append(f"{name}: {path.stat().st_size} octets | sha256={sha256(path)}")
        else:
            lines.append(f"{name}: ABSENT")
    if source_manifest_path and source_manifest_path.exists():
        lines.append(f"source_manifest: sha256={sha256(source_manifest_path)}")
    lines += ["", "ÉCHANTILLON DE GÉNÉRATION", "------------------------"]
    generation_path = run_dir / "generation_sample.txt"
    if generation_path.exists():
        lines.append(generation_path.read_text(encoding="utf-8", errors="replace").strip())
    else:
        lines.append("Aucun échantillon de génération enregistré.")
    lines += [
        "",
        "INTERPRÉTATION", "---------------",
        "Ce run est un pré-entraînement causal expérimental. Une baisse de loss ne prouve ni raisonnement général, ni compétence d'agent, ni autonomie.",
        "Le modèle de base ne doit pas être autorisé à exécuter librement des commandes ou à naviguer sans outils contrôlés, limites et validation des actions.",
        "La prochaine phase est l'instruction-tuning sur des exemples d'outils, suivie d'évaluations sandboxées terminal/web.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Rapport écrit : {report_path}")


if __name__ == "__main__":
    main()
