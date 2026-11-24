#!/usr/bin/env python3
"""Entraînement Ghost segmenté avec checkpoints complets et reprise déterministe.

Le script est distinct du pilote 0.6.2 afin de ne pas modifier son comportement.
Les chemins de données/tokenizer sont réutilisés entre les segments GitHub Actions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from ghost import data, nn
from ghost.llm import GPTModel
from ghost.text import Tokenizer


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def evaluate(model, dataset, context: int, steps: int = 32, seed: int = 90210):
    if len(dataset) < 2:
        return None
    rng = random.Random(seed)
    window = min(context + 1, len(dataset))
    losses = []
    for _ in range(max(1, int(steps))):
        sample = dataset.sample_window(window, rng)
        if len(sample) < 2:
            continue
        logits = model.forward(sample[:-1])
        loss = nn.cross_entropy_loss_ids(logits, sample[1:])
        losses.append(float(loss.data))
        loss.free_graph()
    return sum(losses) / len(losses) if losses else None


def prepare_corpus(args, output: Path):
    dataset_dir = output / "dataset"
    manifest_path = dataset_dir / "manifest.jsonl"
    metadata_path = dataset_dir / "dataset.json"

    if args.resume and manifest_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format") != "ghost-corpus-v1" or not isinstance(metadata.get("report"), dict):
            raise SystemExit("Cache de corpus invalide : supprimez le cache ghost-fr-14day-v1 et relancez.")
        report = data.CorpusBuildReport(**metadata["report"])
        print("Corpus préparé réutilisé depuis le cache.")
    else:
        print("[1/5] Ingestion, provenance, déduplication et splits document-level")
        report = data.DatasetBuilder(seed=args.seed).build(args.inputs, dataset_dir)

    fingerprint = file_sha256(manifest_path)
    print(f"  documents gardés: {report.kept_documents}; doublons exacts: {report.exact_duplicates}; quasi-doublons: {report.near_duplicates}")
    return dataset_dir, report, fingerprint


def ensure_token_shards(documents, tokenizer, directory: Path, *, eos_id: int,
                        tokens_per_shard: int, dataset_fingerprint: str,
                        tokenizer_fingerprint: str, resume: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "ghost-tokenization.json"
    expected = {
        "format": "ghost-tokenization-cache-v1",
        "dataset_fingerprint_sha256": dataset_fingerprint,
        "tokenizer_sha256": tokenizer_fingerprint,
        "eos_id": int(eos_id),
        "tokens_per_shard": int(tokens_per_shard),
    }
    if resume and marker.is_file() and (directory / "tokens.json").is_file():
        try:
            actual = json.loads(marker.read_text(encoding="utf-8"))
            if actual == expected:
                with data.TokenShardDataset(directory) as checked:
                    if len(checked) >= 2:
                        print(f"  shards réutilisés: {directory}")
                        return
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    for stale in directory.glob("tokens-*.bin"):
        stale.unlink(missing_ok=True)
    (directory / "tokens.json").unlink(missing_ok=True)
    marker.unlink(missing_ok=True)
    data.write_token_shards(
        documents, tokenizer, directory,
        eos_id=eos_id, tokens_per_shard=tokens_per_shard,
    )
    atomic_json(marker, expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Fichiers ou dossiers TXT/MD/JSONL/JSON/CSV")
    parser.add_argument("--output", default="runs/ghost-fr-14day")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10000,
                        help="Nombre total de rounds de fenêtres aléatoires")
    parser.add_argument("--steps-per-epoch", type=int, default=20000)
    parser.add_argument("--checkpoint-every-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--tokens-per-shard", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seconds", type=float, default=19300,
                        help="Durée maximale de calcul de ce segment ; le checkpoint est écrit avant de sortir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Prépare corpus, tokenizer et shards sans lancer le modèle")
    args = parser.parse_args()

    if args.heads < 1 or args.vocab_size < 256 or args.d_model < 4 or args.d_model % args.heads:
        parser.error("vocab-size >= 256 ; heads >= 1 ; d-model >= 4 et divisible par heads requis")
    if min(args.layers, args.context, args.epochs, args.steps_per_epoch,
           args.tokens_per_shard, args.checkpoint_every_steps) <= 0:
        parser.error("layers/context/epochs/steps/tokens-per-shard/checkpoint-every doivent être positifs")
    if args.context < 32 or args.learning_rate <= 0 or args.weight_decay < 0 or args.max_seconds <= 0:
        parser.error("context >= 32, learning-rate > 0, weight-decay >= 0 et max-seconds > 0 requis")

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset_dir, report, dataset_fingerprint = prepare_corpus(args, output)

    tokenizer_path = output / "tokenizer.json"
    if args.resume and tokenizer_path.is_file():
        tokenizer = Tokenizer().load(tokenizer_path)
        print("Tokenizer BPE réutilisé.")
    else:
        print("[2/5] Entraînement BPE uniquement sur le split train")
        tokenizer = Tokenizer()
        tokenizer.train((doc.text for doc in data.iter_split(dataset_dir, "train")),
                        vocab_size=args.vocab_size, min_frequency=args.min_frequency, verbose=True)
        tokenizer.add_special_tokens(["<|eos|>"])
        tokenizer.save(tokenizer_path)

    eos_id = tokenizer.token_to_id("<|eos|>")
    tokenizer_fingerprint = file_sha256(tokenizer_path)
    print("[3/5] Encodage en shards uint32/mmap (réutilisés entre segments)")
    ensure_token_shards(
        data.iter_split(dataset_dir, "train"), tokenizer, output / "tokens" / "train",
        eos_id=eos_id, tokens_per_shard=args.tokens_per_shard,
        dataset_fingerprint=dataset_fingerprint, tokenizer_fingerprint=tokenizer_fingerprint,
        resume=args.resume,
    )
    ensure_token_shards(
        data.iter_split(dataset_dir, "validation"), tokenizer, output / "tokens" / "validation",
        eos_id=eos_id, tokens_per_shard=args.tokens_per_shard,
        dataset_fingerprint=dataset_fingerprint, tokenizer_fingerprint=tokenizer_fingerprint,
        resume=args.resume,
    )

    config = dict(vocab_size=tokenizer.vocab_size, d_model=args.d_model,
                  n_heads=args.heads, n_layers=args.layers,
                  max_seq_len=args.context, ff_mult=4, seed=args.seed)
    optimizer_config = {"name": "AdamW", "learning_rate": args.learning_rate,
                        "weight_decay": args.weight_decay, "max_grad_norm": 1.0}
    run_config = {
        "format": "ghost-pretrain-run-v2", "config": config,
        "optimizer": optimizer_config,
        "sampling": {"steps_per_round": args.steps_per_epoch,
                     "window_tokens": args.context + 1,
                     "rounds_are_random_windows_not_full_corpus_epochs": True,
                     "checkpoint_every_steps": args.checkpoint_every_steps},
        "seed": args.seed,
        "dataset_fingerprint_sha256": dataset_fingerprint,
        "corpus_report": report.__dict__,
    }
    config_path = output / "run_config.json"
    if args.resume and config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != run_config:
            raise SystemExit("La configuration ou le corpus diffère du run précédent ; reprise refusée.")
    else:
        atomic_json(config_path, run_config)

    if args.prepare_only:
        print("Préparation terminée : corpus, tokenizer, shards et configuration sont prêts.")
        return

    with data.TokenShardDataset(output / "tokens" / "train") as train_ds, \
         data.TokenShardDataset(output / "tokens" / "validation") as val_ds:
        if len(train_ds) < 2:
            raise SystemExit("Le split train ne contient pas assez de tokens après préparation.")

        model = GPTModel(**config)
        optimizer = nn.AdamW(model.parameters(), lr=args.learning_rate,
                             weight_decay=args.weight_decay)
        checkpoint_path = output / "checkpoint.json"
        metrics_path = output / "metrics.jsonl"
        global_step = 0
        tokens_seen = 0
        completed_epochs = 0
        resume_step = 0
        resume_round_tokens = 0
        resume_loss_sum = 0.0
        resume_loss_count = 0
        rng = random.Random(args.seed)

        def save_checkpoint(epoch_completed, step_in_epoch, round_tokens, loss_sum, loss_count):
            atomic_json(checkpoint_path, {
                "format": "ghost-training-checkpoint-v2",
                "completed_epochs": int(epoch_completed),
                "step_in_epoch": int(step_in_epoch),
                "round_tokens": int(round_tokens),
                "round_loss_sum": float(loss_sum),
                "round_loss_count": int(loss_count),
                "global_step": int(global_step),
                "tokens_seen": int(tokens_seen),
                "config": config,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng_state": list(rng.getstate()[1]),
                "tokenizer_sha256": tokenizer_fingerprint,
                "dataset_fingerprint_sha256": dataset_fingerprint,
            })

        if args.resume and checkpoint_path.is_file():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint.get("format") != "ghost-training-checkpoint-v2":
                raise SystemExit("Format de checkpoint inconnu ou incompatible.")
            if checkpoint.get("config") != config:
                raise SystemExit("La configuration demandée diffère du checkpoint existant.")
            if checkpoint.get("tokenizer_sha256") != tokenizer_fingerprint:
                raise SystemExit("Le tokenizer a changé : reprise refusée pour protéger les IDs.")
            if checkpoint.get("dataset_fingerprint_sha256") != dataset_fingerprint:
                raise SystemExit("Le corpus a changé depuis le checkpoint ; reprise refusée.")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            completed_epochs = int(checkpoint.get("completed_epochs", 0))
            resume_step = int(checkpoint.get("step_in_epoch", 0))
            resume_round_tokens = int(checkpoint.get("round_tokens", 0))
            resume_loss_sum = float(checkpoint.get("round_loss_sum", 0.0))
            resume_loss_count = int(checkpoint.get("round_loss_count", 0))
            global_step = int(checkpoint.get("global_step", 0))
            tokens_seen = int(checkpoint.get("tokens_seen", 0))
            if checkpoint.get("rng_state"):
                rng.setstate((3, tuple(checkpoint["rng_state"]), None))
            print(f"Checkpoint repris : round={completed_epochs + 1}, étape={resume_step}, global_step={global_step}, tokens={tokens_seen}")
        else:
            print("Nouveau modèle initialisé.")

        segment_start_step = global_step
        segment_start_tokens = tokens_seen
        training_start = time.monotonic()
        deadline = training_start + args.max_seconds
        stopped_for_time = False
        try:
            print("[4/5] Pré-entraînement causal next-token")
            initial_val = evaluate(model, val_ds, args.context, steps=32)
            latest_val = initial_val
            if not metrics_path.exists():
                append_jsonl(metrics_path, {
                    "event": "initial_evaluation", "step": global_step,
                    "tokens_seen": tokens_seen, "validation_loss": initial_val,
                })

            for epoch in range(completed_epochs, args.epochs):
                round_started = time.monotonic()
                round_tokens = resume_round_tokens if epoch == completed_epochs else 0
                loss_sum = resume_loss_sum if epoch == completed_epochs else 0.0
                loss_count = resume_loss_count if epoch == completed_epochs else 0
                start_step = resume_step if epoch == completed_epochs else 0
                window = min(args.context + 1, len(train_ds))

                for step_index in range(start_step, args.steps_per_epoch):
                    if time.monotonic() >= deadline:
                        save_checkpoint(epoch, step_index, round_tokens, loss_sum, loss_count)
                        model.save(str(output / "model.json"))
                        stopped_for_time = True
                        break

                    sample = train_ds.sample_window(window, rng)
                    if len(sample) < 2:
                        continue
                    loss_value = model.train_step(sample, optimizer, max_grad_norm=1.0)
                    consumed = len(sample) - 1
                    round_tokens += consumed
                    tokens_seen += consumed
                    global_step += 1
                    loss_sum += float(loss_value)
                    loss_count += 1

                    if global_step % args.checkpoint_every_steps == 0:
                        save_checkpoint(epoch, step_index + 1, round_tokens, loss_sum, loss_count)

                if stopped_for_time:
                    break

                train_loss = loss_sum / loss_count if loss_count else None
                val_loss = evaluate(model, val_ds, args.context, steps=32)
                latest_val = val_loss
                elapsed = max(time.monotonic() - round_started, 1e-9)
                throughput = round_tokens / elapsed
                print(f"round {epoch + 1}/{args.epochs} | train_loss={train_loss} | val_loss={val_loss} | tokens={round_tokens} | tokens/s={throughput:.1f}")
                append_jsonl(metrics_path, {
                    "event": "training_round", "round": epoch + 1,
                    "global_step": global_step, "tokens_seen": tokens_seen,
                    "round_tokens": round_tokens, "train_loss": train_loss,
                    "validation_loss": val_loss, "initial_validation_loss": initial_val,
                    "tokens_per_second": throughput, "elapsed_seconds": round(elapsed, 3),
                })
                model.save(str(output / "model.json"))
                save_checkpoint(epoch + 1, 0, 0, 0.0, 0)
                resume_step = resume_round_tokens = resume_loss_count = 0
                resume_loss_sum = 0.0

            if stopped_for_time:
                latest_val = evaluate(model, val_ds, args.context, steps=32)
                segment_seconds = round(time.monotonic() - training_start, 3)
                segment_token_count = tokens_seen - segment_start_tokens
                append_jsonl(metrics_path, {
                    "event": "training_segment", "global_step": global_step,
                    "tokens_seen": tokens_seen, "validation_loss": latest_val,
                    "segment_steps": global_step - segment_start_step,
                    "segment_tokens": segment_token_count,
                    "tokens_per_second": round(segment_token_count / max(segment_seconds, 1e-9), 2),
                    "segment_seconds": segment_seconds,
                    "step_in_epoch": json.loads(checkpoint_path.read_text(encoding="utf-8")).get("step_in_epoch"),
                })
                print(f"Segment sauvegardé : global_step={global_step}, tokens_seen={tokens_seen}, val_loss={latest_val}")
            else:
                save_checkpoint(args.epochs, 0, 0, 0.0, 0)
                print("Nombre total de rounds demandé atteint.")

            print("[5/5] Génération de contrôle")
            generated = model.generate(tokenizer.encode("Ghost apprend"), max_new_tokens=100,
                                       temperature=0.8, top_k=40, seed=args.seed)
            generated_text = tokenizer.decode(generated)
            (output / "generation_sample.txt").write_text(
                f"PROMPT\nGhost apprend\n\nGENERATION\n{generated_text}\n",
                encoding="utf-8",
            )
            atomic_json(output / "run_status.json", {
                "global_step": global_step,
                "tokens_seen": tokens_seen,
                "completed_rounds": json.loads(checkpoint_path.read_text(encoding="utf-8")).get("completed_epochs", 0),
                "step_in_round": json.loads(checkpoint_path.read_text(encoding="utf-8")).get("step_in_epoch", 0),
                "stopped_for_segment_limit": stopped_for_time,
                "validation_loss": latest_val,
            })
            print(generated_text)
        finally:
            optimizer.free()
            model.free()


if __name__ == "__main__":
    main()
