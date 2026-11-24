#!/usr/bin/env python3
"""Pré-entraînement causal de ghost avec checkpoints reprenables à l'intérieur d'un round.

Les « rounds » sont des fenêtres aléatoires, pas des epochs complètes du corpus.
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


def atomic_json(path: Path, value):
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


def evaluate(model, dataset, context, steps=32, seed=90210):
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


def append_jsonl(path: Path, record: dict):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Fichiers ou dossiers TXT/MD/JSONL/JSON/CSV")
    parser.add_argument("--output", default="runs/ghost")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3, help="Nombre de rounds de fenêtres aléatoires")
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--checkpoint-every-steps", type=int, default=50,
                        help="Sauvegarde de reprise toutes les N mises à jour réussies")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--tokens-per-shard", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Reprendre checkpoint.json si présent")
    args = parser.parse_args()
    if args.heads < 1 or args.vocab_size < 256 or args.d_model < 4 or args.d_model % args.heads:
        parser.error("vocab-size >= 256 ; heads >= 1 ; d-model >= 4 et divisible par heads requis")
    if min(args.layers, args.context, args.epochs, args.steps_per_epoch,
           args.tokens_per_shard, args.checkpoint_every_steps) <= 0:
        parser.error("layers/context/epochs/steps/tokens-per-shard/checkpoint-every doivent être positifs")
    if args.context < 2 or args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("context >= 2, learning-rate > 0 et weight-decay >= 0 requis")

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset_dir = output / "dataset"
    train_tokens_dir = output / "tokens" / "train"
    val_tokens_dir = output / "tokens" / "validation"

    print("[1/5] Ingestion, provenance, déduplication et splits document-level")
    report = data.DatasetBuilder(seed=args.seed).build(args.inputs, dataset_dir)
    dataset_fingerprint = file_sha256(dataset_dir / "manifest.jsonl")
    print(f"  documents gardés: {report.kept_documents}; doublons exacts: {report.exact_duplicates}; quasi-doublons: {report.near_duplicates}")

    tokenizer_path = output / "tokenizer.json"
    if args.resume and tokenizer_path.is_file():
        tokenizer = Tokenizer().load(tokenizer_path)
    else:
        print("[2/5] Entraînement BPE uniquement sur le split train")
        tokenizer = Tokenizer()
        tokenizer.train((doc.text for doc in data.iter_split(dataset_dir, "train")),
                        vocab_size=args.vocab_size, min_frequency=args.min_frequency, verbose=True)
        tokenizer.add_special_tokens(["<|eos|>"])
        tokenizer.save(tokenizer_path)
    eos_id = tokenizer.token_to_id("<|eos|>")

    print("[3/5] Encodage en shards uint32/mmap")
    data.write_token_shards(data.iter_split(dataset_dir, "train"), tokenizer,
                            train_tokens_dir, eos_id=eos_id,
                            tokens_per_shard=args.tokens_per_shard)
    data.write_token_shards(data.iter_split(dataset_dir, "validation"), tokenizer,
                            val_tokens_dir, eos_id=eos_id,
                            tokens_per_shard=args.tokens_per_shard)

    with data.TokenShardDataset(train_tokens_dir) as train_ds, data.TokenShardDataset(val_tokens_dir) as val_ds:
        if len(train_ds) < 2:
            raise SystemExit("Le split train ne contient pas assez de tokens après préparation.")
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
            "seed": args.seed, "dataset_fingerprint_sha256": dataset_fingerprint,
            "corpus_report": report.__dict__,
        }
        config_path = output / "run_config.json"
        if args.resume and config_path.is_file():
            previous_config = json.loads(config_path.read_text(encoding="utf-8"))
            if previous_config != run_config:
                raise SystemExit("Le corpus ou la configuration diffère du run précédent ; reprise refusée.")
        else:
            atomic_json(config_path, run_config)

        model = GPTModel(**config)
        optimizer = nn.AdamW(model.parameters(), lr=args.learning_rate,
                             weight_decay=args.weight_decay)
        checkpoint_path = output / "checkpoint.json"
        metrics_path = output / "metrics.jsonl"
        global_step = 0
        tokens_seen = 0
        completed_epochs = 0
        resume_step_in_epoch = 0
        resume_round_tokens = 0
        resume_loss_sum = 0.0
        resume_loss_count = 0
        rng = random.Random(args.seed)
        tokenizer_sha256 = file_sha256(tokenizer_path)

        def save_checkpoint(epoch_completed, step_in_epoch, round_tokens, loss_sum, loss_count):
            atomic_json(checkpoint_path, {
                "format": "ghost-training-checkpoint-v2",
                "completed_epochs": epoch_completed,
                "step_in_epoch": step_in_epoch,
                "round_tokens": round_tokens,
                "round_loss_sum": loss_sum,
                "round_loss_count": loss_count,
                "global_step": global_step,
                "tokens_seen": tokens_seen,
                "config": config,
                "dataset_fingerprint_sha256": dataset_fingerprint,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng_state": list(rng.getstate()[1]),
                "tokenizer_sha256": tokenizer_sha256,
            })

        if args.resume and checkpoint_path.is_file():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint.get("format") not in {"ghost-training-checkpoint-v1", "ghost-training-checkpoint-v2"}:
                raise SystemExit("Format de checkpoint inconnu.")
            if checkpoint.get("config") != config:
                raise SystemExit("La configuration demandée diffère du checkpoint existant.")
            if checkpoint.get("tokenizer_sha256") != tokenizer_sha256:
                raise SystemExit("Le tokenizer a changé : reprise refusée pour éviter la corruption des IDs.")
            if checkpoint.get("dataset_fingerprint_sha256", dataset_fingerprint) != dataset_fingerprint:
                raise SystemExit("Le corpus a changé depuis le checkpoint ; reprise refusée.")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            if checkpoint.get("format") == "ghost-training-checkpoint-v2":
                completed_epochs = int(checkpoint.get("completed_epochs", 0))
                resume_step_in_epoch = int(checkpoint.get("step_in_epoch", 0))
                resume_round_tokens = int(checkpoint.get("round_tokens", 0))
                resume_loss_sum = float(checkpoint.get("round_loss_sum", 0.0))
                resume_loss_count = int(checkpoint.get("round_loss_count", 0))
            else:
                completed_epochs = int(checkpoint.get("epoch", 0))
            global_step = int(checkpoint.get("global_step", 0))
            tokens_seen = int(checkpoint.get("tokens_seen", 0))
            if checkpoint.get("rng_state"):
                rng.setstate((3, tuple(checkpoint["rng_state"]), None))
            print(f"  reprise round={completed_epochs + 1}, étape={resume_step_in_epoch}, global_step={global_step}")

        try:
            print("[4/5] Entraînement causal next-token")
            baseline_val_loss = evaluate(model, val_ds, args.context) if len(val_ds) >= 2 else None
            if not metrics_path.exists():
                append_jsonl(metrics_path, {
                    "event": "initial_evaluation", "step": global_step,
                    "tokens_seen": tokens_seen, "validation_loss": baseline_val_loss,
                })
            for epoch in range(completed_epochs, args.epochs):
                started = time.monotonic()
                round_tokens = resume_round_tokens if epoch == completed_epochs else 0
                loss_sum = resume_loss_sum if epoch == completed_epochs else 0.0
                loss_count = resume_loss_count if epoch == completed_epochs else 0
                start_step = resume_step_in_epoch if epoch == completed_epochs else 0
                window = min(args.context + 1, len(train_ds))
                for step_index in range(start_step, args.steps_per_epoch):
                    sample = train_ds.sample_window(window, rng)
                    if len(sample) >= 2:
                        loss_value = model.train_step(sample, optimizer, max_grad_norm=1.0)
                        consumed = len(sample) - 1
                        round_tokens += consumed
                        tokens_seen += consumed
                        global_step += 1
                        loss_sum += float(loss_value)
                        loss_count += 1
                        if global_step % args.checkpoint_every_steps == 0:
                            save_checkpoint(epoch, step_index + 1, round_tokens, loss_sum, loss_count)
                train_loss = loss_sum / loss_count if loss_count else None
                val_loss = evaluate(model, val_ds, args.context) if len(val_ds) >= 2 else None
                elapsed = max(time.monotonic() - started, 1e-9)
                throughput = round_tokens / elapsed
                print(f"  round {epoch+1}/{args.epochs} | train_loss={train_loss} | val_loss={val_loss} | tokens={round_tokens} | tokens/s={throughput:.1f}")
                append_jsonl(metrics_path, {
                    "event": "training_round", "round": epoch + 1,
                    "global_step": global_step, "tokens_seen": tokens_seen,
                    "round_tokens": round_tokens, "train_loss": train_loss,
                    "validation_loss": val_loss, "initial_validation_loss": baseline_val_loss,
                    "tokens_per_second": throughput, "elapsed_seconds": round(elapsed, 3),
                })
                model.save(str(output / "model.json"))
                save_checkpoint(epoch + 1, 0, 0, 0.0, 0)
                resume_step_in_epoch = resume_round_tokens = resume_loss_count = 0
                resume_loss_sum = 0.0
            print("[5/5] Génération de contrôle")
            prompt = "Ghost apprend"
            generated = model.generate(tokenizer.encode(prompt), max_new_tokens=80,
                                       temperature=0.8, top_k=40, seed=args.seed)
            generated_text = tokenizer.decode(generated)
            (output / "generation_sample.txt").write_text(
                f"PROMPT\n{prompt}\n\nGENERATION\n{generated_text}\n",
                encoding="utf-8",
            )
            print(generated_text)
            print(f"\nArtefacts: {output}")
            print(f"Métriques: {metrics_path}")
        finally:
            optimizer.free()
            model.free()


if __name__ == "__main__":
    main()
