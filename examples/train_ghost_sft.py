"""Supervised fine-tuning de ghost sur des exemples prompt→completion.

Entrée JSONL : {"prompt":"...", "completion":"...", "license":"..."}
La loss ignore les tokens du prompt et n'entraîne que la completion.

Exemple :
  python examples/train_ghost_sft.py instructions.jsonl --base runs/pretrain \
      --output runs/ghost-tools --epochs 3
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


def encode_pair(tokenizer, prompt, completion, max_seq_len):
    prompt_ids = tokenizer.encode(prompt.rstrip() + "\n")
    completion_ids = tokenizer.encode(completion)
    if not completion_ids:
        return None
    if len(completion_ids) > max_seq_len:
        completion_ids = completion_ids[:max_seq_len]
    max_prompt = max_seq_len + 1 - len(completion_ids)
    if max_prompt < 1:
        return None
    prompt_ids = prompt_ids[-max_prompt:]
    if not prompt_ids:
        return None
    return prompt_ids + completion_ids, len(prompt_ids)


def masked_loss(model, tokenizer, example, context):
    pair = encode_pair(tokenizer, example.prompt, example.completion, context)
    if pair is None:
        return None
    sequence, target_start = pair
    logits_all = model.forward(sequence[:-1])
    start_row = target_start - 1
    logits = logits_all.transpose().slice_cols(start_row, len(sequence) - target_start).transpose()
    loss = nn.cross_entropy_loss_ids(logits, sequence[target_start:])
    value = float(loss.data)
    loss.free_graph()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="JSONL prompt/completion")
    parser.add_argument("--base", required=True, help="Répertoire avec tokenizer.json et model.json du pré-entraînement")
    parser.add_argument("--output", default="runs/ghost-sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("epochs doit être positif, learning-rate > 0 et weight-decay >= 0")

    base = Path(args.base).expanduser().resolve()
    tokenizer_path = base / "tokenizer.json"
    model_path = base / "model.json"
    if not tokenizer_path.is_file() or not model_path.is_file():
        parser.error("--base doit contenir tokenizer.json et model.json")
    tokenizer = Tokenizer().load(tokenizer_path)
    base_state = json.loads(model_path.read_text(encoding="utf-8"))
    config = base_state.get("config")
    if not isinstance(config, dict):
        parser.error("Le model.json fourni ne contient pas une configuration GPT reconnue.")
    model = GPTModel(
        vocab_size=config["vocab_size"], d_model=config["d_model"],
        n_heads=config["n_heads"], n_layers=config["n_layers"],
        max_seq_len=config["max_seq_len"], ff_mult=config["ff_mult"], seed=args.seed,
    )
    model.load(model_path)

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset_dir = output / "dataset"
    report = data.InstructionDatasetBuilder(seed=args.seed).build(args.manifest, dataset_dir)
    train_examples = list(data.iter_instruction_split(dataset_dir, "train"))
    val_examples = list(data.iter_instruction_split(dataset_dir, "validation"))
    if not train_examples:
        model.free()
        raise SystemExit("Le split train est vide. Ajoutez davantage d'exemples ou changez les fractions.")

    tokenizer_hash = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    base_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    optimizer = nn.AdamW(model.parameters(), lr=args.learning_rate,
                         weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    start_epoch = 0
    global_step = 0
    checkpoint_path = output / "checkpoint.json"
    if args.resume and checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("format") != "ghost-sft-checkpoint-v1":
            raise SystemExit("Format de checkpoint SFT inconnu.")
        if checkpoint.get("tokenizer_sha256") != tokenizer_hash or checkpoint.get("base_sha256") != base_hash:
            raise SystemExit("Tokenizer ou modèle de base modifié : reprise refusée.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        rng.setstate((3, tuple(checkpoint["rng_state"]), None))
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])

    try:
        print(f"Exemples: {report.kept_examples}; train={len(train_examples)}; validation={len(val_examples)}")
        for epoch in range(start_epoch, args.epochs):
            started = time.monotonic()
            shuffled = list(train_examples)
            rng.shuffle(shuffled)
            losses = []
            for example in shuffled:
                pair = encode_pair(tokenizer, example.prompt, example.completion, model.max_seq_len)
                if pair is None:
                    continue
                sequence, target_start = pair
                losses.append(model.train_step(sequence, optimizer, target_start=target_start))
                global_step += 1
            val_losses = [value for example in val_examples
                          if (value := masked_loss(model, tokenizer, example, model.max_seq_len)) is not None]
            train_loss = sum(losses) / len(losses) if losses else None
            val_loss = sum(val_losses) / len(val_losses) if val_losses else None
            print(f"epoch {epoch+1}/{args.epochs} train_loss={train_loss} val_loss={val_loss} steps={len(losses)} elapsed={time.monotonic()-started:.1f}s")
            model.save(str(output / "tool-model.json"))
            atomic_json(checkpoint_path, {
                "format": "ghost-sft-checkpoint-v1", "epoch": epoch + 1,
                "global_step": global_step, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "rng_state": list(rng.getstate()[1]),
                "tokenizer_sha256": tokenizer_hash, "base_sha256": base_hash,
                "dataset_report": report.__dict__,
            })
        print(f"Modèle SFT sauvegardé : {output / 'tool-model.json'}")
    finally:
        optimizer.free()
        model.free()


if __name__ == "__main__":
    main()
