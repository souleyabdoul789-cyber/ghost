"""SFT multimodal : capture + prompt → completion/action JSON.

Manifeste JSONL :
  {"image":"page.jpg","prompt":"Observe cette page et choisis un outil.",
   "completion":"{\"type\":\"tool\",...}","source":"...","license":"CC0-1.0"}

Exemple :
  python examples/train_ghost_vlm_sft.py actions.jsonl --images ./captures \
      --base runs/ghost-base --vision-encoder encoder.onnx --feature-dim 768 \
      --output runs/ghost-agent-vlm --epochs 3
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
from ghost.vision import OnnxVisionEncoder, VisionLanguageModel


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def encode_pair(tokenizer, prompt, completion, max_context, visual_tokens):
    available = int(max_context) - int(visual_tokens)
    if available < 1:
        return None
    prompt_ids = tokenizer.encode(prompt.rstrip() + "\n")
    completion_ids = tokenizer.encode(completion)
    if not completion_ids:
        return None
    completion_ids = completion_ids[:available]
    prompt_budget = available - len(completion_ids)
    if prompt_budget <= 0:
        prompt_ids = []
    else:
        prompt_ids = prompt_ids[-prompt_budget:]
    ids = prompt_ids + completion_ids
    if not ids:
        return None
    return ids, len(prompt_ids)


def masked_loss(vlm, encoder, tokenizer, example, max_context):
    features = encoder.encode_tokens(example.image_path)
    pair = encode_pair(tokenizer, example.prompt, example.completion,
                       max_context, len(features))
    if pair is None:
        return None
    ids, target_start = pair
    logits_all = vlm.forward_features(features, ids)
    if target_start:
        logits = logits_all.transpose().slice_cols(target_start, len(ids) - target_start).transpose()
    else:
        logits = logits_all
    loss = nn.cross_entropy_loss_ids(logits, ids[target_start:])
    value = float(loss.data)
    loss.free_graph()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="JSONL image + prompt + completion")
    parser.add_argument("--images", required=True)
    parser.add_argument("--base", required=True, help="Pré-entraînement textuel avec tokenizer.json/model.json")
    parser.add_argument("--vision-encoder", required=True)
    parser.add_argument("--feature-dim", required=True, type=int)
    parser.add_argument("--spatial-layout", choices=["nchw", "nhwc"], default="nchw")
    parser.add_argument("--output", default="runs/ghost-vlm-sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.feature_dim <= 0 or args.epochs < 1 or args.learning_rate <= 0 or args.weight_decay < 0 or args.limit < 0:
        parser.error("feature-dim/epochs/lr/limit invalides")

    base = Path(args.base).expanduser().resolve()
    tokenizer_path, base_model_path = base / "tokenizer.json", base / "model.json"
    if not tokenizer_path.is_file() or not base_model_path.is_file():
        parser.error("--base doit contenir tokenizer.json et model.json")
    tokenizer = Tokenizer().load(tokenizer_path)
    base_state = json.loads(base_model_path.read_text(encoding="utf-8"))
    config = base_state.get("config")
    if not isinstance(config, dict):
        parser.error("model.json n'a pas de configuration GPT valide")

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    image_data_dir = output / "image-instructions"
    report = data.ImageInstructionDatasetBuilder(args.images, seed=args.seed).build(args.manifest, image_data_dir)
    train_examples = list(data.iter_image_instruction_split(image_data_dir, "train"))
    val_examples = list(data.iter_image_instruction_split(image_data_dir, "validation"))
    if args.limit:
        train_examples = train_examples[:args.limit]
    if not train_examples:
        raise SystemExit("Le split train image-instruction est vide.")

    model = GPTModel(**config)
    model.load(base_model_path)
    encoder = OnnxVisionEncoder(args.vision_encoder, spatial_layout=args.spatial_layout)
    vlm = VisionLanguageModel(model, encoder, feature_dim=args.feature_dim, seed=args.seed + 1)
    optimizer = nn.AdamW(vlm.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    hashes = {
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "base_sha256": sha256_file(base_model_path),
        "encoder_sha256": sha256_file(args.vision_encoder),
    }
    checkpoint_path = output / "checkpoint.json"
    start_epoch = global_step = 0
    if args.resume and checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("format") != "ghost-vlm-sft-checkpoint-v1":
            raise SystemExit("Format de checkpoint VLM-SFT inconnu.")
        if checkpoint.get("hashes") != hashes:
            raise SystemExit("Tokenizer, modèle texte ou encodeur visuel modifié : reprise refusée.")
        vlm.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        rng.setstate((3, tuple(checkpoint["rng_state"]), None))
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])

    try:
        print(f"Exemples gardés: {report.kept_examples}; train={len(train_examples)}; validation={len(val_examples)}")
        for epoch in range(start_epoch, args.epochs):
            started = time.monotonic()
            shuffled = list(train_examples)
            rng.shuffle(shuffled)
            losses = []
            for example in shuffled:
                features = encoder.encode_tokens(example.image_path)
                pair = encode_pair(tokenizer, example.prompt, example.completion,
                                   model.max_seq_len, len(features))
                if pair is None:
                    continue
                ids, target_start = pair
                losses.append(vlm.train_step_features(
                    features, ids, optimizer, target_start=target_start
                ))
                global_step += 1
            val_losses = [value for example in val_examples
                          if (value := masked_loss(vlm, encoder, tokenizer, example,
                                                   model.max_seq_len)) is not None]
            train_loss = sum(losses) / len(losses) if losses else None
            val_loss = sum(val_losses) / len(val_losses) if val_losses else None
            print(f"epoch {epoch+1}/{args.epochs} train_loss={train_loss} val_loss={val_loss} steps={len(losses)} elapsed={time.monotonic()-started:.1f}s")
            vlm.save(output / "model.json")
            atomic_json(checkpoint_path, {
                "format": "ghost-vlm-sft-checkpoint-v1", "epoch": epoch + 1,
                "global_step": global_step, "model": vlm.state_dict(),
                "optimizer": optimizer.state_dict(), "rng_state": list(rng.getstate()[1]),
                "hashes": hashes, "dataset_report": report.__dict__,
            })
        print(f"VLM SFT sauvegardé : {output / 'model.json'}")
    finally:
        optimizer.free()
        vlm.free(free_language_model=True)


if __name__ == "__main__":
    main()
