"""Fine-tuning vision-langage sur un manifeste local image→caption.

JSONL: {"image":"relative/path.jpg", "text":"caption", "source":"...", "license":"..."}
Le modèle texte doit exister dans --base. L'encodeur ONNX fourni reste gelé ;
le projecteur visuel et le GPT reçoivent les gradients.

Exemple:
  python examples/train_ghost_vision.py captions.jsonl --images ./images \
      --base runs/ghost-base --vision-encoder vision_encoder.onnx \
      --feature-dim 512 --output runs/ghost-vision --epochs 2
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


def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def encode_caption(tokenizer, text, max_context, visual_tokens):
    ids = tokenizer.encode(text)
    available = max_context - visual_tokens
    if available < 1:
        return None
    # Le VLM prédit le premier token depuis les patches, puis les suivants
    # depuis les tokens précédents : il faut au plus available labels.
    return ids[:available] if ids else None


def evaluate(vlm, encoder, tokenizer, examples, max_context, limit=16):
    values = []
    for example in examples[:limit]:
        # Pour l'encodeur ONNX, il faut connaître le nombre de patches avant
        # de choisir la longueur de la caption ; forward valide le contexte.
        try:
            features = encoder.encode_tokens(example.image_path)
            ids = encode_caption(tokenizer, example.text, max_context, len(features))
            if not ids:
                continue
            logits = vlm.forward_features(features, ids)
            loss = nn.cross_entropy_loss_ids(logits, ids)
            values.append(float(loss.data))
            loss.free_graph()
        except ValueError as exc:
            if "contexte" not in str(exc) and "séquence" not in str(exc):
                raise
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="Manifeste JSONL image→caption")
    parser.add_argument("--images", required=True, help="Dossier racine des images")
    parser.add_argument("--base", required=True, help="Dossier de pré-entraînement avec tokenizer.json/model.json")
    parser.add_argument("--vision-encoder", required=True, help="Encodeur image exporté au format ONNX")
    parser.add_argument("--feature-dim", required=True, type=int, help="Dimension feature de l'encodeur")
    parser.add_argument("--spatial-layout", choices=["nchw", "nhwc"], default="nchw")
    parser.add_argument("--output", default="runs/ghost-vision")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=0, help="Limiter le nombre d'exemples par époque (0 = tout)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.feature_dim <= 0 or args.epochs < 1 or args.learning_rate <= 0 or args.weight_decay < 0 or args.limit < 0:
        parser.error("feature-dim/epochs/lr/limit doivent être valides")

    base = Path(args.base).expanduser().resolve()
    tokenizer_path, model_path = base / "tokenizer.json", base / "model.json"
    if not tokenizer_path.is_file() or not model_path.is_file():
        parser.error("--base doit contenir tokenizer.json et model.json")
    tokenizer = Tokenizer().load(tokenizer_path)
    base_state = json.loads(model_path.read_text(encoding="utf-8"))
    config = base_state.get("config")
    if not isinstance(config, dict):
        parser.error("model.json ne contient pas une configuration GPT valide")

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    image_data_dir = output / "image-data"
    report = data.ImageTextDatasetBuilder(args.images, seed=args.seed).build(args.manifest, image_data_dir)
    train_examples = list(data.iter_image_split(image_data_dir, "train"))
    val_examples = list(data.iter_image_split(image_data_dir, "validation"))
    if args.limit:
        train_examples = train_examples[:args.limit]
    if not train_examples:
        raise SystemExit("Le split train image-texte est vide.")

    model = GPTModel(
        vocab_size=config["vocab_size"], d_model=config["d_model"],
        n_heads=config["n_heads"], n_layers=config["n_layers"],
        max_seq_len=config["max_seq_len"], ff_mult=config["ff_mult"], seed=args.seed,
    )
    model.load(model_path)
    encoder = OnnxVisionEncoder(args.vision_encoder, spatial_layout=args.spatial_layout)
    vlm = VisionLanguageModel(model, encoder, feature_dim=args.feature_dim, seed=args.seed + 1)
    optimizer = nn.AdamW(vlm.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    tokenizer_hash = sha256_file(tokenizer_path)
    base_hash = sha256_file(model_path)
    encoder_hash = sha256_file(args.vision_encoder)
    checkpoint_path = output / "checkpoint.json"
    start_epoch = global_step = 0

    if args.resume and checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("format") != "ghost-vision-training-v1":
            raise SystemExit("Format de checkpoint vision inconnu.")
        for key, current in (("tokenizer_sha256", tokenizer_hash),
                             ("base_sha256", base_hash), ("encoder_sha256", encoder_hash)):
            if checkpoint.get(key) != current:
                raise SystemExit(f"{key} modifié : reprise refusée.")
        vlm.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        rng.setstate((3, tuple(checkpoint["rng_state"]), None))
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])

    try:
        print(f"Paires gardées: {report.kept_pairs}; train={len(train_examples)}; validation={len(val_examples)}")
        for epoch in range(start_epoch, args.epochs):
            started = time.monotonic()
            shuffled = list(train_examples)
            rng.shuffle(shuffled)
            losses = []
            for example in shuffled:
                try:
                    features = encoder.encode_tokens(example.image_path)
                    ids = encode_caption(tokenizer, example.text, model.max_seq_len, len(features))
                    if not ids:
                        continue
                    loss = vlm.train_step_features(features, ids, optimizer)
                except ValueError as exc:
                    if "contexte" in str(exc) or "séquence" in str(exc):
                        continue
                    raise
                losses.append(loss)
                global_step += 1
            val_loss = evaluate(vlm, encoder, tokenizer, val_examples, model.max_seq_len)
            train_loss = sum(losses) / len(losses) if losses else None
            print(f"epoch {epoch+1}/{args.epochs} train_loss={train_loss} val_loss={val_loss} steps={len(losses)} elapsed={time.monotonic()-started:.1f}s")
            vlm.save(str(output / "model.json"))
            atomic_json(checkpoint_path, {
                "format": "ghost-vision-training-v1", "epoch": epoch + 1,
                "global_step": global_step, "model": vlm.state_dict(),
                "optimizer": optimizer.state_dict(), "rng_state": list(rng.getstate()[1]),
                "tokenizer_sha256": tokenizer_hash, "base_sha256": base_hash,
                "encoder_sha256": encoder_hash, "dataset_report": report.__dict__,
            })
        print(f"VLM sauvegardé : {output / 'model.json'}")
    finally:
        optimizer.free()
        vlm.free(free_language_model=True)


if __name__ == "__main__":
    main()
