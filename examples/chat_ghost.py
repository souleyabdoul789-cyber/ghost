"""Petit terminal de dialogue avec un checkpoint ghost déjà entraîné.

Exemple :
  python examples/chat_ghost.py runs/ghost-base --model model.json
  python examples/chat_ghost.py runs/ghost-sft --model tool-model.json --tokenizer runs/ghost-base/tokenizer.json

Ce programme expose la génération du modèle ; il ne transforme pas un checkpoint
pré-entraîné en assistant instruction-tuned. La qualité dépend du corpus et du SFT.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from ghost.llm import GPTModel
from ghost.text import Tokenizer


def load_model(path: Path):
    state = json.loads(path.read_text(encoding="utf-8"))
    config = state.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Checkpoint sans configuration Transformer : {path}")
    model = GPTModel(**config)
    model.load_state_dict(state)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="Dossier contenant le checkpoint et le tokenizer")
    parser.add_argument("--model", default="model.json", help="Checkpoint JSON (model.json ou tool-model.json)")
    parser.add_argument("--tokenizer", type=Path, help="Tokenizer JSON ; par défaut model_dir/tokenizer.json")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    model_path = (model_dir / args.model).resolve()
    tokenizer_path = (args.tokenizer or (model_dir / "tokenizer.json")).expanduser().resolve()
    if not model_path.is_file() or not tokenizer_path.is_file():
        parser.error("Fichiers absents : vérifier --model et --tokenizer.")
    if args.max_new_tokens < 1 or args.temperature < 0 or args.top_k < 0:
        parser.error("max-new-tokens >= 1, temperature >= 0 et top-k >= 0 requis.")

    tokenizer = Tokenizer().load(tokenizer_path)
    model = load_model(model_path)
    eos_id = tokenizer.token_to_id("<|eos|>")
    print(f"ghost {__import__('ghost').__version__} | vocab={model.vocab_size} | context={model.max_seq_len}")
    print("Dialogue local. /quit pour quitter. La génération brute peut être incohérente sans SFT.")
    try:
        while True:
            try:
                user_text = input("\nVous > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nFin du dialogue.")
                break
            if not user_text:
                continue
            if user_text.lower() in {"/quit", "/exit"}:
                break
            prompt_ids = tokenizer.encode(user_text)
            if not prompt_ids:
                print("ghost > Le tokenizer n'a produit aucun token.")
                continue
            prompt_ids = prompt_ids[-model.max_seq_len:]
            generated = model.generate(
                prompt_ids, max_new_tokens=min(args.max_new_tokens, model.max_seq_len),
                temperature=args.temperature, top_k=args.top_k,
                repetition_penalty=1.08, seed=args.seed,
            )
            answer_ids = generated[len(prompt_ids):]
            if eos_id is not None and eos_id in answer_ids:
                answer_ids = answer_ids[:answer_ids.index(eos_id)]
            print("ghost > " + tokenizer.decode(answer_ids).strip())
    finally:
        model.free()


if __name__ == "__main__":
    main()
