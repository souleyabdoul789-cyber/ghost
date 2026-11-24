"""Entraîne un Transformer GPT miniature natif G sur un corpus UTF-8.

Exemple :
  python examples/train_transformer.py corpus.txt --epochs 10 --d-model 64
  python examples/train_transformer.py --corpus corpus.txt --output runs/g0
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from ghost import llm, nn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_pos", nargs="?", help="Fichier texte UTF-8")
    parser.add_argument("--corpus", dest="corpus_option", help="Fichier texte UTF-8")
    parser.add_argument("--output", default=None, help="Dossier de sortie (par défaut : dossier du corpus)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--vocab-size", "--vocab", dest="vocab_size", type=int, default=512)
    parser.add_argument("--steps-per-epoch", "--steps", dest="steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    corpus_path = args.corpus_option or args.corpus_pos
    if not corpus_path:
        parser.error("Indique un corpus en argument ou avec --corpus.")
    if args.epochs <= 0 or args.context < 2 or args.vocab_size < 256:
        parser.error("epochs > 0, context >= 2 et vocab-size >= 256 requis.")
    if args.steps is not None and args.steps <= 0:
        parser.error("steps-per-epoch doit être positif.")
    if args.d_model <= 0 or args.heads <= 0 or args.d_model % args.heads:
        parser.error("d-model doit être positif et divisible par heads.")
    if args.layers <= 0 or args.learning_rate <= 0:
        parser.error("layers et learning-rate doivent être positifs.")

    with open(corpus_path, "r", encoding="utf-8") as stream:
        corpus = stream.read()
    if len(corpus.strip()) < 2:
        raise SystemExit("Le corpus est vide ou trop court.")

    tokenizer = llm.Tokenizer()
    tokenizer.train([corpus], vocab_size=args.vocab_size, verbose=True)
    ids = tokenizer.encode(corpus)
    if len(ids) < 2:
        raise SystemExit("Le corpus produit moins de deux tokens.")

    model = llm.TransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        max_seq_len=args.context,
        seed=args.seed,
    )
    optimizer = nn.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    try:
        history = model.fit(
            ids, optimizer, epochs=args.epochs, steps_per_epoch=args.steps,
            seq_len=args.context, seed=args.seed,
        )
        for epoch, loss in enumerate(history, 1):
            print(f"epoch {epoch:04d} | loss {loss:.5f}")

        output_dir = args.output or os.path.dirname(os.path.abspath(corpus_path))
        os.makedirs(output_dir, exist_ok=True)
        tokenizer_path = os.path.join(output_dir, "ghost-tokenizer.json")
        model_path = os.path.join(output_dir, "g-transformer.json")
        tokenizer.save(tokenizer_path)
        model.save(model_path)
        prompt = ids[:min(12, len(ids))]
        sample = model.generate(prompt, max_new_tokens=80, temperature=0.8, top_k=40)
        print("\n--- G GENERATED TEXT ---\n")
        print(tokenizer.decode(sample))
        print("\nTokenizer :", tokenizer_path)
        print("Modèle    :", model_path)
    finally:
        optimizer.free()
        model.free()


if __name__ == "__main__":
    main()
