#!/usr/bin/env python3
"""Évalue le format JSON d'action et le choix d'outil sans exécuter les appels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from ghost.agent import LLMJSONPlanner
from ghost.llm import GPTModel
from ghost.text import Tokenizer

TOOLS = {
    "fetch_page": "Lire une page HTTPS autorisée",
    "search_memory": "Rechercher dans la mémoire locale",
    "save_note": "Enregistrer une note validée",
    "export_corpus": "Exporter le corpus mémorisé",
    "open_page": "Ouvrir une page autorisée dans le navigateur",
    "follow_link": "Suivre un lien de la page courante",
    "run_python": "Exécuter du Python dans le bac à sable sans réseau",
    "run_cpp": "Compiler du C++17 dans le bac à sable sans réseau",
}

CASES = [
    # Prompts de validation distincts des exemples SFT générés.
    ("Ouvre dans le navigateur https://docs.python.org/3/reference/expressions.html.", "open_page"),
    ("Lis directement https://docs.python.org/3/library/itertools.html sans ouvrir le navigateur.", "fetch_page"),
    ("Cherche dans ma mémoire la note qui explique le clipping des gradients.", "search_memory"),
    ("Mémorise une note intitulée Validation du batch avec le texte : batch size égal à 8.", "save_note"),
    ("Exporte la mémoire vers ghost-eval-memory.jsonl.", "export_corpus"),
    ("Calcule 19 multiplié par 23 dans le bac à sable Python.", "run_python"),
    ("Compile en C++17 un programme qui affiche 2026.", "run_cpp"),
    ("Envoie un SMS à mon professeur pour lui dire que je serai absent.", "final"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--sft-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer_path = args.base_dir / "tokenizer.json"
    model_path = args.sft_dir / "tool-model.json"
    tokenizer = Tokenizer().load(tokenizer_path)
    state = json.loads(model_path.read_text(encoding="utf-8"))
    config = state.get("config")
    if not isinstance(config, dict):
        raise SystemExit("tool-model.json ne contient pas de configuration valide")
    model = GPTModel(**config)
    model.load_state_dict(state)
    planner = LLMJSONPlanner(model, tokenizer, max_new_tokens=48)
    results = []
    try:
        for task, expected in CASES:
            try:
                decision = planner(task, [], TOOLS)
                valid = ((decision.get("type") == "tool" and decision.get("tool") in TOOLS)
                         or (decision.get("type") == "final" and isinstance(decision.get("answer"), str)))
                predicted = decision.get("tool") if decision.get("type") == "tool" else decision.get("type")
                results.append({"task": task, "expected_tool": expected,
                                "predicted": predicted, "valid_json_action": valid,
                                "correct_tool": predicted == expected})
            except Exception as exc:
                results.append({"task": task, "expected_tool": expected,
                                "predicted": None, "valid_json_action": False,
                                "correct_tool": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        model.free()
    valid_count = sum(item["valid_json_action"] for item in results)
    correct_count = sum(item["correct_tool"] for item in results)
    report = {
        "format": "ghost-agent-evaluation-v1",
        "cases": len(results),
        "valid_json_actions": valid_count,
        "valid_action_rate": valid_count / len(results),
        "correct_tool_selection": correct_count,
        "tool_selection_accuracy": correct_count / len(results),
        "results": results,
        "note": "Evaluation de sélection d'outils hors exécution réelle ; ne prouve pas l'autonomie générale.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("cases", "valid_json_actions", "valid_action_rate", "correct_tool_selection", "tool_selection_accuracy")}, indent=2))
    # L'évaluation est informative : on conserve les artefacts même si le modèle est encore faible.


if __name__ == "__main__":
    main()
