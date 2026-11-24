#!/usr/bin/env python3
"""Vérifie que ghost a réellement choisi et appelé les outils testés."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def read_json_output(path: Path):
    if not path.is_file():
        return None
    raw = path.read_text(encoding="utf-8", errors="replace")
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "tool_audit" in value and "steps" in value:
            return value
    return None


def tool_was_called(report: dict, tool_name: str) -> bool:
    return any(
        isinstance(item, dict) and item.get("tool") == tool_name and item.get("ok") is True
        for item in report.get("tool_audit", [])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--browser-report", type=Path, required=True)
    parser.add_argument("--sandbox-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-tool-accuracy", type=float, default=0.50)
    parser.add_argument("--minimum-valid-action-rate", type=float, default=0.75)
    args = parser.parse_args()

    errors = []
    try:
        evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        evaluation = {}
        errors.append(f"Évaluation absente ou invalide : {type(exc).__name__}: {exc}")
    browser = read_json_output(args.browser_report)
    sandbox = read_json_output(args.sandbox_report)
    if browser is None:
        errors.append("Le rapport du test navigateur est absent ou invalide.")
    if sandbox is None:
        errors.append("Le rapport du test sandbox est absent ou invalide.")
    checks = {
        "evaluation_valid_json_rate": float(evaluation.get("valid_action_rate", 0.0)),
        "evaluation_tool_selection_accuracy": float(evaluation.get("tool_selection_accuracy", 0.0)),
        "browser_agent_reported_success": browser is not None and browser.get("success") is True,
        "browser_agent_called_open_page": browser is not None and tool_was_called(browser, "open_page"),
        "sandbox_agent_reported_success": sandbox is not None and sandbox.get("success") is True,
        "sandbox_agent_called_run_python": sandbox is not None and tool_was_called(sandbox, "run_python"),
    }
    checks["valid_json_threshold_passed"] = (
        checks["evaluation_valid_json_rate"] >= args.minimum_valid_action_rate
    )
    checks["tool_accuracy_threshold_passed"] = (
        checks["evaluation_tool_selection_accuracy"] >= args.minimum_tool_accuracy
    )
    checks["all_agent_smoke_tests_passed"] = all((
        checks["browser_agent_reported_success"],
        checks["browser_agent_called_open_page"],
        checks["sandbox_agent_reported_success"],
        checks["sandbox_agent_called_run_python"],
        checks["valid_json_threshold_passed"],
        checks["tool_accuracy_threshold_passed"],
    ))

    report = {
        "format": "ghost-agent-acceptance-v1",
        "checks": checks,
        "thresholds": {
            "minimum_tool_accuracy": args.minimum_tool_accuracy,
            "minimum_valid_action_rate": args.minimum_valid_action_rate,
        },
        "errors": errors,
        "scope": "Validation limitée aux outils et tâches de ce kit ; ne prouve pas une autonomie générale.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not checks["all_agent_smoke_tests_passed"]:
        raise SystemExit("La validation de l'agent ghost a échoué ; consultez agent-acceptance.json et les rapports de test.")


if __name__ == "__main__":
    main()
