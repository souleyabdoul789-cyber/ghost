"""Lance la boucle LLM→outils→observations de ghost, avec permissions explicites.

Exemple texte :
  python examples/run_ghost_agent.py runs/ghost-tools \
      --model tool-model.json --tokenizer runs/ghost-base/tokenizer.json \
      --allow-domain docs.python.org \
      --task "Lis la documentation autorisée et trouve la page sur pathlib." \
      --browser

Vision optionnelle :
  --vision-onnx encoder.onnx --vision-model runs/ghost-vlm/vision-model.json \
  --vision-feature-dim 768

L'agent ne reçoit que les outils explicitement activés et les domaines passés.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from ghost.agent import AgentRuntime, LLMJSONPlanner, create_research_tools
from ghost.browser import BrowserTool
from ghost.llm import GPTModel
from ghost.memory import GMemory
from ghost.sandbox import CodeSandbox
from ghost.text import Tokenizer
from ghost.vision import OnnxVisionEncoder, VisionLanguageModel
from ghost.web import WebReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--model", default="tool-model.json")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--allow-domain", action="append", required=True,
                        help="Domaine HTTPS autorisé ; répétable, ex. docs.python.org")
    parser.add_argument("--task", required=True)
    parser.add_argument("--memory", type=Path, default=Path("ghost-memory.sqlite3"))
    parser.add_argument("--browser", action="store_true", help="Activer Playwright Chromium")
    parser.add_argument("--enable-code", action="store_true",
                        help="Activer run_python/run_cpp uniquement via bubblewrap sur Linux")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--vision-onnx", type=Path)
    parser.add_argument("--vision-model", type=Path)
    parser.add_argument("--vision-feature-dim", type=int)
    args = parser.parse_args()

    if bool(args.vision_onnx) != bool(args.vision_model) or bool(args.vision_onnx) != bool(args.vision_feature_dim):
        parser.error("Vision : fournir ensemble --vision-onnx, --vision-model et --vision-feature-dim.")
    model_dir = args.model_dir.expanduser().resolve()
    model_path = (model_dir / args.model).resolve()
    tokenizer_path = args.tokenizer.expanduser().resolve()
    if not tokenizer_path.is_file():
        parser.error("Tokenizer introuvable.")
    tokenizer = Tokenizer().load(tokenizer_path)

    encoder = None
    vlm = None
    if args.vision_onnx:
        # Un checkpoint VLM contient déjà son propre état du Transformer.
        if not args.vision_model.is_file():
            parser.error("Checkpoint vision-langage introuvable.")
        vlm_state = json.loads(args.vision_model.read_text(encoding="utf-8"))
        language_state = vlm_state.get("language_model", {})
        config = language_state.get("config")
        if not isinstance(config, dict):
            parser.error("Le checkpoint VLM ne contient pas de configuration GPT valide.")
        model = GPTModel(**config)
        encoder = OnnxVisionEncoder(args.vision_onnx)
        vlm = VisionLanguageModel(model, encoder, feature_dim=args.vision_feature_dim)
        vlm.load_state_dict(vlm_state)
    else:
        if not model_path.is_file():
            parser.error("Checkpoint textuel introuvable.")
        state = json.loads(model_path.read_text(encoding="utf-8"))
        config = state.get("config")
        if not isinstance(config, dict):
            parser.error("Checkpoint sans configuration GPT valide.")
        model = GPTModel(**config)
        model.load_state_dict(state)

    reader = WebReader(args.allow_domain)
    memory = GMemory(args.memory)
    browser = BrowserTool(reader) if args.browser else None
    sandbox = CodeSandbox() if args.enable_code else None
    if sandbox is not None and not sandbox.available:
        print("AVERTISSEMENT : bubblewrap indisponible ; les outils de code refuseront l'exécution.")

    tools = create_research_tools(reader, memory, browser=browser, code_sandbox=sandbox)
    planner = LLMJSONPlanner(model, tokenizer, vision_model=vlm)
    runtime = AgentRuntime(planner, tools, max_steps=args.max_steps, timeout_seconds=args.timeout)
    try:
        result = runtime.run(args.task)
        print(json.dumps({
            "success": result.success,
            "answer": result.answer,
            "steps": result.steps,
            "elapsed_seconds": round(result.elapsed_seconds, 3),
            "tool_audit": tools.audit_log,
        }, ensure_ascii=False, indent=2, default=str))
    finally:
        if browser is not None:
            browser.close()
        memory.close()
        if vlm is not None:
            vlm.free(free_language_model=True)
        else:
            model.free()


if __name__ == "__main__":
    main()
