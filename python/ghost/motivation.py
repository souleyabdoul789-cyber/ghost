"""ghost.motivation — priorisation par curiosité simulée, sans prétention de sentience.

La politique choisit uniquement parmi les objectifs approuvés par l'utilisateur,
avec domaine réseau autorisé, budget et coût explicites. Elle ne crée pas de
permission, ne déclenche pas de tâche seule et ne modifie pas les poids du LLM.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path


@dataclass(frozen=True)
class GoalCandidate:
    goal_id: str
    description: str
    topic: str
    priority: float = 0.5
    uncertainty: float = 0.5
    expected_value: float = 0.5
    estimated_cost: float = 0.2
    domain: str = ""
    approved: bool = False

    def __post_init__(self):
        if not self.goal_id.strip() or not self.description.strip() or not self.topic.strip():
            raise ValueError("Un objectif doit avoir un identifiant, une description et un sujet.")
        for name in ("priority", "uncertainty", "expected_value", "estimated_cost"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} doit être fini et dans [0,1].")


class CuriosityPolicy:
    """Score nouveauté × incertitude × utilité, pénalisé par coût et répétition."""

    FORMAT = "ghost-curiosity-v1"

    def __init__(self, *, allowed_domains=(), action_budget=20,
                 state_path: str | os.PathLike | None = None):
        self.allowed_domains = {str(domain).lower().strip(".") for domain in allowed_domains}
        self.action_budget = max(0, min(int(action_budget), 10000))
        self.actions_used = 0
        self.state_path = Path(state_path) if state_path else None
        self.topic_counts: dict[str, int] = {}
        self.outcomes: dict[str, dict] = {}
        if self.state_path and self.state_path.is_file():
            self.load(self.state_path)

    def score(self, candidate: GoalCandidate) -> float:
        if not candidate.approved:
            return float("-inf")
        if candidate.domain:
            domain = candidate.domain.lower().strip(".")
            if not any(domain == allowed or domain.endswith("." + allowed)
                       for allowed in self.allowed_domains):
                return float("-inf")
        visits = self.topic_counts.get(candidate.topic.casefold(), 0)
        novelty = 1.0 / (1.0 + visits)
        # L'incertitude augmente l'intérêt d'une exploration, mais l'utilité et
        # la priorité restent plus importantes que la simple nouveauté.
        raw = (0.30 * candidate.priority + 0.25 * candidate.expected_value
               + 0.20 * candidate.uncertainty + 0.25 * novelty)
        return raw * (1.0 - 0.65 * candidate.estimated_cost)

    def choose(self, candidates):
        if self.actions_used >= self.action_budget:
            return None
        valid = [candidate for candidate in candidates if isinstance(candidate, GoalCandidate)]
        ranked = [(self.score(candidate), candidate.goal_id, candidate)
                  for candidate in valid]
        ranked = [item for item in ranked if math.isfinite(item[0])]
        if not ranked:
            return None
        return max(ranked, key=lambda item: (item[0], item[1]))[2]

    def record(self, candidate: GoalCandidate, *, success: bool, note=""):
        topic = candidate.topic.casefold()
        self.topic_counts[topic] = self.topic_counts.get(topic, 0) + 1
        self.actions_used += 1
        self.outcomes[candidate.goal_id] = {
            "topic": candidate.topic, "success": bool(success),
            "note": str(note)[:2000], "actions_used": self.actions_used,
        }
        if self.state_path:
            self.save(self.state_path)

    def state_dict(self):
        return {"format": self.FORMAT, "action_budget": self.action_budget,
                "actions_used": self.actions_used, "topic_counts": self.topic_counts,
                "outcomes": self.outcomes}

    def load(self, path):
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        if state.get("format") != self.FORMAT:
            raise ValueError("État de curiosité ghost non reconnu.")
        counts = state.get("topic_counts", {})
        outcomes = state.get("outcomes", {})
        if not isinstance(counts, dict) or not isinstance(outcomes, dict):
            raise ValueError("État de curiosité corrompu.")
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in counts.values()):
            raise ValueError("Compteurs de sujets invalides.")
        used = state.get("actions_used", 0)
        if not isinstance(used, int) or isinstance(used, bool) or used < 0:
            raise ValueError("Budget d'actions utilisé invalide.")
        self.action_budget = max(0, min(int(state.get("action_budget", self.action_budget)), 10000))
        self.actions_used = used
        self.topic_counts = {str(k): v for k, v in counts.items()}
        self.outcomes = outcomes

    def save(self, path=None):
        destination = Path(path) if path else self.state_path
        if destination is None:
            raise ValueError("Aucun chemin d'état de curiosité configuré.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(self.state_dict(), stream, ensure_ascii=False, separators=(",", ":"))
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, destination)
