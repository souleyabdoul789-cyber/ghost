import tempfile
import unittest
from pathlib import Path

from ghost.motivation import CuriosityPolicy, GoalCandidate


class MotivationTests(unittest.TestCase):
    def test_approved_goals_only_domain_and_budget(self):
        policy = CuriosityPolicy(allowed_domains=["python.org"], action_budget=1)
        candidates = [
            GoalCandidate("denied", "Lire", "docs", approved=False),
            GoalCandidate("bad-domain", "Lire", "docs", domain="example.net", approved=True),
            GoalCandidate("good", "Étudier pathlib", "pathlib", priority=.8,
                          domain="docs.python.org", approved=True),
        ]
        chosen = policy.choose(candidates)
        self.assertEqual(chosen.goal_id, "good")
        policy.record(chosen, success=True, note="Lecture validée")
        self.assertIsNone(policy.choose(candidates))

    def test_state_persistence_and_repetition_penalty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "drive.json"
            policy = CuriosityPolicy(state_path=path, action_budget=5)
            first = GoalCandidate("a", "Explorer A", "topic-a", approved=True)
            other = GoalCandidate("b", "Explorer B", "topic-b", priority=.5, approved=True)
            policy.record(first, success=True)
            restored = CuriosityPolicy(state_path=path)
            self.assertEqual(restored.topic_counts["topic-a"], 1)
            self.assertEqual(restored.actions_used, 1)
            self.assertEqual(restored.choose([first, other]).goal_id, "b")


if __name__ == "__main__":
    unittest.main()
