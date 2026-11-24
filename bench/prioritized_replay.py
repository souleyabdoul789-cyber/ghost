"""
ghost.bench.prioritized_replay
----------------------------------------------------------------
Replay buffer avec priorite = |erreur TD| : les transitions mal
predites sont rejouees plus souvent. Utilise comme UNE DES
REFERENCES a battre (pas encore l'arbitre complet de G-Brain, juste
la brique "priorite = erreur" isolee du reste de l'hypothese).
"""

import random


class PrioritizedReplayBuffer:
    def __init__(self, capacity=5000, alpha=0.6, eps=1e-3):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.data = []       # liste de transitions (s,a,r,s2,done)
        self.priorities = []  # priorite associee (float > 0)
        self._pos = 0

    def push(self, transition, priority=None):
        max_p = max(self.priorities) if self.priorities else 1.0
        p = priority if priority is not None else max_p
        if len(self.data) < self.capacity:
            self.data.append(transition)
            self.priorities.append(p)
        else:
            self.data[self._pos] = transition
            self.priorities[self._pos] = p
            self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size):
        n = len(self.data)
        batch_size = min(batch_size, n)
        weights = [p ** self.alpha for p in self.priorities]
        total = sum(weights)
        probs = [w / total for w in weights]
        idx = random.choices(range(n), weights=probs, k=batch_size)
        batch = [self.data[i] for i in idx]
        return idx, batch

    def update_priorities(self, idx, td_errors):
        for i, err in zip(idx, td_errors):
            self.priorities[i] = abs(err) + self.eps

    def __len__(self):
        return len(self.data)
