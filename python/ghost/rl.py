"""
ghost.rl
----------------------------------------------------------------
Brique pour entrainer des agents autonomes (les futurs "robots
virtuels" dans Godot).

Agents disponibles :

  - `QAgent`   : Q-learning en ligne, un pas a la fois. Simple,
                 pedagogique, pas recommande pour un vrai agent.

  - `DQNAgent` : actions DISCRETES (un choix parmi N). Replay buffer
                 + reseau cible + entrainement par batch. `.save()` /
                 `.load()` persistent TOUT l'etat utile pour reprendre
                 l'entrainement plus tard (poids, replay buffer,
                 epsilon, compteur d'apprentissage) -- pas seulement
                 les poids.

  - `CEMAgent` : actions CONTINUES (ex: vitesse, angle, force -- un
                 vecteur de floats, pas un choix parmi N). Utilise le
                 Cross-Entropy Method : pas besoin de calculer un
                 gradient de la recompense (souvent impossible/instable
                 en continu avec peu de code) -- on evalue une
                 population de variantes des poids, on garde les
                 meilleures, on recentre la recherche dessus. Simple,
                 robuste, bien adapte a de petits reseaux.

Ce que ghost.rl NE fait PAS encore : pas de couche recurrente (utile pour
des etats sequentiels/memoire) ni de vision par convolution -- voir
docs/api.md pour le detail.
"""

import random
import json
from collections import deque
from .tensor import Tensor
from . import nn


class Environment:
    """Interface a implementer pour tout environnement a actions discretes.

    reset() -> state (liste de floats)
    step(action: int) -> (next_state, reward: float, done: bool)
    """

    n_actions: int = 0
    state_size: int = 0

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError


class ContinuousEnvironment:
    """Interface a implementer pour un environnement a actions continues.

    reset() -> state (liste de floats)
    step(action: liste de floats, une par dimension) -> (next_state, reward, done)
    """

    action_size: int = 0
    state_size: int = 0

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError


# ============================================================
# Q-learning simple (reference / petits problemes, actions discretes)
# ============================================================

class QAgent:
    """Agent Q-learning en ligne (un pas a la fois, sans replay buffer).
    Garde pour les petits problemes et la pedagogie -- pour un vrai
    entrainement robuste en actions discretes, utilise DQNAgent."""

    def __init__(self, state_size, n_actions, hidden=16, lr=0.05, gamma=0.95, seed=42):
        self.state_size = state_size
        self.n_actions = n_actions
        self.gamma = gamma
        self.model = nn.Sequential([
            nn.Dense(state_size, hidden, activation="tanh", seed=seed),
            nn.Dense(hidden, n_actions, activation="none", seed=seed + 1),
        ])
        self.opt = nn.SGD(self.model.parameters(), lr=lr)

    def q_values(self, state):
        return nn.forward_values(self.model, [state])

    def act(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        q = self.q_values(state)
        return max(range(self.n_actions), key=lambda i: q[i])

    def learn(self, state, action, reward, next_state, done):
        x = Tensor([state])
        pred_tensor = self.model(x)
        pred = pred_tensor.data

        target_value = reward if done else reward + self.gamma * max(self.q_values(next_state))
        target = list(pred)
        target[action] = target_value
        y = Tensor([target])

        self.opt.zero_grad()
        loss = nn.mse_loss(pred_tensor, y)
        loss.backward()
        self.opt.step()

        loss_value = loss.data
        loss.free_intermediates()  # libere pred_tensor + tout le calcul intermediaire
        x.free()
        y.free()
        return loss_value


# ============================================================
# Replay buffer
# ============================================================

class ReplayBuffer:
    """Stocke les transitions (s, a, r, s', done) et permet d'en tirer
    des lots aleatoires -- casse la correlation entre pas consecutifs,
    ce qui stabilise beaucoup l'entrainement par rapport a du online pur."""

    def __init__(self, capacity=5000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)

    def to_list(self):
        """Pour serialisation JSON (les tuples deviennent des listes)."""
        return [list(t) for t in self.buffer]

    def load_list(self, items):
        self.buffer = deque((tuple(t) for t in items), maxlen=self.capacity)

    def __len__(self):
        return len(self.buffer)


# ============================================================
# DQN : agent autonome a actions DISCRETES
# ============================================================

class DQNAgent:
    """DQN avec replay buffer + reseau cible + entrainement par batch,
    pour un espace d'actions discret (un choix parmi n_actions).

    Exemple :
        agent = DQNAgent(state_size=2, n_actions=4)
        for episode in range(n_episodes):
            state = env.reset()
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.learn()
                state = next_state
            agent.decay_epsilon()

        agent.save("mon_modele.json")   # sauvegarde TOUT (poids, buffer, epsilon)
        # plus tard, sur un autre appareil ou une autre session :
        agent2 = DQNAgent(state_size=2, n_actions=4)
        agent2.load("mon_modele.json")  # reprend exactement ou l'entrainement s'est arrete
    """

    def __init__(self, state_size, n_actions, hidden=32, lr=0.05, gamma=0.95,
                 buffer_size=5000, batch_size=32, target_update_every=100,
                 epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.97,
                 seed=42):
        self.state_size = state_size
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.learn_steps = 0

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        def make_net(s):
            return nn.Sequential([
                nn.Dense(state_size, hidden, activation="relu", seed=s),
                nn.Dense(hidden, hidden, activation="relu", seed=s + 1),
                nn.Dense(hidden, n_actions, activation="none", seed=s + 2),
            ])

        self.model = make_net(seed)
        self.target_model = make_net(seed)
        self._sync_target()

        self.opt = nn.SGD(self.model.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

    def _sync_target(self):
        for src, dst in zip(self.model.layers, self.target_model.layers):
            dst.weights.set_data(src.weights.data)
            dst.bias.set_data(src.bias.data)

    def act(self, state, epsilon=None):
        eps = self.epsilon if epsilon is None else epsilon
        if random.random() < eps:
            return random.randrange(self.n_actions)
        q = nn.forward_values(self.model, [state])
        return max(range(self.n_actions), key=lambda i: q[i])

    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        states = [b[0] for b in batch]
        actions = [b[1] for b in batch]
        rewards = [b[2] for b in batch]
        next_states = [b[3] for b in batch]
        dones = [b[4] for b in batch]

        x = Tensor(states)
        pred_tensor = self.model(x)
        preds = pred_tensor.data
        next_q = nn.forward_values(self.target_model, next_states)  # deja nettoye en interne

        targets = [list(row) for row in preds]
        for i in range(len(batch)):
            target_val = rewards[i] if dones[i] else rewards[i] + self.gamma * max(next_q[i])
            targets[i][actions[i]] = target_val
        y = Tensor(targets)

        self.opt.zero_grad()
        loss = nn.mse_loss(pred_tensor, y)
        loss.backward()
        self.opt.step()

        loss_value = loss.data
        loss.free_intermediates()  # libere pred_tensor + tout le calcul intermediaire
        x.free()
        y.free()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_every == 0:
            self._sync_target()

        return loss_value

    def save(self, path):
        """Sauvegarde l'etat COMPLET : poids, replay buffer, epsilon,
        compteur d'apprentissage -- de quoi reprendre l'entrainement
        exactement ou il s'est arrete (pas juste utiliser le modele)."""
        state = {
            "model": self.model.state_dict(),
            "epsilon": self.epsilon,
            "learn_steps": self.learn_steps,
            "buffer": self.buffer.to_list(),
            "buffer_capacity": self.buffer.capacity,
        }
        with open(path, "w") as f:
            json.dump(state, f)

    def load(self, path):
        with open(path) as f:
            state = json.load(f)
        self.model.load_state_dict(state["model"])
        self._sync_target()
        self.epsilon = state.get("epsilon", self.epsilon)
        self.learn_steps = state.get("learn_steps", 0)
        if "buffer_capacity" in state:
            self.buffer.capacity = state["buffer_capacity"]
        self.buffer.load_list(state.get("buffer", []))


# ============================================================
# CEM : agent autonome a actions CONTINUES
# ============================================================

class CEMAgent:
    """Politique deterministe (reseau -> tanh -> action continue),
    entrainee par Cross-Entropy Method : aucune backprop a travers la
    recompense, on cherche directement dans l'espace des poids.

    A chaque generation :
      1. On tire `population` variantes des poids (bruit gaussien
         autour de la moyenne actuelle).
      2. On evalue chaque variante sur un episode complet.
      3. On garde les meilleures (`elite_frac`), on recalcule la
         moyenne et l'ecart-type des poids a partir d'elles.
      4. On recommence -- la distribution des poids se resserre autour
         des solutions qui marchent.

    Les actions renvoyees par `.act()` sont dans [-1, 1] par dimension
    (sortie d'un tanh) -- a toi de les remettre a l'echelle dans ton
    `ContinuousEnvironment.step()` (ex: `action[0] * vitesse_max`).

    Exemple :
        agent = CEMAgent(state_size=1, action_size=1, population=40)
        for generation in range(50):
            avg_reward, best_reward = agent.train_generation(env, episode_len=100)
            print(generation, avg_reward, best_reward)

        agent.save("mon_agent_continu.json")
    """

    def __init__(self, state_size, action_size, hidden=32, population=40,
                 elite_frac=0.2, init_std=0.5, min_std=0.05, seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.population = population
        self.elite_frac = elite_frac
        self.min_std = min_std

        self.model = nn.Sequential([
            nn.Dense(state_size, hidden, activation="tanh", seed=seed),
            nn.Dense(hidden, action_size, activation="tanh", seed=seed + 1),
        ])

        self.n_params = len(self._get_flat_params())
        self.std = [init_std] * self.n_params

    # -- gestion des poids sous forme de vecteur plat --------------

    def _get_flat_params(self):
        flat = []
        for layer in self.model.layers:
            flat.extend(layer.weights.flat_data())
            flat.extend(layer.bias.flat_data())
        return flat

    def _set_flat_params(self, flat):
        idx = 0
        for layer in self.model.layers:
            wsize = layer.in_features * layer.out_features
            layer.weights.set_flat_data(flat[idx:idx + wsize])
            idx += wsize
            bsize = layer.out_features
            layer.bias.set_flat_data(flat[idx:idx + bsize])
            idx += bsize

    # -- action / evaluation -----------------------------------------

    def act(self, state):
        out = nn.forward_values(self.model, [state])
        if isinstance(out, (int, float)):  # cas action_size == 1 -> Tensor 1x1
            out = [out]
        return out

    def _evaluate(self, env, episode_len):
        state = env.reset()
        total = 0.0
        for _ in range(episode_len):
            action = self.act(state)
            state, reward, done = env.step(action)
            total += reward
            if done:
                break
        return total

    # -- entrainement (CEM) --------------------------------------------

    def train_generation(self, env, episode_len=200):
        """Un cycle CEM complet. Renvoie (recompense_moyenne, meilleure_recompense)
        de la population evaluee. A appeler en boucle (une fois par generation)."""
        base = self._get_flat_params()
        candidates = []
        rewards = []

        for _ in range(self.population):
            noise = [random.gauss(0.0, s) for s in self.std]
            candidate = [b + n for b, n in zip(base, noise)]
            candidates.append(candidate)
            self._set_flat_params(candidate)
            rewards.append(self._evaluate(env, episode_len))

        n_elite = max(1, int(self.population * self.elite_frac))
        elite_idx = sorted(range(len(rewards)), key=lambda i: rewards[i], reverse=True)[:n_elite]
        elite = [candidates[i] for i in elite_idx]

        n = len(base)
        new_mean = [sum(e[j] for e in elite) / n_elite for j in range(n)]
        new_std = []
        for j in range(n):
            var = sum((e[j] - new_mean[j]) ** 2 for e in elite) / n_elite
            new_std.append(max(self.min_std, var ** 0.5))

        self.std = new_std
        self._set_flat_params(new_mean)

        return sum(rewards) / len(rewards), max(rewards)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"params": self._get_flat_params(), "std": self.std}, f)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self._set_flat_params(data["params"])
        self.std = data["std"]
