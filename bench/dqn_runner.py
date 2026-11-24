"""
ghost.bench.dqn_runner
----------------------------------------------------------------
Boucle d'entrainement DQN generique + mesures, partagee par toutes
les references (baselines) du banc de test. Ecrite directement avec
ghost.nn / ghost.Tensor (pas ghost.rl.DQNAgent) pour pouvoir brancher soit un
ReplayBuffer normal, soit un PrioritizedReplayBuffer, et pour pouvoir
partir d'un modele deja entraine (fine-tuning).
"""

import sys
import os
import random
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from ghost import Tensor, nn
from ghost.rl import ReplayBuffer
from prioritized_replay import PrioritizedReplayBuffer


def make_model(state_size, n_actions, hidden=64, seed=42):
    return nn.Sequential([
        nn.Dense(state_size, hidden, activation="relu", seed=seed),
        nn.Dense(hidden, hidden, activation="relu", seed=seed + 1),
        nn.Dense(hidden, n_actions, activation="none", seed=seed + 2),
    ])


def clone_model_weights(src, dst):
    for s_layer, d_layer in zip(src.layers, dst.layers):
        d_layer.weights.set_data(s_layer.weights.data)
        d_layer.bias.set_data(s_layer.bias.data)


def train_dqn(env_factory, state_size, n_actions,
              n_episodes=300, hidden=64, lr=0.05, gamma=0.9,
              batch_size=32, target_update_every=100,
              epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.97,
              use_prioritized=False, init_model=None,
              success_window=20, success_threshold=0.9,
              seed=0):
    """Entraine un DQN sur env_factory(seed) et renvoie un rapport de mesures.

    init_model : si fourni, les poids de depart sont COPIES depuis ce modele
                 (utilise pour le baseline "pre-entraine puis affine").

    Renvoie un dict :
        episodes_to_threshold : nb d'episodes avant que le taux de reussite
                                 glissant (sur `success_window` episodes)
                                 atteigne `success_threshold`, ou None si
                                 jamais atteint dans `n_episodes`.
        success_history : liste du taux de reussite glissant, episode par episode
        final_model      : le modele entraine (pour tester le transfert/l'oubli ensuite)
        env_steps_to_threshold : nb total de pas d'environnement (proxy de cout de calcul)
    """
    random.seed(seed)
    env = env_factory(seed=seed)

    model = make_model(state_size, n_actions, hidden=hidden, seed=seed)
    if init_model is not None:
        clone_model_weights(init_model, model)

    target_model = make_model(state_size, n_actions, hidden=hidden, seed=seed)
    clone_model_weights(model, target_model)

    opt = nn.SGD(model.parameters(), lr=lr)
    buffer = PrioritizedReplayBuffer(capacity=5000) if use_prioritized else ReplayBuffer(capacity=5000)

    epsilon = epsilon_start
    learn_steps = 0
    total_env_steps = 0
    successes = deque(maxlen=success_window)
    success_history = []
    episodes_to_threshold = None
    env_steps_to_threshold = None

    def act(state, eps):
        if random.random() < eps:
            return random.randrange(n_actions)
        q_tensor = model(Tensor([state]))
        q = q_tensor.data
        q_tensor.free_graph()  # libere l'entree + le graphe ephemere (PAS les poids, proteges)
        return max(range(n_actions), key=lambda i: q[i])

    def learn():
        nonlocal learn_steps
        if len(buffer) < batch_size:
            return
        if use_prioritized:
            idx, batch = buffer.sample(batch_size)
        else:
            batch = buffer.sample(batch_size)

        states = [b[0] for b in batch]
        actions = [b[1] for b in batch]
        rewards = [b[2] for b in batch]
        next_states = [b[3] for b in batch]
        dones = [b[4] for b in batch]

        pred_tensor = model(Tensor(states))
        preds = pred_tensor.data
        next_q_tensor = target_model(Tensor(next_states))
        next_q = next_q_tensor.data
        next_q_tensor.free_graph()  # graphe separe de celui de la loss -- a liberer a part

        targets = [list(row) for row in preds]
        td_errors = []
        for i in range(len(batch)):
            target_val = rewards[i] if dones[i] else rewards[i] + gamma * max(next_q[i])
            td_errors.append(target_val - preds[i][actions[i]])
            targets[i][actions[i]] = target_val

        opt.zero_grad()
        loss = nn.mse_loss(pred_tensor, Tensor(targets))
        loss.backward()
        opt.step()
        loss.free_graph()  # libere pred_tensor + tous les intermediaires (poids proteges)

        if use_prioritized:
            buffer.update_priorities(idx, td_errors)

        learn_steps += 1
        if learn_steps % target_update_every == 0:
            clone_model_weights(model, target_model)

    for ep in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            a = act(state, epsilon)
            next_state, reward, done = env.step(a)
            transition = (state, a, reward, next_state, done)
            if use_prioritized:
                buffer.push(transition)
            else:
                buffer.push(*transition)
            learn()
            state = next_state
            total_env_steps += 1

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        successes.append(1 if reward > 5 else 0)  # reward>5 seulement si objectif atteint (voir envs.py)
        success_rate = sum(successes) / len(successes)
        success_history.append(success_rate)

        if episodes_to_threshold is None and len(successes) == success_window and success_rate >= success_threshold:
            episodes_to_threshold = ep + 1
            env_steps_to_threshold = total_env_steps

    return {
        "episodes_to_threshold": episodes_to_threshold,
        "env_steps_to_threshold": env_steps_to_threshold,
        "success_history": success_history,
        "final_model": model,
        "total_env_steps": total_env_steps,
    }


def evaluate_success_rate(model, env_factory, n_episodes=30, seed=1000, n_actions=4):
    """Evalue un modele DEJA ENTRAINE (epsilon=0, pas d'apprentissage) --
    sert a mesurer le transfert et l'oubli."""
    env = env_factory(seed=seed)
    successes = 0
    for ep in range(n_episodes):
        state = env.reset()
        done = False
        last_reward = 0
        while not done:
            q_tensor = model(Tensor([state]))
            q = q_tensor.data
            q_tensor.free_graph()
            a = max(range(n_actions), key=lambda i: q[i])
            state, reward, done = env.step(a)
            last_reward = reward
        if last_reward > 5:
            successes += 1
    return successes / n_episodes
