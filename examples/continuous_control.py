"""
Exemple : controle CONTINU avec CEMAgent. Contrairement a DQNAgent
(qui choisit parmi un nombre fixe d'actions), ici l'agent produit un
vecteur de floats -- utile pour piloter une vitesse, un angle, une
force en continu (plus realiste pour un robot/personnage 3D que 4
directions figees).

Tache jouet : un point sur une ligne doit atteindre une position
cible. L'action est un deplacement continu dans [-1, 1].

Lancer : python3 continuous_control.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import random
from ghost.rl import ContinuousEnvironment, CEMAgent


class ReachTarget(ContinuousEnvironment):
    state_size = 1
    action_size = 1

    def __init__(self, target=5.0):
        self.target = target

    def reset(self):
        self.pos = 0.0
        self.t = 0
        return [self.pos]

    def step(self, action):
        # action[0] est dans [-1, 1] (sortie tanh du reseau) -> on l'utilise
        # directement comme deplacement, remis a l'echelle ici si besoin.
        self.pos += action[0] * 0.5
        self.t += 1
        reward = -abs(self.pos - self.target)  # plus proche de la cible = mieux
        done = self.t >= 40
        return [self.pos], reward, done


def main():
    random.seed(0)
    env = ReachTarget(target=5.0)
    agent = CEMAgent(state_size=1, action_size=1, hidden=16, population=30, elite_frac=0.2)

    n_generations = 25
    for gen in range(n_generations):
        avg_reward, best_reward = agent.train_generation(env, episode_len=40)
        if gen % 5 == 0 or gen == n_generations - 1:
            print(f"generation {gen:2d}  recompense moyenne={avg_reward:7.2f}  meilleure={best_reward:7.2f}")

    print("\n--- trajectoire finale (politique moyenne) ---")
    state = env.reset()
    trajectory = [state[0]]
    done = False
    while not done:
        action = agent.act(state)
        state, reward, done = env.step(action)
        trajectory.append(state[0])
    print("positions :", [round(p, 2) for p in trajectory])
    print(f"position finale : {trajectory[-1]:.2f}  (cible : {env.target})")

    save_path = os.path.join(os.path.dirname(__file__), "continuous_model.json")
    agent.save(save_path)
    print(f"\nModele sauvegarde dans {save_path}")


if __name__ == "__main__":
    main()
