"""
ghost.bench.envs
----------------------------------------------------------------
4 variantes d'un monde en grille, pour tester si un agent transfere
une regle apprise plutot que de memoriser une carte precise.

Regle cachee, stable dans les 4 variantes :
    "certaines cases bloquent le passage ; il faut les contourner
     pour atteindre l'objectif."

Ce qui change :
    A -> reference (obstacles fixes, encodage canonique)
    B -> memes positions qu'A, mais le canal qui code "obstacle" et
         celui qui code "objectif" sont permutes (surface differente,
         structure identique) -- teste si l'agent a appris la RELATION
         ou juste memorise des vecteurs bruts.
    C -> encodage identique a A, mais disposition différente des
         obstacles/objectif -- teste la generalisation spatiale.
    D -> encodage et disposition proches de A, mais une regle
         supplementaire est ajoutee (une porte bloque le passage tant
         que l'agent n'a pas ramasse une cle) -- teste la
         generalisation compositionnelle (2 regles combinees).

Etat renvoye : toujours un vecteur de taille fixe (48 floats), pour
que le MEME reseau (DQN) puisse etre reutilise/affine d'une variante
a l'autre -- condition necessaire pour mesurer un vrai transfert.

Encodage EGOCENTRIQUE (fenetre locale 3x3 + direction vers l'objectif)
plutot que la grille entiere en one-hot : la version "grille entiere"
(125+ dims, tres creuse) s'est averee bien trop lente a apprendre pour
notre moteur (SGD simple, sans Adam/momentum) dans un budget
d'episodes raisonnable -- verifie empiriquement avant d'ecrire cette
version. Une premiere version avec seulement les 4 voisins orthogonaux
s'est aussi averee INSUFFISANTE : verifie par calcul (comparaison a la
politique BFS optimale) qu'elle laissait des cases ambigues (meme vue
locale, action optimale differente selon la case). La fenetre 3x3
leve cette ambiguite (verifie : 0 conflit sur la variante A) :

    fenetre 3x3 autour de l'agent (9 cases, one-hot 5 canaux)  = 45
  + vecteur relatif vers l'objectif (dx, dy) normalise          = 2
  + a_la_cle (0 ou 1)                                           = 1
  ------------------------------------------------------------------
  = 48
"""

import random

GRID_SIZE = 5
N_CHANNELS = 5  # empty, obstacle, goal, key, door (dans cet ordre pour le canal CANONIQUE)
STATE_SIZE = 9 * N_CHANNELS + 2 + 1  # = 48 (fenetre 3x3 + direction + cle)
N_ACTIONS = 4  # haut, bas, gauche, droite

CANONICAL_MAP = {"empty": 0, "obstacle": 1, "goal": 2, "key": 3, "door": 4}
PERMUTED_MAP = {"empty": 0, "obstacle": 2, "goal": 1, "key": 3, "door": 4}  # obstacle/goal swap (variante B)


class GridBenchEnv:
    """Environnement generique. `layout` decrit la carte fixe (obstacles,
    objectif, et optionnellement cle+porte) ; `channel_map` decrit quel
    canal one-hot represente quel type de case."""

    n_actions = N_ACTIONS
    state_size = STATE_SIZE

    def __init__(self, layout, channel_map, max_steps=50, seed=None):
        self.layout = layout  # dict: obstacles(list), goal(tuple), key(tuple|None), door(tuple|None), start(tuple)
        self.channel_map = channel_map
        self.max_steps = max_steps
        self._rng = random.Random(seed)

    def reset(self):
        self.x, self.y = self.layout["start"]
        self.has_key = False
        self.steps = 0
        self._door_open = self.layout.get("door") is None
        return self._encode_state()

    def _cell_type_at(self, x, y):
        if (x, y) in self.layout.get("obstacles", []):
            return "obstacle"
        if (x, y) == self.layout.get("goal"):
            return "goal"
        if (x, y) == self.layout.get("key") and not self.has_key:
            return "key"
        if (x, y) == self.layout.get("door") and not self._door_open:
            return "door"
        return "empty"

    def _blocks_movement(self, x, y):
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            return True
        t = self._cell_type_at(x, y)
        if t == "obstacle":
            return True
        if t == "door":
            return True  # porte fermee bloque, tant que pas de cle
        return False

    def _one_hot_cell(self, x, y):
        vec = [0.0] * N_CHANNELS
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            t = self._cell_type_at(x, y)
        else:
            t = "obstacle"  # hors grille = traite comme un mur (empeche de "voir a travers" le bord)
        vec[self.channel_map[t]] = 1.0
        return vec

    def _encode_state(self):
        # fenetre 3x3 complete autour de l'agent (ordre : ligne par ligne, dy=-1..1, dx=-1..1)
        cells = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                cells.append(self._one_hot_cell(self.x + dx, self.y + dy))
        local = [v for cell in cells for v in cell]

        gx, gy = self.layout["goal"]
        direction = [(gx - self.x) / (GRID_SIZE - 1), (gy - self.y) / (GRID_SIZE - 1)]
        flag = [1.0 if self.has_key else 0.0]
        return local + direction + flag

    def step(self, action):
        nx, ny = self.x, self.y
        if action == 0:
            ny -= 1
        elif action == 1:
            ny += 1
        elif action == 2:
            nx -= 1
        elif action == 3:
            nx += 1

        bumped = self._blocks_movement(nx, ny)
        if not bumped:
            self.x, self.y = nx, ny

        # ramasser la cle
        if self.layout.get("key") and (self.x, self.y) == self.layout["key"] and not self.has_key:
            self.has_key = True
            self._door_open = True

        self.steps += 1
        reached = (self.x, self.y) == self.layout["goal"]
        timed_out = self.steps >= self.max_steps

        reward = -0.02
        if bumped:
            reward -= 0.1
        if reached:
            reward += 10.0

        done = reached or timed_out
        return self._encode_state(), reward, done


# ============================================================
# Les 4 variantes
# ============================================================

def make_variant_a(seed=None):
    """Reference : obstacles fixes, encodage canonique."""
    layout = {
        "start": (0, 0),
        "goal": (4, 4),
        "obstacles": [(2, 0), (2, 1), (2, 2), (2, 3)],  # mur vertical avec un trou en (2,4)
        "key": None,
        "door": None,
    }
    return GridBenchEnv(layout, CANONICAL_MAP, seed=seed)


def make_variant_b(seed=None):
    """Memes positions qu'A, canaux obstacle/goal permutes (surface differente)."""
    layout = {
        "start": (0, 0),
        "goal": (4, 4),
        "obstacles": [(2, 0), (2, 1), (2, 2), (2, 3)],
        "key": None,
        "door": None,
    }
    return GridBenchEnv(layout, PERMUTED_MAP, seed=seed)


def make_variant_c(seed=None):
    """Encodage canonique, disposition differente (mur horizontal, trou ailleurs)."""
    layout = {
        "start": (0, 4),
        "goal": (4, 0),
        "obstacles": [(0, 2), (1, 2), (2, 2), (3, 2)],  # mur horizontal avec trou en (4,2)
        "key": None,
        "door": None,
    }
    return GridBenchEnv(layout, CANONICAL_MAP, seed=seed)


def make_variant_d(seed=None):
    """Encodage canonique, disposition proche de A + regle composee (cle/porte).
    Le mur en x=2 est complet SAUF a la porte : impossible de traverser
    sans passer par la porte, donc la cle devient necessaire."""
    layout = {
        "start": (0, 0),
        "goal": (4, 4),
        "obstacles": [(2, 0), (2, 1), (2, 2), (2, 4)],
        "key": (0, 4),       # cote depart, atteignable sans traverser le mur
        "door": (2, 3),      # seule ouverture du mur -> bloque sans la cle
    }
    return GridBenchEnv(layout, CANONICAL_MAP, seed=seed)


VARIANTS = {"A": make_variant_a, "B": make_variant_b, "C": make_variant_c, "D": make_variant_d}
