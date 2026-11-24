# Banc de test de transfert — spécification

Protocole établi avant tout code, pour éviter de se convaincre soi-même
avec un simple score sur un gridworld (voir la discussion G-Brain,
§18). Sert à mesurer, AVANT de construire quoi que ce soit dans
G-Brain, ce qu'un DQN classique fait déjà (ou ne fait pas) en matière
de transfert et d'oubli — pour avoir une vraie référence à battre.

## Question testée

Un agent qui a appris une règle dans un monde A a-t-il besoin de
**moins d'expérience** pour réussir dans des mondes B, C, D où cette
même règle s'applique, mais où les détails changent — comparé à un
agent qui repart de zéro ?

## Les 4 mondes (`envs.py`)

Grille 5×5, 4 actions (haut/bas/gauche/droite), règle cachée stable
dans les 4 variantes : *certaines cases bloquent le passage ; il faut
les contourner pour atteindre l'objectif.*

| Variante | Ce qui change | Teste |
|---|---|---|
| A | Référence | — |
| B | Mêmes positions qu'A, mais les canaux "obstacle" et "objectif" sont permutés | Transfert de **surface** (l'agent a-t-il appris la relation, ou juste mémorisé des vecteurs bruts ?) |
| C | Encodage identique à A, disposition différente (mur horizontal, trou ailleurs) | Transfert **spatial** |
| D | Encodage/disposition proches d'A, + règle composée (clé/porte) | Généralisation **compositionnelle** |

Chaque monde est vérifié par calcul (BFS) : résoluble, et pour D, la
clé est mathématiquement nécessaire (sans elle, l'objectif est
inatteignable — pas de raccourci qui rendrait la règle inutile).

## Encodage de l'état (`STATE_SIZE = 48`)

Fenêtre 3×3 autour de l'agent (9 cases × 5 canaux one-hot) + vecteur
direction vers l'objectif (dx, dy) + indicateur "a la clé". Taille
FIXE quelle que soit la variante — condition nécessaire pour réutiliser
le même réseau d'une variante à l'autre (fine-tuning).

Deux choix antérieurs ont été essayés et rejetés, par la mesure, pas
par intuition :
- Grille entière en one-hot (128 dims) : bien trop lent à apprendre
  pour notre moteur (SGD simple, sans Adam/momentum).
- Vue à 4 voisins seulement (28 dims) : vérifié par calcul qu'elle
  laisse des cases ambiguës (même vue locale, action optimale
  différente selon la case) — la fenêtre 3×3 lève cette ambiguïté
  (0 conflit vérifié sur la variante A).

## Références (baselines) à battre

1. **DQN depuis zéro** sur chaque variante.
2. **DQN pré-entraîné sur A puis affiné** sur B/C/D (fine-tuning).
3. **DQN + replay prioritaire** (priorité = erreur TD) depuis zéro.

## Métriques

- **Épisodes jusqu'à 90 % de réussite** (moyenne glissante sur 20
  épisodes) — métrique principale.
- **Oubli catastrophique** : succès sur A après avoir appris D, sans
  ré-entraînement sur A.
- Toute statistique est répétée sur plusieurs graines aléatoires
  (`--seeds`) et rapportée en moyenne ± écart-type — une seule
  exécution ne prouve rien en RL (bruit important).

## Résultats de référence obtenus (3 graines, 300 épisodes)

| Variante | Scratch | Fine-tune (A→X) | Prioritized |
|---|---|---|---|
| A | 26 (±1) | — | 26 (±2) |
| B | 26 (±1) | 24 (±1) | 28 (±3) |
| C | 30 (±3) | 38 (±6) | 29 (±3) |
| D | 41 (±9) | 41 (±6) | 47 (±10) |

**Oubli catastrophique : succès sur A après avoir appris D = 0.00**
(constant sur les 3 graines — A est totalement oublié à chaque fois).

### Ce que ça dit

- Le fine-tuning depuis A n'aide **jamais** de façon fiable : neutre
  sur B (transfert de surface), **négatif** sur C (le fine-tuning
  dégrade la performance par rapport à repartir de zéro — sur-ajustement
  à la disposition précise d'A), neutre sur D.
- Le replay prioritaire n'apporte rien ici (ces tâches sont déjà assez
  denses en récompense pour que la priorisation n'aide pas) — il
  pourrait aider davantage sur un problème plus difficile.
- L'oubli catastrophique est total et systématique : un DQN classique
  n'a aucun mécanisme pour retenir A une fois qu'il apprend D.

C'est exactement le genre de résultat qui donne un sens à G-Brain :
une cible chiffrée et reproductible à battre, pas une impression.

## Reproduire / étendre

```bash
cd G/bench
python3 run_benchmark.py --seeds 5 --episodes 300   # config recommandee
python3 run_benchmark.py --seeds 10                 # plus fiable, plus long
```

## Bugs du moteur G trouvés et corrigés pendant la construction de ce banc

Deux bugs sérieux dans `G` lui-même ont été trouvés en essayant de
faire converger le DQN sur ces mondes — corrigés à la source (pas
seulement contournés ici), donc bénéfiques à tout usage futur de G :

1. **`seed=0` cassait l'initialisation des poids.** Le générateur
   xorshift a un point fixe en 0 : avec `seed=0`, l'état restait à 0
   indéfiniment, donc tous les poids étaient initialisés à la même
   valeur négative — tuant tous les neurones ReLU dès le départ
   (sortie 0 pour n'importe quelle entrée, quel que soit le nombre
   d'itérations d'entraînement). Corrigé dans `c/tensor.c`
   (`tensor_randn`) : `seed=0` est maintenant remplacé par une
   constante non nulle.
2. **Fuite mémoire massive.** `tensor_free_graph()` existait mais
   n'était appelée nulle part : chaque forward/backward laissait son
   graphe de calcul alloué pour toujours (~400 Mo pour 150 épisodes
   d'entraînement). Corrigé : les poids/biais des couches sont
   désormais marqués `is_protected` à la création (`nn.c`), et
   `tensor_free_graph()` libère tout le reste. `ghost.rl` et ce banc de
   test appellent maintenant systématiquement ce nettoyage après
   chaque forward/backward — vérifié stable sur 2000+ itérations
   (mémoire constante, plus de croissance).
