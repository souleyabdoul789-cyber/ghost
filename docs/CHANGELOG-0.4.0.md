# Ghost 0.5.0 — changements et validation

## Moteur C

- Allocation des tenseurs vérifiée (dimensions, dépassements, échecs d'allocation).
- Tri topologique du graphe protégé contre les échecs d'allocation.
- Initialisation gaussienne He/Box-Muller déterministe, avec fan-in corrigé et seed nul traité.
- Produit matriciel réordonné `i-k-j` pour améliorer la localité mémoire.
- Lookup natif d'embeddings et rétropropagation qui additionne les occurrences répétées.
- Cross-entropy fusionnée utilisant les IDs des classes cibles, sans matrice one-hot.
- Transposition, softmax causal, LayerNorm, slicing et concaténation avec gradients.
- Correction du gradient de transposition révélée par AddressSanitizer.
- AdamW natif avec moments, biais corrigés, weight decay découplé sur les matrices et clipping global.
- Vérification des échecs d'allocation dans les chemins Dense et MSE.

## Modèles et outils Python

- Binding `ctypes` aligné sur la nouvelle structure `Tensor` et les nouvelles fonctions natives.
- Embedding récurrent sans one-hot ; GRU/TinyLanguageModel conservés.
- Nouveau Transformer décodeur GPT compact : attention multi-tête causale, pré-LayerNorm, SwiGLU, embeddings de position appris et partage des poids d'embedding/sortie.
- Entraînement next-token par cross-entropy d'IDs, AdamW et clipping ; génération avec nettoyage des graphes temporaires ; checkpoints JSON.
- GMemory : SQLite, passages, recherche lexicale BM25 simplifiée, notes et export de corpus.
- GWeb : HTTPS, liste explicite de domaines, validation DNS IP publique, robots.txt, limites de téléchargement, extraction HTML et suivi de redirections autorisées.
- GAgent : exploration bornée, registre fermé d'outils, journal d'audit et récupération de preuves.
- Compilation C native pendant `pip install .` et avec `build.sh`.

## Résultats de validation

- 10 groupes de tests C compilés avec `-Wall -Wextra -Werror -pedantic`, AddressSanitizer et UBSan : succès.
- 8 tests Python d'intégration : succès.
- Entraînement du motif `[1,2,3,4,1]` : loss d'environ `4.56` à `0.0012` en 50 mises à jour ; le token suivant attendu est généré.
- DQN GridWorld : objectif atteint après entraînement et après sauvegarde/rechargement.
- Wheel 0.4.0 construit, installé dans un répertoire isolé et import natif vérifié sur Linux x86-64.
- Source tarball construit puis recompilé en wheel : succès.
- Exemple CLI d'entraînement exécuté sur un petit corpus de contrôle.

## Limites connues

Le Transformer est un prototype 2D float32, sans KV cache, RoPE, dropout, batch ou quantification. Il utilise des embeddings de position appris et recalcule tout le contexte à chaque token. La recherche mémoire est lexicale. L'agent Web est procédural et borné ; il ne possède pas encore de planification avancée pilotée par LLM et ne met pas à jour les poids automatiquement. Aucune validation Android ARM32/ARM64 n'a été exécutée dans l'environnement de développement x86-64 : le binaire doit être recompilé sur l'appareil Android ou dans un environnement de cross-compilation adapté.
