# ghost 0.6.0 — LLM expérimental, données, vision et agent

## API et distribution
- Import public en minuscules : `import ghost`.
- Distribution : `ghost-gsociety`.
- Compilation du noyau C pendant le build pour l'OS hôte ; les wheels natifs sont spécifiques à la plateforme.

## Noyau et LLM
- Concaténation native de séquences avec rétropropagation.
- Transformer causal prenant des embeddings continus, pour brancher le projecteur visuel.
- SFT prompt/completion avec masquage de la loss sur le prompt.
- État AdamW sérialisable pour reprise exacte de l'optimiseur.
- Tokenizer BPE avec comptage incrémental des paires et validation différentielle contre un algorithme de référence.

## Données
- Ingestion textuelle en blocs, provenance, licences, langue heuristique, déduplication et splits par document.
- Shards de tokens uint32 lus via mmap.
- Manifestes image/caption et instruction/completion avec split par groupe d'image pour limiter les fuites entre splits.

## Vision
- Encodeur ONNX optionnel, prétraitement configurable et conservation des tokens spatiaux/patches.
- Projecteur visuel entraînable et modèle vision-langage raccordé au Transformer.
- Exemple d'entraînement et checkpoint incluant les identifiants des encodeurs de base.

## Web et agent
- Navigateur Playwright optionnel, pages et liens limités aux domaines autorisés.
- Planificateur de tâches au format JSON strict et runtime avec registre d'outils fermé, budgets et historique limité.
- Exécution de code uniquement dans un bac à sable Linux bubblewrap ; absence d'isolation = refus d'exécuter.
- Politique algorithmique de curiosité fondée sur des objectifs explicitement approuvés.

## Tests et limites
- 11 groupes de tests natifs C sous ASan/UBSan et 31 tests Python d'intégration passent sur Linux x86-64.
- Ajout de `examples/chat_ghost.py` pour tester un checkpoint local et de `examples/run_ghost_agent.py` pour piloter outils Web, navigateur, mémoire et option vision.
- Les tests vision utilisent un encodeur factice ; aucun poids ONNX ni corpus multimodal substantiel n'est inclus.
- Le jeu de démonstration est minuscule et sert uniquement à valider le pipeline, pas à produire un modèle de qualité.
- L'archive ne constitue pas une AGI, n'a pas d'émotions subjectives et n'inclut pas de modèle général pré-entraîné.
- Linux x86-64 est la cible effectivement validée ici ; Android/Termux, Windows, macOS et autres architectures nécessitent une validation native sur chaque cible.
