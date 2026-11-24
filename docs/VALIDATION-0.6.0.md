# Rapport de validation — ghost 0.6.0

Date : 24 novembre 2026

## Tests exécutés

| Contrôle | Résultat |
|---|---|
| Compilation du noyau C Linux x86-64 | Réussie |
| Tests C avec AddressSanitizer + UBSan | 11 groupes réussis |
| Tests Python d'intégration | 31 réussis |
| `py_compile` sur les modules, exemples et tests | Réussi |
| Entraînement texte bout en bout sur le mini-corpus inclus | Pipeline exécuté |
| SFT texte bout en bout sur le mini-corpus outils | Pipeline exécuté et checkpoint écrit |
| Dialogue `examples/chat_ghost.py` avec checkpoint | Chargement et génération exécutés |
| Construction du wheel natif | Réussie |
| Installation isolée du wheel et import `ghost` | Réussie |
| Reconstruction d'un wheel depuis le source tarball | Réussie |

## Mesures du mini-corpus de démonstration

- Corpus texte : 30 documents, aucune déduplication dans ce jeu.
- Configuration smoke-test : vocabulaire 301 tokens, `d_model=16`, 1 bloc, contexte 48.
- Entraînement causal : 1 époque, 2 mises à jour ; train loss ≈ 5,7775 et validation loss ≈ 5,7699.
- SFT d'outils : 8 exemples, 8 mises à jour, train loss ≈ 5,7029 ; le split validation est vide à cette taille.
- Le texte généré par le modèle de démonstration reste incohérent. Ces données et ces quelques mises à jour valident le pipeline, pas la qualité linguistique du modèle.

## Limites de validation de cette machine

- Environnement contrôlé : Linux x86-64, Python 3.13.
- Aucun modèle ONNX de vision ni poids visuels n'était disponible. Les tests VLM utilisent un encodeur factice : formes, gradients, génération, checkpoint et masquage SFT ont été validés, mais pas la compréhension réelle d'images.
- Le package Python Playwright est présent, mais Chromium n'est pas installé dans cet environnement ; la navigation graphique réelle n'a donc pas été exécutée.
- `bubblewrap` n'est pas présent ; le test confirme que le sandbox refuse de lancer du code lorsqu'il ne peut pas garantir l'isolation. L'exécution isolée réelle n'a pas été validée ici.
- Android/Termux, Windows, macOS et autres architectures n'ont pas été compilés dans cet environnement. Compiler et exécuter les tests sur chaque cible avant distribution.

## Interprétation

Cette version fournit une base expérimentale intégrée : moteur C, tokenizer BPE, Transformer causal, préparation des données, SFT texte et image, projecteur vision-langage, lecture Web, navigateur optionnel, mémoire et runtime d'agent.

Elle ne contient pas de poids de LLM ou d'encodeur visuel pré-entraîné et ne constitue pas une AGI. Les prochains jalons déterminants sont l'entraînement sur des corpus autorisés substantiels, l'évaluation de généralisation, les poids/mesures réels de vision, puis la validation du navigateur et du sandbox sur les plateformes visées.
