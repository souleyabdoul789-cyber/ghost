# ghost 0.6.1 — premier pipeline de pré-entraînement reproductible

- Ajout d'un extracteur Hugging Face en streaming, borné par nombre de documents et budget de caractères.
- Conservation de l'URL/source, de la configuration du dataset et de la licence ODC-By 1.0 dans chaque exemple.
- Reprise déterministe de l'extraction à partir du même flux mélangé.
- Ajout d'une configuration versionnée pour le premier pilote français.
- Ajout d'un workflow GitHub Actions manuel pour lancer le pilote et archiver les poids, le tokenizer, les métriques et le manifeste.
- Le trainer écrit désormais `metrics.jsonl` avec loss initiale de validation, loss train/validation, tokens traités, débit et durée ; les checkpoints conservent `tokens_seen`.
- Clarification : une « époque » du trainer correspond à un nombre fixe de fenêtres aléatoires, pas à un parcours exhaustif du corpus.
- Aucun changement de l'architecture du Transformer dans cette version.
