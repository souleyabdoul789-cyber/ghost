# ghost 0.7.0 — entraînement CPU segmenté et agent outillé

## Ce que le pipeline fait

1. Récupère un échantillon pouvant atteindre 200 millions de caractères du corpus français `Brainquiver/general-web-fr-202608`, configuration `fineweb2-hq-fra_Latn`, à la révision immuable `47d4541fa39938293ec294baccd91de876c32ac9`.
2. Prépare les documents, leurs sources/licences, les splits train/validation/test, le tokenizer BPE et les shards mmap.
3. Entraîne un Transformer causal plus grand que le pilote : vocabulaire 4096, dimension 128, 4 couches, 4 têtes, contexte 512.
4. Prépare et met en cache le corpus dans un job dédié, puis effectue 66 segments séquentiels, chacun plafonné à 18 000 secondes (5 heures) de calcul, avec reprise de l'état complet du modèle, d'AdamW et du RNG.
5. Réutilise un cache immuable pour le corpus et les shards, et remplace un seul artefact roulant pour éviter de conserver 66 gros checkpoints.
6. Après le pré-entraînement, exporte les poids du checkpoint, mélange 500 exemples d'instructions françaises humaines du corpus MIT `angeluriot/french_instruct` avec 173 exemples synthétiques conçus pour les signatures exactes des outils `ghost`, puis réalise cinq époques de SFT. Le workflow évalue le format et tente deux tests contrôlés : documentation Python autorisée et Python dans bubblewrap. Le workflow n’est déclaré réussi que si au moins 75 % des décisions produisent une action JSON reconnue, au moins 4 outils sur 8 sont correctement sélectionnés dans le test tenu à part, et l’agent appelle réellement `open_page` et `run_python`.

## Limites matérielles à comprendre

Les runners sont des CPU standard. Chaque job GitHub-hosted est limité à 6 heures et une exécution de workflow à 35 jours. Le runner `ubuntu-latest` standard public dispose de 4 CPU, 16 Go de RAM et 14 Go de SSD. Le job de préparation est séparé. Les 66 segments d'entraînement sont séquentiels et représentent au maximum 330 heures de calcul d’entraînement (13,75 jours), auxquelles s’ajoutent les installations, préparations et transferts ; la cible pratique est d’environ 14 jours. Le temps réel dépend de la file d'attente, du débit CPU, de la taille des shards, des limites de stockage et de la disponibilité GitHub. Le corpus French Instruct complet annoncé contient 276 000 conversations (environ 85 millions de tokens) ; le workflow en prélève 500 pour le SFT final afin de rester dans le budget d'un job CPU de six heures. Les fenêtres aléatoires sont tirées du même échantillon pendant tous les segments : les 330 heures ne correspondent pas à 330 heures de nouveaux documents téléchargés. Il n'y a pas de GPU dans ce pipeline.

Le sous-ensemble français FineWeb2-HQ est annoncé à environ 483 Go et 32,2 millions de documents. Ce workflow ne télécharge pas les 483 Go complets : le runner GitHub standard ne dispose que de 14 Go de SSD. La fiche du corpus avertit que des informations personnelles peuvent subsister et recommande un filtrage supplémentaire ; le seuil de qualité et le filtre de langue ne garantissent pas un corpus exempt de données personnelles. Il prélève donc jusqu’à 200 millions de caractères (100 000 documents), environ dix fois les 20 millions de caractères du pilote précédent. Il s'agit d'un échantillon agrandi, pas du corpus intégral.

Une baisse de loss ne suffit pas à prouver que le modèle sait agir. Le rapport d'évaluation de sélection d'outils et les deux tests d'agent doivent être consultés. Les deux commandes de test sont tolérantes aux erreurs pour que leurs rapports soient conservés ; une validation finale obligatoire fait ensuite échouer le workflow si l'agent ne passe pas les tests.

## Application depuis Termux

Dans le dépôt local `ghost`, à la racine du projet :

```bash
git fetch origin pretraining/0.6.2
git switch -c training/ghost-0.7.0 FETCH_HEAD
unzip -o ~/storage/downloads/ghost-training-kit-0.7.0.zip -d .
python -m py_compile examples/train_ghost_long.py examples/export_ghost_checkpoint.py examples/build_ghost_agent_sft.py examples/evaluate_ghost_agent.py examples/verify_ghost_agent_smoke.py examples/write_ghost_long_report.py examples/fetch_french_instruct_sft.py examples/merge_ghost_sft_manifests.py
git add .github/workflows/ghost-14day-training.yml .github/workflows/ghost-training-segment.yml examples/train_ghost_long.py examples/export_ghost_checkpoint.py examples/build_ghost_agent_sft.py examples/evaluate_ghost_agent.py examples/verify_ghost_agent_smoke.py examples/write_ghost_long_report.py examples/fetch_french_instruct_sft.py examples/merge_ghost_sft_manifests.py docs/GHOST-0.7.0-14DAY-TRAINING.md
git commit -m "Add long resumable ghost training and agent SFT pipeline"
git push -u origin training/ghost-0.7.0
```

Le `git push` déclenche le workflow. La préparation du corpus arrive d'abord, puis les 66 segments se succèdent et enfin le SFT français ciblé et les tests d'agent. Le dépôt doit rester public pour que l'utilisation des runners Linux standards soit gratuite et illimitée selon la documentation GitHub actuelle. N'ajoute aucun secret à ce workflow : aucun jeton Hugging Face privé n'est nécessaire pour le corpus public.

## Arrêt et reprise

L'exécution entière apparaît dans GitHub Actions sous `ghost — 14-day CPU pretraining + agent SFT`. Si un segment échoue, le workflow s'arrête au lieu de continuer sur un checkpoint potentiellement incohérent. Corriger la cause, puis relancer le job en échec depuis GitHub Actions. L'artefact roulant `ghost-state` contient le dernier checkpoint valide.
