# ghost — campagne de premier pré-entraînement français

## Objectif du pilote

Cette campagne doit vérifier le chemin complet entre un corpus français public, le tokenizer byte-level BPE, les shards mmap, le Transformer causal, les checkpoints, la génération et un rapport TXT vérifiable.

Ce pilote n'est ni un entraînement de plusieurs mois, ni un LLM compétitif, ni un agent autonome. Les « rounds » sont un nombre défini de fenêtres de contexte aléatoires ; ils ne représentent pas des passages complets sur les 32 millions de documents du dataset.

## Source

- Dataset : `Brainquiver/general-web-fr-202608`
- Configuration : `fineweb2-hq-fra_Latn`
- Split source : `train`
- Licence annoncée sur la fiche : ODC-By 1.0
- Taille annoncée : 31 999 309 documents, 118 346 333 763 caractères, environ 41,3 GB de fichiers.
- Provenance : `epfml/FineWeb2-HQ`, configuration `fra_Latn`, avec origine Common Crawl.
- Règle : conserver l'attribution et examiner aussi les conditions Common Crawl ; ne pas supposer que chaque droit sur les documents web est automatiquement réglé par la licence du dataset.

Fiche : https://huggingface.co/datasets/Brainquiver/general-web-fr-202608

## Démarrage local

```bash
python -m pip install -U '.[data]'
./build.sh
python examples/fetch_hf_stream.py \
  --dataset Brainquiver/general-web-fr-202608 \
  --config fineweb2-hq-fra_Latn \
  --revision main \
  --output data/fineweb2_fr_pilot.jsonl \
  --max-documents 10000 \
  --max-total-chars 20000000 \
  --min-language-score 0.95 \
  --min-quality-score 0.30 --resume
python examples/train_ghost.py data/fineweb2_fr_pilot.jsonl \
  --output runs/ghost-fr-pilot-001 \
  --vocab-size 1024 --d-model 64 --heads 4 --layers 2 --context 128 \
  --epochs 3 --steps-per-epoch 500 --checkpoint-every-steps 50 --resume
python examples/write_training_report.py \
  --run-dir runs/ghost-fr-pilot-001 \
  --source-manifest data/fineweb2_fr_pilot.jsonl.source.json \
  --output runs/ghost-fr-pilot-001/report.txt
```

Pour une expérience durable, remplacer `main` par un SHA Hugging Face fixe et conserver le manifeste source.

## GitHub Actions

Le workflow `.github/workflows/ghost-first-pretrain.yml` se lance manuellement via `workflow_dispatch`. Il met en cache le dataset pilote et le répertoire du run, sauvegarde les checkpoints et publie le modèle, le tokenizer, les métriques, la génération et `report.txt` comme artefact.

Les runners GitHub-hosted ont une limite maximale de six heures par job. Le workflow est donc fait pour des pilotes reprenables, pas pour une session de plusieurs mois sans interruption. Une campagne de grande ampleur nécessitera un environnement d'entraînement persistant et un stockage fiable pour les checkpoints.

## Paramètres de départ

- Transformer décodeur causal : 2 couches, largeur 64, 4 têtes, contexte 128.
- BPE byte-level : vocabulaire cible 1 024.
- AdamW : learning rate 3e-4, weight decay 0,01, clipping 1,0.
- Corpus pilote : maximum 10 000 documents ou 20 millions de caractères.
- 3 rounds de 500 fenêtres aléatoires de 129 tokens.
- Checkpoint toutes les 50 mises à jour réussies.

Ce choix est volontairement modeste : le but est de valider pipeline, stabilité, pertes et vitesse avant d'augmenter les dimensions et le nombre de tokens.

## Conditions de réussite

- La révision distante est résolue en SHA immuable et consignée.
- La source et son attribution/licence sont dans le manifeste.
- Les documents sont normalisés, dédupliqués et répartis en train/validation/test au niveau document.
- Le tokenizer est appris uniquement sur le split train.
- Le test reste réservé et n'est jamais utilisé pour ajuster le modèle.
- Le checkpoint se reprend avec le même corpus, le même tokenizer et la même configuration.
- La loss de validation est suivie ; une baisse ne suffit pas à établir une compréhension générale.
- Le rapport TXT conserve loss, tokens, vitesse, durée, génération et empreintes SHA-256.

## Étape suivante : modèle capable d'utiliser des outils

Le pré-entraînement causal apprend à prédire le token suivant. Il n'apprend pas automatiquement à invoquer un terminal ou un navigateur. Il faudra ensuite préparer des exemples d'instructions et de traces d'outils, faire un SFT, puis mesurer les capacités dans un environnement isolé avec commandes autorisées, quotas, délais et approbation des actions sensibles.

La vision sera entraînée séparément avec des paires image-texte autorisées et un encodeur visuel ; le navigateur fournit les observations, mais ne remplace pas ce modèle visuel.

## Dépôt dédié ghost

Le projet est maintenant destiné au dépôt autonome `souleyabdoul789-cyber/ghost`, et non à l'ancien dépôt `G`. Créer ce dépôt vide sur GitHub, puis exécuter `bash scripts/publish_training_branch.sh`. La branche publiée est `pretraining/0.6.2`.
