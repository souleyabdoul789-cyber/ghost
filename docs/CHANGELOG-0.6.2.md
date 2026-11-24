# ghost 0.6.2 — premier entraînement reproductible

- Le récupérateur Hugging Face résout une branche/tag en SHA immuable avant le streaming.
- Le JSONL conserve source, URL, scores, licence, split, SHA de dataset et index de flux.
- Le manifeste latéral conserve la révision, les filtres, la provenance et le SHA-256 du corpus écrit.
- La reprise du streaming vérifie les métadonnées et refuse silencieusement un corpus d'une autre révision.
- Le trainer sauvegarde maintenant modèle, optimiseur, RNG, position et accumulateurs à intervalles configurables au milieu d'un round.
- La reprise vérifie le fingerprint du corpus, le tokenizer et la configuration avant de continuer.
- L'échantillon de génération est sauvegardé dans `generation_sample.txt`.
- Le générateur `write_training_report.py` produit un rapport TXT avec configuration, mesures, provenance et empreintes.
- Le workflow GitHub Actions restaure/sauvegarde le corpus et les checkpoints entre exécutions et publie les preuves sous forme d'artefact.
- Cette version reste un pilote CPU expérimental ; elle ne constitue pas un modèle agent final.

- Publication prévue sur un dépôt GitHub autonome nommé `ghost`, séparé de l’ancien dépôt `G`. Le script de publication vérifie le nom du remote.
