# ghost 0.5.0 — migration de nom et tokenizer

## Identité et packaging

- Nouveau package Python public : `ghost` (`import ghost`).
- Nom de distribution : `ghost-gsociety`.
- Bibliothèque native renommée `libghost.so`, `libghost.dylib` ou `ghost.dll`.
- Compilation choisie selon l’OS hôte et le compilateur disponible.
- La compilation Linux x86-64 est testée ; les toolchains Windows/macOS et les cibles croisées demandent encore une validation sur machines dédiées.
- Le code source reste commun : chaque cible compile son propre binaire natif.

## Tokenizer BPE

- Entraînement depuis un itérable ou depuis un fichier UTF-8 lu ligne par ligne.
- Agrégation des morceaux identiques pour réduire la mémoire du corpus.
- Seuil `min_frequency` et choix déterministe des fusions à fréquence égale.
- Reconnaissance des tokens spéciaux à l’encodage et option pour les ignorer au décodage.
- Format de sauvegarde versionné, chargement avec validation des IDs et compatibilité avec les anciens fichiers BPE.
- Statistiques d’entraînement et encodage par lots.

## Validation effectuée

- 10 groupes de tests natifs C avec AddressSanitizer et UBSan : succès (état 0.5.0).
- 12 tests Python : succès (état 0.5.0).
- Wheel Linux x86-64 construit, installé dans un répertoire isolé et import `ghost` + appel natif `Tensor` vérifiés.
- Aucun binaire Linux n’est inclus dans l’archive source ; le moteur doit être compilé sur la machine cible.

## Non inclus

- Pas encore de pipeline complet de dataset ou d’entraînement à grande échelle.
- Pas d’encodeur d’image ni de modèle vision-langage.
- Pas de Playwright ni d’autonomie LLM contrôlant le navigateur.
- Pas de preuve de sentience ou d’émotions subjectives.
