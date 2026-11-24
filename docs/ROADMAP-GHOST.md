# ghost — feuille de route R&D

`ghost` est une plateforme expérimentale pour construire, entraîner et évaluer un petit modèle de langage puis un agent multimodal. « AGI » est une ambition de recherche, pas une capacité déclarée. Chaque palier doit franchir des tests reproductibles et un jeu d'évaluation séparé.

## État de la version 0.6.0

| Sous-système | État réel | Limite principale |
|---|---|---|
| Noyau C/autograd | Implémenté et testé sous ASan/UBSan Linux x86-64 | Tenseurs 2D float32, kernels CPU simples |
| Tokenizer BPE | BPE byte-level, tokens spéciaux, fusion incrémentale avec heap | Pré-tokens uniques conservés en RAM |
| GPT texte | Transformer causal entraînable, génération et état AdamW/RNG dans le script | Pas de RoPE/KV cache/batch/dropout ; aucun modèle pré-entraîné fourni |
| Ghost Data | Ingestion, provenance, hash, dédup, splits et shards mmap | Détection de langue heuristique ; JSON géant en mémoire |
| SFT outils | Dataset prompt/completion et loss masquée sur le prompt | Un corpus spécialisé et une validation plus large sont nécessaires |
| Vision | ONNX encoder gelé, tokens visuels/patches, projecteur entraînable, captioning et SFT multimodal image+prompt→completion | Aucun poids visuel livré ; encodeur ONNX réel non validé ici |
| Web | WebReader HTTPS + allowlist + robots + contrôles d'accès | DOM complexe nécessite l'option Playwright ; prudence DNS/SSRF |
| Navigateur | Playwright optionnel, texte/DOM/liens/captures | Installation Chromium externe ; cible Android non garantie |
| Mémoire | SQLite, documents, notes, recherche lexicale | Pas encore de retrieval vectoriel sémantique |
| Agent | Boucle outillée JSON, registre fermé, budgets, historique réduit, vision facultative via VLM | Dépend d'un SFT texte ou multimodal réussi ; outil limité au registre |
| Sandbox | Bwrap Linux fail-closed, limites ressources et réseau désactivé | Disponibilité dépend des namespaces système ; tests runtime complets requis |
| Motivation | Score de curiosité/priorité sur objectifs préapprouvés | Variables de contrôle logicielles, pas des émotions ressenties |

## Palier 1 — corpus texte de qualité (0.6.x)

- Construire un corpus légalement réutilisable, dédupliqué et documenté : source, licence, langue, hash et date.
- Couvrir français, anglais, haoussa, zarma et autres langues ciblées ; mesurer la couverture réelle plutôt que supposer la qualité.
- Inspecter le split document-level et les quasi-doublons ; faire une vérification humaine sur un échantillon.
- Rapporter tokens/octet, débit tokenizer, RAM maximale, taille des shards et distribution des longueurs.
- Conserver un jeu de validation propre et un jeu test inaccessible à l'entraînement.

## Palier 2 — LLM texte plus robuste (0.7.x)

- Ajouter RoPE ou une autre méthode positionnelle évaluée, KV cache et génération avec cache.
- Ajouter batching, accumulation de gradients, dropout si bénéfique, scheduler et clipping testés.
- Développer des kernels matmul/SIMD/BLAS selon la plateforme ; conserver un chemin C portable de référence.
- Mesurer perplexité, loss de validation, vitesse tokens/s, RAM, taille du checkpoint et stabilité de reprise.
- Tester la généralisation sur des textes jamais vus, les répétitions, les prompts longs et les langues à faible ressource.
- Fine-tuner par SFT sur des consignes vérifiées et exemples outils ; valider la syntaxe, les arguments, les refus et l'exactitude des appels.

## Palier 3 — modèle vision-langage (0.8.x)

1. Choisir un encodeur vision ONNX documenté, avec poids dont la licence autorise l'usage.
2. Vérifier preprocessing, résolution, normalisation, format de sortie et feature_dim avec des images de test.
3. Préparer des paires image-description, image-question et image-prompt→action avec provenance/licence ; les splits sont groupés par hash d'image.
4. Entraîner le projecteur sur les patches, mesurer la loss sur images non vues, puis affiner en SFT multimodal les sorties d'outils JSON.
5. Étudier l'entraînement de l'encodeur ou l'adaptation LoRA seulement après démonstration du projecteur.
6. Tester captures Web et UI, OCR si nécessaire, localisation d'éléments et compréhension de graphiques.

Un encodeur gelé + projecteur apprend un pont image-texte ; cela ne prouve pas encore une compréhension visuelle générale.

## Palier 4 — navigateur et apprentissage Web contrôlé (0.9.x)

- BrowserTool : DOM, accessibilité, screenshots, navigation par liens autorisés, limites de débit.
- Séparer les observations visuelles, DOM et texte ; ajouter provenance et dates.
- Les contenus Web sont non fiables et ne peuvent pas remplacer les règles système.
- Respecter robots.txt, conditions d'accès, licences et droits de réutilisation ; aucun contournement de login, paywall ou CAPTCHA.
- Évaluer la résistance au prompt injection, au DNS rebinding, aux redirections et aux ressources de sous-domaines.
- Stocker les observations dans la mémoire ; toute transformation en données d'entraînement exige un filtrage et une validation séparés.

## Palier 5 — agent expérimental (1.0.x)

Cycle : observer → planifier → appeler un outil du registre → vérifier → journaliser → mémoriser → rapporter.

- SFT et tests structurés des appels d'outils, y compris les arguments invalides et le refus d'outils non autorisés.
- Mémoire épisodique, sémantique et procédurale avec source, confiance, date et expiration.
- Sandbox isolé, pas d'accès réseau ni secret par défaut, limites CPU/RAM/temps/fichiers.
- Proposer une modification ou un apprentissage ; ne jamais écraser automatiquement le modèle en service.
- Boucle d'évaluation avant/après chaque mise à jour ; retour arrière si la qualité régresse.

## Motivation simulée

`ghost.motivation` attribue un score à des objectifs déjà approuvés selon priorité, utilité, incertitude, nouveauté et coût. La politique ne crée pas de permissions ni d'objectifs non autorisés. Les termes curiosité, ennui ou envie désignent ici des états de contrôle calculés, pas des sentiments subjectifs démontrés.

## Porte de sortie de chaque version

Aucun palier n'est « terminé » sans suite de tests, rapport de benchmark, comparaison avant/après, corpus et licences documentés, limites publiées, exemple reproductible et archive source propre. Une compilation Linux ne prouve pas la compatibilité Android, Windows ou macOS ; chaque cible doit être construite et testée séparément.
