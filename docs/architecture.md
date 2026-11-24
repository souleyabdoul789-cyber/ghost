# Architecture interne — ghost 0.6.0

## Vue des composants

```text
APPLICATION / interface utilisateur
              |
ghost.agent — boucle de planification JSON bornée
       |             |               |
ghost.browser   ghost.web        ghost.motivation
 Playwright      HTTPS/robots     objectifs approuvés
       |             |
ghost.vision    ghost.memory — SQLite + recherche lexicale
       |
ghost.data — corpus, provenance, splits, shards, images, SFT
       |
ghost.llm / ghost.seq — tokenizer -> embeddings -> Transformer
       |
ghost.nn / ghost.tensor — ctypes et optimisation
       |
C CORE — Tensor 2D float32 / autograd / AdamW
```

## Moteur natif C

- Matrices contiguës row-major, `float32`, autograd par graphe dynamique.
- Primitives différentiables : opérations élémentaires, matmul, transpose, softmax causal, LayerNorm, slice, concat colonnes/lignes, embedding lookup, cross-entropy à labels entiers.
- Optimiseurs SGD et AdamW natifs ; moments AdamW sérialisables depuis Python.
- Propriété mémoire explicite : les paramètres protégés survivent au nettoyage des graphes ; les tenseurs temporaires doivent être libérés explicitement.
- Noyau portable C ; les binaires sont compilés pour la machine cible et ne doivent pas être partagés entre architectures incompatibles.

## Données et tokenizer

`ghost.data.DatasetBuilder` convertit les sources documentaires en enregistrements avec source, licence, langue, SHA-256 et split. La déduplication exacte passe par SHA-256 ; la quasi-déduplication utilise SimHash 64 bits indexé en quatre bandes. Le split est calculé au niveau document avant entraînement du BPE.

Le BPE est byte-level : 256 octets constituent le vocabulaire de base. Les fusions sont apprises sur les pré-tokens pondérés ; les comptes de paires et l'index des mots concernés sont mis à jour incrémentalement à l'aide d'un tas. Les séquences tokenisées sont écrites dans des shards `uint32-le`, puis lues par `mmap`.

Les manifestes image-texte groupent toutes les légendes d'une même empreinte image dans le même split. Les exemples SFT sont séparés en prompt/completion et le modèle ignore les labels correspondant au prompt.

## Transformer

`TransformerLanguageModel` est un décodeur causal compact : embeddings de tokens et positions apprises, blocs pré-LayerNorm, projection QKV, attention multi-tête causale, projection attention, MLP SwiGLU, résiduels, LayerNorm final et poids de sortie liés à la table d'embedding. La loss prédit le prochain token.

L'entraînement principal utilise une séquence à la fois et une fenêtre de contexte bornée. Le script sauvegarde poids, configuration, moments AdamW, RNG, progression et hash tokenizer/base. L'implémentation actuelle ne possède pas encore KV cache, RoPE, dropout ou batching.

## Vision-langage

`OnnxVisionEncoder` prétraite les images RGB et produit soit un embedding global, soit une séquence de patches/feats. Le modèle ONNX est gelé ; `VisionProjector` apprend à aligner chaque représentation visuelle à `d_model`. `VisionLanguageModel` concatène la séquence visuelle aux embeddings textuels et aligne les logits sur les tokens de la légende. Le projecteur et le GPT reçoivent les gradients.

Il faut fournir un encodeur ONNX dont les poids, prétraitements et licence sont connus. L'archive ne contient pas ces poids. Les tests utilisent un encodeur synthétique, donc ne valident pas l'accuracy d'un modèle visuel réel.

## Acquisition Web et agent

`WebReader` fait de la lecture HTTPS sur allowlist, applique les contrôles DNS/redirection et robots.txt, limite la taille et extrait du texte. `BrowserTool` ajoute DOM, liens accessibles et capture d'écran dans une session Playwright séparée. Les pages sont des données non fiables.

`ToolRegistry` ne permet que les noms d'outils déclarés. `LLMJSONPlanner` demande une sortie `tool` ou `final`, vérifie le nom et le type des arguments, compacte l'historique et tient compte de la fenêtre du LLM. `AgentRuntime` impose budget de temps, nombre d'étapes et historique. L'agent a besoin d'un modèle SFT ; un GPT next-token brut n'est pas garanti capable de ce protocole.

`CodeSandbox` exécute seulement via bubblewrap sur Linux : namespaces, réseau désactivé, fichiers système en lecture seule et limites CPU/RAM/fichiers. Si le sandbox est indisponible, il refuse le lancement.

## Limites et considérations sécurité

- Le moteur C est 2D float32 ; pas de GPU ni kernels BLAS/SIMD spécifiques en 0.6.0.
- La validation DNS ne suffit pas à elle seule à éliminer tous les scénarios de rebinding au niveau socket ; pour des déploiements sensibles, la résolution validée doit être liée à l'IP réellement utilisée pour la connexion TLS.
- Playwright dépend de ses binaires navigateurs compatibles ; Android/Termux doit considérer un navigateur distant.
- L'import d'une donnée Web dans le corpus d'entraînement est explicite et doit être précédé de contrôles de provenance, licence, qualité, contamination et sécurité.
- Les états de curiosité sont des heuristiques programmées, pas une preuve de volonté ou d'émotion.
