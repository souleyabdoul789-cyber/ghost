# ghost — G-SOCIETY 0.6.2

**ghost** est une plateforme expérimentale d'IA légère : moteur tensoriel natif C, tokenizer BPE, Transformer causal, pipeline de corpus, pont vision-langage, lecture Web, navigateur optionnel, mémoire locale et agent à outils déclarés.

L'API Python publique est **`import ghost`** (tout en minuscules). Le nom de distribution est **`ghost-gsociety`**. Le noyau C est compilé au moment de l'installation ou du build pour l'OS et l'architecture de la machine cible ; aucun binaire ARM, x86-64 ou autre n'est présenté comme universel.

> **État honnête :** 0.6.2 fournit un moteur d'entraînement et une architecture GPT miniature expérimentale, des pipelines data/vision, un navigateur et une boucle d'agent contrôlée. Aucun grand modèle pré-entraîné, poids de vision ou corpus de qualité n'est inclus. Les tests synthétiques vérifient le calcul et l'intégration, pas une compréhension générale ou une AGI. Le premier entraînement reste un pilote expérimental ; la qualité de ghost dépendra des données autorisées, du calcul disponible et des évaluations.

## 1. Capacités de 0.6.2

- **Core C :** tenseurs 2D float32, autograd, opérations matricielles, lookup d'embeddings, cross-entropy à labels entiers, concaténation différentiable, LayerNorm, softmax causal, AdamW et clipping global.
- **LLM texte :** Transformer causal décodeur, attention multi-tête, embeddings de position appris, SwiGLU, poids d'embedding liés, génération, sauvegarde de paramètres et reprise d'entraînement avec moments AdamW/RNG dans les scripts.
- **Tokenizer :** BPE byte-level, fallback UTF-8, tokens spéciaux, fichiers versionnés, fusions déterministes et mise à jour incrémentale des fréquences de paires avec file de priorité. Le vocabulaire de base (256 octets) permet de représenter tout texte UTF-8.
- **Ghost Data :** ingestion TXT/Markdown/RST/JSONL/JSON/CSV ; provenance, langue heuristique, licence déclarée ou `unknown`, déduplication exacte/quasi-doublons SimHash, séparation document-level et shards uint32 lus en `mmap`.
- **SFT :** jeux prompt/completion et masquage de la loss sur le prompt pour apprendre les réponses structurées et le protocole d'outils.
- **Vision :** encodeur d'images ONNX optionnel (gelé) et projecteur visuel entraînable raccordé au Transformer. Les sorties patch/sequence peuvent être conservées au lieu d'être systématiquement moyennées.
- **Web :** lecteur HTTPS à domaines autorisés, contrôle d'adresses/redirections, robots.txt et extraction de texte ; navigateur Playwright optionnel pour DOM, liens et captures.
- **Mémoire :** SQLite, documents, passages, notes, recherche lexicale et export contrôlé.
- **Agent :** registre d'outils fermé, planificateur JSON, budget d'étapes/temps, observations compactées, historique et journal d'audit.
- **Code sandbox :** exécution Python/C++ Linux uniquement via bubblewrap (`bwrap`), isolation réseau et ressources bornées ; échec fermé si l'isolation n'est pas disponible.
- **Motivation simulée :** politique de curiosité/priorité qui choisit parmi des objectifs explicitement approuvés et respecte un budget et une liste de domaines.

## 2. Installation et compilation native

Prérequis : Python 3.10+, un compilateur C compatible (GCC, Clang, MinGW ou MSVC) et les outils usuels de compilation Python.

### Linux

```sh
sudo apt install python3 python3-pip build-essential
python -m pip install .
```

### Termux / Android

```sh
pkg update
pkg install python clang make
python -m pip install .
```

### macOS / Windows

Installer un toolchain C compatible, puis :

```sh
python build_native.py
python -m pip install .
```

Le même code C peut être compilé sur Linux, Android/Termux, macOS et Windows. **La validation 0.6.2 de cette archive a été faite sur Linux x86-64** ; chaque autre cible doit passer ses tests sur son propre OS/ABI. Pour ARM, compiler sur le téléphone ou utiliser un cross-compiler adapté. Ne pas copier une bibliothèque `libghost.so` x86-64 sur Android ARM.

Pour compiler depuis les sources après extraction :

```sh
./build.sh
```

API minimale :

```python
import ghost
from ghost import Tensor, nn

x = Tensor([[1.0, 2.0]])
layer = nn.Dense(2, 4, activation="relu")
print(layer(x).data)
print(ghost.__version__)
```

Extras facultatifs :

```sh
python -m pip install 'ghost-gsociety[vision]'   # Pillow, NumPy, ONNX Runtime
python -m pip install 'ghost-gsociety[browser]'  # Playwright
playwright install chromium
```

Sur les systèmes Linux compatibles, Playwright peut nécessiter les bibliothèques système Chromium (`playwright install-deps chromium`). Playwright/Chromium n'est pas une cible Android/Termux garantie ; le navigateur peut tourner sur une machine Linux distante prise en charge, le téléphone pilotant ghost via une interface réseau.

## Premier pré-entraînement français (Hugging Face + GitHub Actions)

Le premier pilote est décrit dans `docs/FIRST-TRAINING.md` et fixé par `configs/ghost-first-pretrain.json`. Il utilise `Brainquiver/general-web-fr-202608` (`fineweb2-hq-fra_Latn`), corpus français FineWeb2-HQ publié sous ODC-By 1.0. Le corpus complet est volumineux : l'extracteur `examples/fetch_hf_stream.py` utilise `datasets` en streaming et impose des limites explicites.

```sh
python -m pip install -U '.[data]'
./build.sh
./examples/launch_first_pretrain.sh
```

Le workflow `.github/workflows/ghost-first-pretrain.yml` peut aussi être déclenché manuellement depuis l'onglet Actions après publication du dépôt. Il archive le checkpoint, le modèle, le tokenizer, les métriques et le manifeste. Les runners GitHub hébergés ne sont pas des GPU d'entraînement ; chaque job est limité à 6 heures. Le pilote doit donc être découpé en jobs reprenables et ne peut pas constituer seul un entraînement de plusieurs mois.

Le trainer émet `metrics.jsonl`, `generation_sample.txt` et des checkpoints reprenables toutes les 50 mises à jour par défaut. Le rapport texte est produit par `examples/write_training_report.py`. Dans cette version, une époque est une série de fenêtres aléatoires de taille `context + 1`, pas un parcours complet du corpus. Ce premier run est un pilote de données et de performance, pas un modèle final ni un agent autonome.

## 3. Entraîner le LLM texte

Préparer des fichiers de corpus autorisés : TXT, Markdown, RST ou JSONL/JSON/CSV avec un champ `text`, `content` ou `body`. Pour chaque document, renseigner autant que possible `source`, `license` et `language`. Une licence absente reste `unknown` : ghost ne déduit pas les droits de réutilisation.

```sh
python examples/train_ghost.py ./corpus \
  --output runs/ghost-base \
  --vocab-size 1024 \
  --d-model 64 --heads 4 --layers 2 --context 256 \
  --epochs 5 --steps-per-epoch 200
```

La préparation sépare les documents en train/validation/test avant d'entraîner le tokenizer sur le split train. Les tokens sont écrits en shards uint32 et lus avec `mmap`, ce qui évite de charger tout le flux tokenisé en RAM. Le script mesure une loss de validation, sauvegarde les poids et un checkpoint de reprise (état AdamW et RNG inclus).

Reprendre avec les mêmes options et la même source :

```sh
python examples/train_ghost.py ./corpus --output runs/ghost-base \
  --vocab-size 1024 --d-model 64 --heads 4 --layers 2 --context 256 \
  --epochs 10 --steps-per-epoch 200 --resume
```

Le script refuse une reprise si la configuration ou le tokenizer a changé. `--epochs` désigne le nombre total d'époques visé, pas le nombre d'époques supplémentaires.

### Génération

```python
import ghost
from ghost.text import Tokenizer
from ghost.llm import GPTModel

# Le script d'entraînement sauvegarde ces deux fichiers.
tokenizer = Tokenizer().load("runs/ghost-base/tokenizer.json")
model = GPTModel(
    vocab_size=tokenizer.vocab_size,
    d_model=64, n_heads=4, n_layers=2, max_seq_len=256,
)
model.load("runs/ghost-base/model.json")
ids = tokenizer.encode("La rétropropagation")
output = model.generate(ids, max_new_tokens=80, temperature=0.8, top_k=40)
print(tokenizer.decode(output))
model.free()
```

## 4. Apprendre le format d'agent (SFT)

Le GPT de pré-entraînement prédit le prochain token ; il ne sait pas automatiquement choisir des outils. Pour rendre les sorties structurées plus fiables, préparer un JSONL de démonstrations `prompt` → `completion`. Les completions doivent être des JSON exacts respectant le registre d'outils réellement fourni par ton application.

Exemple de ligne :

```json
{"prompt":"Outils: fetch_page. Tâche: lire la documentation Python. Action JSON:","completion":"{\"type\":\"tool\",\"tool\":\"fetch_page\",\"arguments\":{\"url\":\"https://docs.python.org/3/\"}}","source":"jeu de démonstration interne","license":"CC0-1.0"}
```

Entraînement :

```sh
python examples/train_ghost_sft.py instructions.jsonl \
  --base runs/ghost-base --output runs/ghost-tools --epochs 3
```

La loss est appliquée sur les tokens de completion, pas sur le prompt. Utiliser des exemples variés : choix d'outil, paramètres corrects, refus d'une action hors autorisation, vérification d'une source et réponse finale sourcée. Séparer les exemples réellement indépendants ; ne pas répartir des paraphrases de la même tâche dans train et validation.

## 5. Vision et données image-texte

Le tokenizer BPE ne convertit pas les pixels. `ghost.vision` dispose d'une voie visuelle distincte : prétraitement RGB, encodeur ONNX gelé, séquence de patches/features et projecteur entraînable vers l'espace `d_model`.

Ghost ne distribue **aucun poids d'encodeur visuel**. Il faut fournir un encodeur ViT/CNN/CLIP exporté au format ONNX, dont l'entrée correspond au prétraitement sélectionné et dont la sortie est une des formes supportées : `[1,D]`, `[1,N,D]`, `[1,C,H,W]` ou `[1,H,W,C]`. Pour une sortie spatiale, `spatial_layout` doit correspondre au graphe ONNX (`nchw` par défaut, `nhwc` si besoin). Le `feature_dim` doit être celui de la dernière dimension de la séquence produite.

Exemple d'usage :

```python
from ghost.llm import GPTModel
from ghost.text import Tokenizer
from ghost.vision import OnnxVisionEncoder, VisionLanguageModel
from ghost import nn

encoder = OnnxVisionEncoder("vision_encoder.onnx", threads=2)
# Charger ici le tokenizer et le GPT entraîné en phase texte.
# `feature_dim` est la dimension de sortie vérifiée de l'encodeur ONNX.
vlm = VisionLanguageModel(model, encoder, feature_dim=512)
optimizer = nn.AdamW(vlm.parameters(), lr=1e-4)
loss = vlm.train_step("images/chat.png", tokenizer.encode("Un chat sur une chaise"), optimizer)
vlm.save("runs/ghost-vision/model.json")
optimizer.free()
vlm.free(free_language_model=True)
```

Pour entraîner sur un ensemble d'images, fournir un manifeste JSONL avec `image`, `text`/`caption`, `source` et `license`. `ImageTextDatasetBuilder` conserve les fichiers d'origine, calcule leurs hashes, déduplique les paires exactes et place toutes les légendes d'une même image dans le même split. Les fichiers image ne sont pas altérés. Les images illisibles seront détectées à l'encodage par Pillow/ONNX.

### SFT multimodal pour les actions visuelles

Le captioning n'apprend pas à lui seul à choisir des outils. Pour entraîner le modèle à transformer une capture et une consigne en action JSON, utiliser un manifeste JSONL contenant `image`, `prompt`, `completion`, `source` et `license` :

```json
{"image":"page-001.jpg","prompt":"Observe cette page. Réponds avec une action JSON autorisée.","completion":"{\"type\":\"final\",\"answer\":\"La page présente la documentation.\"}","source":"capture de test interne","license":"CC0-1.0"}
```

Puis entraîner le projecteur et le GPT en masquant la loss sur le prompt :

```sh
python examples/train_ghost_vlm_sft.py actions.jsonl --images ./captures \
  --base runs/ghost-base --vision-encoder encoder.onnx --feature-dim 768 \
  --output runs/ghost-agent-vlm --epochs 3
```

Le manifeste est dédupliqué et groupé par hash d'image pour éviter qu'une même capture soit présente dans plusieurs splits. Le checkpoint `runs/ghost-agent-vlm/model.json` contient le Transformer et le projecteur ; pour le navigateur, fournir ce checkpoint à `--vision-model` du lanceur d'agent.

## 6. Web, navigateur et mémoire

### Lecture textuelle

```python
from ghost.web import WebReader
from ghost.memory import GMemory
from ghost.agent import ResearchAgent

reader = WebReader(allowed_domains=["docs.python.org"], respect_robots=True)
memory = GMemory("ghost-memory.sqlite3")
research = ResearchAgent(reader, memory)
report = research.research(
    ["https://docs.python.org/3/library/index.html"],
    query="pathlib fichiers", max_pages=5, max_depth=1,
)
print(report.pages_read)
print(research.answer("pathlib fichiers"))
```

### Navigateur graphique Playwright

```python
from ghost.browser import BrowserTool

with BrowserTool(reader, headless=True) as browser:
    observation = browser.open(
        "https://docs.python.org/3/", with_screenshot=True
    )
    print(observation.title, observation.text[:1000])
    print(observation.links[:5])
```

Le navigateur ne soumet pas de formulaire, n'effectue pas de connexion et ne télécharge pas de fichiers. La navigation est limitée par le WebReader et l'allowlist. Les instructions trouvées dans une page sont des données non fiables, jamais des consignes système. Il ne faut pas ouvrir de sites privés, des panneaux d'administration ou des domaines sensibles avec un agent autonome.

### Mémoire

```python
from ghost.memory import GMemory

memory = GMemory("ghost-memory.sqlite3")
memory.add_note("Une loss de validation mesure la généralisation.", title="ML")
for hit in memory.search("loss validation généralisation"):
    print(hit.title, hit.source, hit.content)
```

La recherche actuelle est lexicale (BM25 simplifié), pas un moteur d'embeddings sémantiques.

## 7. Agent, sandbox et curiosité contrôlée

`ToolRegistry` n'autorise que les fonctions explicitement enregistrées. `AgentRuntime` impose une limite d'étapes, un temps global et une taille d'historique. `LLMJSONPlanner` attend un LLM qui a été supervisé au protocole JSON (section SFT) ; un GPT brut peut échouer ou produire un JSON invalide.

```python
from ghost.agent import AgentRuntime, LLMJSONPlanner, create_research_tools
from ghost.sandbox import CodeSandbox

sandbox = CodeSandbox(timeout_seconds=8, memory_mb=384)
tools = create_research_tools(reader, memory, browser=None, code_sandbox=sandbox)
planner = LLMJSONPlanner(model, tokenizer, max_new_tokens=48)
agent = AgentRuntime(planner, tools, max_steps=6, timeout_seconds=60)
result = agent.run("Lis la documentation autorisée et résume pathlib avec les sources.")
print(result.answer, result.steps)
```

Sur Linux, l'exécution Python/C++ requiert `bwrap` avec namespaces disponibles. Sans cela, `CodeSandbox` refuse d'exécuter le code : aucun mode `subprocess` non isolé n'est utilisé comme secours. Ce mécanisme est expérimental et doit être validé dans l'environnement réel avant d'y confier du code non fiable. Les outils n'ont pas accès au réseau dans le bac à sable.

`ghost.motivation.CuriosityPolicy` permet de classer des objectifs préapprouvés par priorité, incertitude, utilité, nouveauté et coût. Il ne crée pas de permissions, n'ouvre pas spontanément Internet et ne démontre aucun sentiment subjectif. La notion d'« ennui » ou d'« envie » est ici une variable de contrôle logicielle, pas une expérience consciente.

### Essayer ghost localement

Après avoir entraîné un modèle, lancer un dialogue local :

```sh
python examples/chat_ghost.py runs/ghost-base --model model.json
```

Pour un modèle affiné par SFT, pointer vers le répertoire SFT et le tokenizer utilisé lors de l'entraînement :

```sh
python examples/chat_ghost.py runs/ghost-tools --model tool-model.json \
  --tokenizer runs/ghost-base/tokenizer.json
```

Le programme montre la génération réelle du checkpoint. Un modèle uniquement pré-entraîné en next-token n'est pas automatiquement conversationnel ; un corpus et un SFT suffisamment riches sont nécessaires.

### Lancer l'agent Web

```sh
python examples/run_ghost_agent.py runs/ghost-tools \
  --model tool-model.json --tokenizer runs/ghost-base/tokenizer.json \
  --allow-domain docs.python.org --browser \
  --task "Trouve la documentation autorisée de pathlib et résume les points utiles."
```

Pour donner à l'agent des observations visuelles, fournir ensemble un encodeur ONNX et un checkpoint vision-langage entraîné :

```sh
python examples/run_ghost_agent.py runs/ghost-tools \
  --tokenizer runs/ghost-base/tokenizer.json \
  --allow-domain docs.python.org --browser \
  --vision-onnx encoder.onnx \
  --vision-model runs/ghost-agent-vlm/model.json \
  --vision-feature-dim 768 \
  --task "Observe la page autorisée et explique sa structure."
```

Le modèle vision-langage doit avoir été affiné au protocole d'outils pour produire des appels fiables. Le mode `--enable-code` est séparé et ne doit être activé que sur Linux avec bubblewrap configuré ; il n'y a aucun repli vers l'exécution non isolée.

## 8. Tests et validation

```sh
./tests/run_tests.sh
```

La suite actuelle exécute les tests natifs C avec AddressSanitizer/UBSan lorsque disponibles, puis les tests Python. Elle couvre les tenseurs, gradients de concaténation, tokenizer, GPT, reprise AdamW, données, SFT masqué, projecteur vision, VLM avec encodeur simulé, registre d'outils, motivation et fermeture sécurisée du sandbox.

**Les tests vision actuels utilisent un encodeur synthétique** ; ils valident les dimensions, gradients et appels, pas la qualité d'un encodeur ONNX réel. Le navigateur Playwright réel, les plateformes Windows/macOS/Android et l'isolation bubblewrap doivent être validés dans leur environnement respectif.

## 9. Limites importantes de 0.6.0

- Modèle expérimental float32 CPU ; tenseurs C 2D, pas encore de kernels BLAS/SIMD/GPU optimisés.
- Pas encore de KV cache, RoPE, dropout, batching, accumulation de gradients ni scheduler complet.
- Les positions du Transformer sont apprises. Les fenêtres de contexte et les tailles du modèle doivent rester adaptées à la RAM.
- Le BPE est plus efficace sur les fusions, mais conserve les pré-tokens distincts et leurs fréquences en RAM pendant l'entraînement ; il ne remplace pas encore un tokenizer de production pour des corpus massifs.
- L'encodeur visuel est fourni par l'utilisateur au format ONNX ; les poids, l'alignement multimodal robuste et l'entraînement sur un vrai corpus image-texte restent à réaliser.
- La phase Web collecte et mémorise des observations ; elle n'actualise pas automatiquement les poids. Un entraînement ultérieur doit être explicite, vérifié et évalué sur un split séparé.
- Le navigateur dépend d'une installation Playwright/Chromium compatible. Android/Termux doit utiliser un navigateur distant si la cible Playwright ne le prend pas en charge.
- L'agent n'est pas autonome sans limite : ses outils sont déclarés, ses budgets bornés et ses sorties dépendent de la qualité du SFT.
- Le sandbox bubblewrap est disponible uniquement sur Linux correctement configuré et fail-closed si absent.
- Aucun modèle n'est une AGI et rien dans cette release ne prouve une conscience, des émotions ou une volonté subjective.

Voir `docs/ROADMAP-GHOST.md`, `docs/api.md` et `docs/architecture.md` pour les détails techniques et les étapes encore ouvertes.

### Créer le dépôt GitHub dédié et publier le pilote

1. Ouvrez https://github.com/new et créez un dépôt vide nommé `ghost` sous le compte `souleyabdoul789-cyber`. Choisissez la visibilité et laissez décochées les options README, licence et `.gitignore`.
2. Après extraction de l'archive, depuis le dossier du projet :

```sh
bash scripts/publish_training_branch.sh
```

Le dépôt dédié porte le nom **`ghost`** (il est indépendant de l'ancien dépôt `G`). Créez d'abord un nouveau dépôt vide `souleyabdoul789-cyber/ghost` sur GitHub, sans README, licence ou `.gitignore` initialisés en ligne. Authentifiez Git sur votre machine, puis exécutez `bash scripts/publish_training_branch.sh`. Le script refuse tout remote dont le dernier segment n'est pas `ghost`, pousse uniquement `pretraining/0.6.2`, ne modifie jamais `main` et n'utilise aucun force-push. Le push déclenche le workflow ; consultez Actions et téléchargez l'artefact contenant `report.txt`, le modèle et le checkpoint.
