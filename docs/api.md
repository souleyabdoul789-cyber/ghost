# Référence API — ghost 0.6.0

L'API publique est en minuscules : `import ghost`. La distribution s'appelle `ghost-gsociety`.

## `ghost.Tensor`

Tenseur matriciel 2D `float32`, stocké en row-major dans le moteur C.

| API | Fonction |
|---|---|
| `Tensor(values, requires_grad=False)` | Crée une matrice à partir de listes 1D/2D. |
| `Tensor.randn(rows, cols, requires_grad=False, seed=42)` | Initialisation gaussienne. |
| `.shape`, `.data`, `.grad` | Forme et données/gradients. |
| `+`, `-`, `*`, `.matmul(other)` | Opérations matricielles et élémentaires. |
| `.relu()`, `.sigmoid()`, `.tanh()` | Activations. |
| `.sum()`, `.scale(k)` | Réduction et multiplication par constante. |
| `.transpose()`, `.softmax(causal=False)` | Transposition et softmax. |
| `.layer_norm(gamma,beta,eps)` | Normalisation par ligne. |
| `.slice_cols(start,count)` | Sélection différentiable de colonnes. |
| `.concat_cols(other)`, `.concat_rows(other)` | Concaténations différentiables. |
| `.cross_entropy_ids(ids)` | Cross-entropy stable à cibles entières. |
| `.backward()`, `.zero_grad()` | Autograd. |
| `.free()`, `.free_graph()`, `.free_intermediates()` | Gestion mémoire explicite. |
| `embedding_lookup(weight, ids)` | Lookup natif des lignes d'embedding avec gradient pour IDs répétés. |

Les paramètres d'une couche sont protégés pendant `free_graph()`. Les feuilles d'entrée ne le sont pas automatiquement ; libérer chaque graphe et chaque feuille selon leur propriétaire.

## `ghost.nn`

- `Dense(in_features, out_features, activation="none", seed=42)` : couche affine.
- `Sequential(layers)` : séquence de couches.
- `mse_loss(pred,target)`, `cross_entropy_loss_ids(logits,targets)` : fonctions de perte.
- `SGD(parameters,lr=0.01)` : SGD natif.
- `AdamW(parameters,lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)` : AdamW natif.
- `AdamW.clip_grad_norm(max_norm=1.0)` : clipping global.
- `AdamW.state_dict()` / `load_state_dict(state)` : moments, hyperparamètres et compteur de pas.
- `zero_grad()`, `step()`, `free()` : gestion de l'optimiseur.

## `ghost.text.Tokenizer`

BPE byte-level déterministe et versionné.

- `train(texts,vocab_size=1000,min_frequency=2)` : apprend les fusions sur un itérable de texte.
- `train_file(path,...)` : lit le fichier ligne par ligne.
- `add_special_tokens([...])`, `token_to_id(token)` : tokens réservés.
- `encode(text)`, `decode(ids)`, `encode_batch(texts)` : conversion texte/IDs.
- `save(path)`, `load(path)` : sauvegarde et validation du vocabulaire.
- `vocab_size`, `training_stats` : statistiques.

Le fallback sur les octets permet de représenter tout texte UTF-8. Les statistiques de paires sont mises à jour incrémentalement pendant les fusions. Les pré-tokens distincts restent conservés en mémoire pendant l'apprentissage.

## `ghost.seq` et `ghost.llm`

`ghost.llm.GPTModel` est un alias de `TransformerLanguageModel`.

`TransformerLanguageModel(vocab_size,d_model=64,n_heads=4,n_layers=2,max_seq_len=128,ff_mult=4,seed=42)` expose :

- `parameters()` : paramètres entraînables ;
- `forward(token_ids)` : logits `T × vocab_size` ;
- `forward_embeddings(input_embeddings,position_offset=0)` : entrée d'embeddings continus (texte ou image) ;
- `embed_tokens(token_ids)` : embeddings sans position ;
- `train_step(token_ids,optimizer,max_grad_norm=1.0,target_start=1)` : entraînement next-token ; `target_start` masque les labels précédant le token cible, utile pour SFT ;
- `fit(...)` : entraînement sur fenêtres aléatoires ;
- `generate(prompt_ids,max_new_tokens=40,temperature=.8,top_k=40,...)` : génération ;
- `state_dict()`, `load_state_dict(state)`, `save(path)`, `load(path)`, `free()`.

Les positions sont apprises ; pas de RoPE, KV cache, dropout ni batching dans cette version.

## `ghost.data`

- `DatasetBuilder(seed=42,val_fraction=.05,test_fraction=.05,near_duplicate_threshold=3)` : ingestion texte et séparation de corpus.
- `iter_split(dataset_dir,split)` : documents d'un split.
- `write_token_shards(documents,tokenizer,output_dir,eos_id=None,tokens_per_shard=...)` : encoding uint32.
- `TokenShardDataset(directory)` : dataset `mmap`, `sample_window()` et `read_window()` ; contexte manager recommandé.
- `ImageTextDatasetBuilder(image_root,...)` : manifeste image-texte, hash et split groupé par image.
- `iter_image_split(dataset_dir,split)` : paires image-caption.
- `ImageInstructionDatasetBuilder(image_root,...)`, `iter_image_instruction_split(...)` : données multimodales image + prompt → completion/action JSON ; split groupé par image et chemins confinés à `image_root`.
- `InstructionDatasetBuilder(...)`, `iter_instruction_split(...)` : jeux SFT prompt/completion textuels.

Les licences ne sont pas vérifiées automatiquement. Une licence absente devient `unknown` et doit être clarifiée par le fournisseur des données.

## `ghost.vision`

- `ImagePreprocess(size=224,mean=...,std=...)` : paramètres RGB de prétraitement.
- `OnnxVisionEncoder(model_path,input_name=None,output_name=None,preprocess=None,threads=2,spatial_layout="nchw")` : encodeur ONNX gelé, méthode `encode_tokens(image)` pour séquences et `encode(image)` pour pooling global.
- `VisionProjector(feature_dim,d_model,seed=...)` : projection entraînable, `parameters`, `state_dict`, `load_state_dict`, `free`.
- `VisionLanguageModel(language_model,encoder,feature_dim=...,seed=...)` : `forward`, `forward_features`, `train_step`, `train_step_features`, `generate`, `state_dict`, `save`, `load`, `free`. `train_step_features(...,target_start=0)` masque les tokens du prompt ; pour image+consigne→action, `target_start` est l'index du premier token de completion.

`examples/train_ghost_vlm_sft.py` entraîne cette voie image+prompt→completion et sauvegarde `model.json`, compatible avec `LLMJSONPlanner(vision_model=...)`.

Extras requis : `pip install 'ghost-gsociety[vision]'`. Aucun encodeur/poids visuel n'est livré.

## `ghost.web` et `ghost.browser`

- `WebReader(allowed_domains,...)` : lecture HTTPS limitée aux domaines autorisés et extraction de contenu.
- `BrowserTool(web_reader,headless=True,...)` : `open(url,with_screenshot=False)`, `follow_link(label,...)`, `close()` et context manager. Dépend de `pip install 'ghost-gsociety[browser]'` et `playwright install chromium`.

Ni login, ni contournement paywall/CAPTCHA, ni soumission de formulaire ou téléchargement automatisé.

## `ghost.memory`

`GMemory(path="ghost-memory.sqlite3",chunk_chars=1200)` : `add_document`, `add_note`, `search`, `export_corpus`, `close`. La recherche est lexicale, pas vectorielle.

## `ghost.agent`

- `ToolRegistry.register(name,function,description)` : registre d'outils fermé ; `call(name,**kwargs)` journalise les appels.
- `ResearchAgent(web_reader,memory)` : `research`, `answer`, `remember_feedback`, `export_training_corpus`.
- `LLMJSONPlanner(model,tokenizer,vision_model=None,max_new_tokens=48,max_history_chars=3000)` : planificateur JSON pour modèle SFT.
- `AgentRuntime(planner,tools,max_steps=8,timeout_seconds=90,max_history_chars=12000)` : boucle avec budgets.
- `create_research_tools(web_reader,memory,browser=None,code_sandbox=None)` : registre de recherche.

Le modèle doit avoir été entraîné au format JSON outils. L'agent vérifie le nom demandé contre le registre mais ne garantit pas la vérité des sorties du modèle : vérifier les observations et leurs sources.

## `ghost.sandbox` et `ghost.motivation`

- `CodeSandbox(...)` : `run_python(source)` / `run_cpp(source)` ; exige Linux + bubblewrap et refuse si l'isolation n'est pas disponible.
- `GoalCandidate(...)` : objectif avec `approved`, priorité, incertitude, utilité, coût et domaine.
- `CuriosityPolicy(allowed_domains=(),action_budget=20,state_path=None)` : `score`, `choose`, `record`, `state_dict`, `load`, `save`.

La politique motivationnelle ne crée pas de permissions ni de tâches non approuvées ; les variables de curiosité ne sont pas des émotions subjectives.
