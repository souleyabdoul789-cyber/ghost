"""
ghost.seq
----------------------------------------------------------------
Briques pour traiter des séquences tokenisées par ghost.text : lookup
'd'embeddings natif, RNN/GRU et Transformer décodeur de type GPT.
Les opérations lourdes (lookup, autograd, LayerNorm, softmax causal,
produits matriciels) sont exécutées par le moteur C.

Le lookup extrait directement les lignes de la table d'embedding et
accumule les gradients des tokens répétés. Il ne construit aucune
matrice one-hot.
"""

from .tensor import Tensor, embedding_lookup
from . import nn


# ============================================================
# Embedding
# ============================================================

class Embedding:
    """Table d'embedding : transforme des ids de tokens en vecteurs continus
    appris. Les vecteurs sont appris comme n'importe quel poids (le gradient
    remonte depuis la suite du reseau jusqu'a cette table).

    Exemple :
        emb = Embedding(vocab_size=300, embed_dim=16)
        vectors = emb(token_ids)   # Tensor (len(token_ids) x embed_dim)
    """

    def __init__(self, vocab_size, embed_dim, seed=42):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.weight = Tensor.randn(vocab_size, embed_dim, requires_grad=True, seed=seed)
        self.weight.protect()  # survit a free_graph() -- comme les poids d'un Dense

    def _one_hot_batch(self, token_ids):
        rows = []
        for tid in token_ids:
            row = [0.0] * self.vocab_size
            row[tid] = 1.0
            rows.append(row)
        return Tensor(rows)

    def __call__(self, token_ids):
        """Lookup natif : renvoie (len(token_ids) x embed_dim), sans one-hot."""
        return embedding_lookup(self.weight, token_ids)

    def parameters(self):
        return [self.weight]

    def free(self):
        self.weight.free()


# ============================================================
# Cellule recurrente (RNN simple, type Elman)
# ============================================================

class RNNCell:
    """Une cellule recurrente classique :
        h_t = tanh(x_t . Wxh + h_{t-1} . Whh + b)

    Garde une "memoire" (l'etat cache h) qui se transmet d'un pas de
    temps au suivant -- ce qui manque a une simple Dense pour traiter
    des sequences (une Dense traite chaque entree independamment, sans
    se souvenir de ce qui precede).
    """

    def __init__(self, input_size, hidden_size, seed=42):
        self.hidden_size = hidden_size
        self.Wxh = nn.Dense(input_size, hidden_size, activation="none", seed=seed)
        self.Wxh.bias.set_data([0.0] * hidden_size)  # evite un double biais avec Whh
        self.Whh = nn.Dense(hidden_size, hidden_size, activation="none", seed=seed + 1)

    def init_hidden(self):
        return Tensor(rows=1, cols=self.hidden_size, requires_grad=False)

    def step(self, x_t, h_prev):
        """Un seul pas de temps. x_t : Tensor (1 x input_size). h_prev : Tensor (1 x hidden_size).
        Renvoie le nouvel etat cache h_t (1 x hidden_size)."""
        pre = self.Wxh(x_t) + self.Whh(h_prev)
        return pre.tanh()

    def forward_sequence(self, x_seq_rows):
        """x_seq_rows : liste de listes de floats, UNE PAR PAS DE TEMPS (deja les
        embeddings, sous forme de valeurs Python -- voir TinyLanguageModel pour
        pourquoi on repasse par des valeurs plutot que par le Tensor empile directement).
        Renvoie la liste des etats caches [h_1, ..., h_T] (chacun 1 x hidden_size),
        CHAQUE h_t reste connecte au graphe d'autograd en remontant jusqu'a x_seq_rows
        si ces valeurs proviennent elles-memes d'un Tensor relie a des poids appris
        (ce qui est le cas dans TinyLanguageModel : Wxh(x_t) reconstruit le lien)."""
        h = self.init_hidden()
        hiddens = []
        for row in x_seq_rows:
            x_t = Tensor([row])
            h = self.step(x_t, h)
            hiddens.append(h)
        return hiddens

    def parameters(self):
        return self.Wxh.parameters() + self.Whh.parameters()


# ============================================================
# GRU (Gated Recurrent Unit) -- memoire a plus long terme que RNNCell
# ============================================================

class GRUCell:
    """Cellule recurrente a PORTES (comme RNNCell, mais plus solide sur le
    long terme). Trois sous-calculs a chaque pas de temps :

      z_t (porte de mise a jour)  = sigmoid(x_t.Wxz + h_{t-1}.Whz)
      r_t (porte de reinitialisation) = sigmoid(x_t.Wxr + h_{t-1}.Whr)
      h~_t (candidat)             = tanh(x_t.Wxh + (r_t * h_{t-1}).Whh)
      h_t                         = (1 - z_t) * h_{t-1} + z_t * h~_t

    Intuition : z_t decide COMBIEN du nouveau contenu (h~_t) remplace
    l'ancien etat, et r_t decide combien de l'ancien etat compte encore
    pour calculer ce nouveau contenu. Avec RNNCell, l'etat cache est
    entierement recalcule a chaque pas -- un signal lointain s'estompe
    vite (gradient qui s'evanouit sur de longues sequences). Ici, si
    z_t reste proche de 0, h_t peut recopier h_{t-1} presque sans
    changement, ce qui permet de preserver une information sur
    beaucoup plus de pas de temps.

    Construite entierement avec des briques DEJA existantes de G
    (Dense, sigmoid, tanh, multiplication elementwise) -- aucun nouvel
    ajout au moteur C n'a ete necessaire pour cette cellule."""

    def __init__(self, input_size, hidden_size, seed=42):
        self.hidden_size = hidden_size
        self.Wxz = nn.Dense(input_size, hidden_size, activation="none", seed=seed)
        self.Whz = nn.Dense(hidden_size, hidden_size, activation="none", seed=seed + 1)
        self.Wxr = nn.Dense(input_size, hidden_size, activation="none", seed=seed + 2)
        self.Whr = nn.Dense(hidden_size, hidden_size, activation="none", seed=seed + 3)
        self.Wxh = nn.Dense(input_size, hidden_size, activation="none", seed=seed + 4)
        self.Wxh.bias.set_data([0.0] * hidden_size)  # evite un double biais avec Whh
        self.Whh = nn.Dense(hidden_size, hidden_size, activation="none", seed=seed + 5)
        self._ones = Tensor([[1.0] * hidden_size])  # constante, pas de gradient
        self._ones.protect()  # BUG EVITE : sans ca, free_graph() la liberait apres le
        # 1er pas (elle est "parent" de one_minus_z a CHAQUE pas de temps), causant un
        # use-after-free des le pas suivant -- meme categorie de bug que pour Embedding.

    def init_hidden(self):
        return Tensor(rows=1, cols=self.hidden_size, requires_grad=False)

    def step(self, x_t, h_prev):
        z = (self.Wxz(x_t) + self.Whz(h_prev)).sigmoid()
        r = (self.Wxr(x_t) + self.Whr(h_prev)).sigmoid()
        h_candidate = (self.Wxh(x_t) + self.Whh(r * h_prev)).tanh()
        one_minus_z = z.scale(-1.0) + self._ones
        return (one_minus_z * h_prev) + (z * h_candidate)

    def parameters(self):
        return (self.Wxz.parameters() + self.Whz.parameters()
                + self.Wxr.parameters() + self.Whr.parameters()
                + self.Wxh.parameters() + self.Whh.parameters())


# ============================================================
# Petit modele de langage de bout en bout (embedding + RNN + sortie)
# ============================================================

class TinyLanguageModel:
    """Embedding -> RNNCell -> Dense(vocab_size) : predit le prochain token
    a partir des tokens precedents. Entraine par RETROPROPAGATION A TRAVERS
    LE TEMPS (BPTT) : la loss totale (somme des pertes de chaque position de
    la sequence) est retropropagee EN UNE FOIS, ce qui fait remonter le
    gradient a travers tous les pas de temps jusqu'aux poids de l'embedding.

    La sortie produit des LOGITS BRUTS (pas d'activation), et l'entrainement
    utilise un vrai softmax + entropie croisee (ghost.nn.cross_entropy_loss,
    fusionnes et stables numeriquement -- voir c/tensor.c). Une premiere
    version utilisait sigmoid+MSE : ca marchait sur un motif jouet parfait
    (4 symboles equilibres) mais s'effondrait completement sur du vrai texte
    (le modele predisait toujours le token le plus frequent, peu importe le
    contexte) -- verifie, puis corrige avec un vrai softmax+entropie croisee.

    Exemple :
        lm = TinyLanguageModel(vocab_size=300, embed_dim=16, hidden_size=32)
        opt = nn.SGD(lm.parameters(), lr=0.1)
        for epoch in range(200):
            loss = lm.train_step(token_ids, opt)
        generated = lm.generate(prompt_ids, length=20)
    """

    def __init__(self, vocab_size, embed_dim=16, hidden_size=32, seed=42, cell_type="gru"):
        """cell_type : "gru" (par defaut -- meilleure memoire long terme) ou
        "rnn" (plus simple/leger, suffisant pour de tres courtes sequences)."""
        self.vocab_size = vocab_size
        self.embedding = Embedding(vocab_size, embed_dim, seed=seed)
        if cell_type == "gru":
            self.rnn = GRUCell(embed_dim, hidden_size, seed=seed + 1)
        elif cell_type == "rnn":
            self.rnn = RNNCell(embed_dim, hidden_size, seed=seed + 1)
        else:
            raise ValueError('cell_type doit etre "gru" ou "rnn"')
        self.output = nn.Dense(hidden_size, vocab_size, activation="none", seed=seed + 2)

    def parameters(self):
        return self.embedding.parameters() + self.rnn.parameters() + self.output.parameters()

    def _one_hot(self, token_id):
        row = [0.0] * self.vocab_size
        row[token_id] = 1.0
        return row

    def _forward_logits(self, token_ids):
        """token_ids (longueur T) -> liste de T Tensors de logits (1 x vocab_size),
        TOUS encore connectes au graphe d'autograd (embedding -> rnn -> output).
        Un embedding separe par pas de temps (plutot qu'un seul gros matmul pour
        toute la sequence) : c'est ce qui garde CHAQUE pas relie individuellement
        a self.embedding.weight, condition necessaire pour que le gradient mette
        vraiment a jour la table d'embedding via BPTT."""
        h = self.rnn.init_hidden()
        logits = []
        for tid in token_ids:
            x_t = self.embedding([tid])   # Tensor (1 x embed_dim), relie a embedding.weight
            h = self.rnn.step(x_t, h)
            logits.append(self.output(h))
        return logits

    def train_step(self, token_ids, opt):
        """Un pas d'entrainement : predire token_ids[t+1] a partir de token_ids[0..t].
        Renvoie la loss moyenne par position (float)."""
        if len(token_ids) < 2:
            return None
        inputs = token_ids[:-1]
        targets = token_ids[1:]

        logits_seq = self._forward_logits(inputs)

        opt.zero_grad()
        total_loss = None
        for logit_t, target_id in zip(logits_seq, targets):
            step_loss = nn.cross_entropy_loss_ids(logit_t, [target_id])
            total_loss = step_loss if total_loss is None else total_loss + step_loss

        total_loss.backward()
        opt.step()
        loss_value = total_loss.data / len(logits_seq)
        total_loss.free_graph()  # libere tout l'ephemere ; embedding/rnn/output sont proteges
        return loss_value

    def generate(self, prompt_ids, length=20, repetition_penalty=1.3, penalty_window=8):
        """Genere `length` tokens de plus apres prompt_ids.

        repetition_penalty : affaiblit le score des tokens recemment generes
        (divise leur logit par ce facteur) avant de choisir le suivant --
        sans ca, un decodage glouton (argmax pur) tombe facilement dans une
        boucle ("tomber tomber tomber...") des qu'il derive legerement de ce
        qu'il a vu a l'entrainement. C'est un defaut CONNU du decodage glouton
        en general (pas specifique a G) ; la penalite de repetition est la
        parade standard (utilisee par ex. dans GPT-2). Mets a 1.0 pour
        desactiver et revenir a un argmax pur.

        L'inférence détache l'état caché après chaque token : le graphe
        d'autograd de l'étape précédente est alors libéré immédiatement.
        La mémoire reste ainsi bornée par un seul pas de calcul, même si
        generate() est appelée longtemps ou très souvent."""
        tokens = list(prompt_ids)
        h = self.rnn.init_hidden()
        for tid in tokens:
            x_t = self.embedding([tid])
            h_new = self.rnn.step(x_t, h)
            h = h_new.detach()
            h_new.free_graph()

        for _ in range(length):
            logits = self.output(h)
            scores = list(logits.data if isinstance(logits.data[0], (int, float)) else logits.data[0])
            logits.free_intermediates()

            if repetition_penalty != 1.0:
                recent = set(tokens[-penalty_window:])
                for tid in recent:
                    scores[tid] = scores[tid] / repetition_penalty if scores[tid] > 0 else scores[tid] * repetition_penalty

            next_token = max(range(self.vocab_size), key=lambda i: scores[i])
            tokens.append(next_token)
            x_t = self.embedding([next_token])
            h_new = self.rnn.step(x_t, h)
            h = h_new.detach()
            h_new.free_graph()
        h.free()
        return tokens


# ============================================================
# Transformer décodeur causal (GPT miniature)
# ============================================================

class LayerNorm:
    """LayerNorm native C, paramètres gamma/beta entraînables."""

    def __init__(self, width):
        self.gamma = Tensor([[1.0] * width], requires_grad=True)
        self.beta = Tensor([[0.0] * width], requires_grad=True)
        self.gamma.protect()
        self.beta.protect()

    def __call__(self, x):
        return x.layer_norm(self.gamma, self.beta)

    def parameters(self):
        return [self.gamma, self.beta]

    def free(self):
        self.gamma.free()
        self.beta.free()


class TransformerBlock:
    """Bloc pré-normalisé : attention multi-tête causale + SwiGLU."""

    def __init__(self, d_model, n_heads, ff_dim, seed=42):
        if d_model <= 0 or n_heads <= 0 or d_model % n_heads:
            raise ValueError("d_model doit être divisible par n_heads.")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln1 = LayerNorm(d_model)
        self.qkv = nn.Dense(d_model, 3 * d_model, seed=seed)
        self.attn_proj = nn.Dense(d_model, d_model, seed=seed + 1)
        self.ln2 = LayerNorm(d_model)
        self.ff_gate = nn.Dense(d_model, ff_dim, seed=seed + 2)
        self.ff_up = nn.Dense(d_model, ff_dim, seed=seed + 3)
        self.ff_down = nn.Dense(ff_dim, d_model, seed=seed + 4)

    def __call__(self, x):
        normalized = self.ln1(x)
        qkv = self.qkv(normalized)
        q = qkv.slice_cols(0, self.d_model)
        k = qkv.slice_cols(self.d_model, self.d_model)
        v = qkv.slice_cols(2 * self.d_model, self.d_model)

        heads = []
        scale = 1.0 / (self.head_dim ** 0.5)
        for head in range(self.n_heads):
            start = head * self.head_dim
            qh = q.slice_cols(start, self.head_dim)
            kh = k.slice_cols(start, self.head_dim)
            vh = v.slice_cols(start, self.head_dim)
            scores = qh.matmul(kh.transpose()).scale(scale)
            probabilities = scores.softmax(causal=True)
            heads.append(probabilities.matmul(vh))

        attended = heads[0]
        for head_output in heads[1:]:
            attended = attended.concat_cols(head_output)
        x = x + self.attn_proj(attended)

        normalized = self.ln2(x)
        gate = self.ff_gate(normalized)
        silu_gate = gate * gate.sigmoid()
        value = self.ff_up(normalized)
        feed_forward = self.ff_down(silu_gate * value)
        return x + feed_forward

    def parameters(self):
        return (self.ln1.parameters() + self.qkv.parameters()
                + self.attn_proj.parameters() + self.ln2.parameters()
                + self.ff_gate.parameters() + self.ff_up.parameters()
                + self.ff_down.parameters())

    def free(self):
        self.ln1.free(); self.qkv.free(); self.attn_proj.free()
        self.ln2.free(); self.ff_gate.free(); self.ff_up.free(); self.ff_down.free()


class TransformerLanguageModel:
    """GPT compact, entraîné à prédire le prochain token.

    Architecture : token/position embeddings, blocs pré-LayerNorm,
    attention multi-tête causale, SwiGLU, poids de sortie liés à la
    table d'embedding et AdamW natif conseillé.

    Le calcul tensoriel et l'autograd restent dans le moteur C. Python
    orchestre les blocs, les données et la boucle d'entraînement.
    """

    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2,
                 max_seq_len=128, ff_mult=4, seed=42):
        if vocab_size < 2 or d_model <= 0 or n_layers <= 0 or max_seq_len < 2:
            raise ValueError("Configuration Transformer invalide.")
        if n_heads <= 0 or d_model % n_heads:
            raise ValueError("d_model doit être divisible par n_heads.")
        if ff_mult < 1:
            raise ValueError("ff_mult doit être >= 1.")
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.max_seq_len = int(max_seq_len)
        self.ff_mult = int(ff_mult)
        self.embedding = Embedding(vocab_size, d_model, seed=seed)
        self.position_embedding = Embedding(max_seq_len, d_model, seed=seed + 1)
        ff_dim = d_model * ff_mult
        self.blocks = [
            TransformerBlock(d_model, n_heads, ff_dim, seed=seed + 10 + i * 7)
            for i in range(n_layers)
        ]
        self.final_norm = LayerNorm(d_model)

    def parameters(self):
        params = self.embedding.parameters() + self.position_embedding.parameters()
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.final_norm.parameters())
        return params

    def embed_tokens(self, token_ids):
        """Retourne les embeddings de tokens sans encodage positionnel."""
        ids = list(token_ids)
        if not ids:
            raise ValueError("La séquence d'entrée ne peut pas être vide.")
        if any(isinstance(t, bool) or not isinstance(t, int) or t < 0 or t >= self.vocab_size for t in ids):
            raise ValueError("Un token ID est hors du vocabulaire du modèle.")
        return self.embedding(ids)

    def forward_embeddings(self, input_embeddings, position_offset=0):
        """Passe avant depuis des embeddings continus (texte, image ou multimodal).

        input_embeddings doit être une matrice T x d_model reliée à l'autograd.
        Cette entrée permet au projecteur visuel d'apprendre avec le LLM.
        """
        if not isinstance(input_embeddings, Tensor):
            raise TypeError("input_embeddings doit être un Tensor ghost.")
        rows, width = input_embeddings.shape
        if width != self.d_model or rows < 1:
            raise ValueError("La forme des embeddings doit être (T, d_model).")
        if isinstance(position_offset, bool) or not isinstance(position_offset, int) or position_offset < 0:
            raise ValueError("position_offset doit être un entier positif ou nul.")
        if position_offset + rows > self.max_seq_len:
            raise ValueError(f"Séquence trop longue : maximum {self.max_seq_len} positions.")
        positions = list(range(position_offset, position_offset + rows))
        x = input_embeddings + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return x.matmul(self.embedding.weight.transpose())

    def forward(self, token_ids):
        """Retourne les logits (T x vocab_size) pour une séquence d'IDs."""
        embeddings = self.embed_tokens(token_ids)
        return self.forward_embeddings(embeddings)

    def train_step(self, token_ids, optimizer, max_grad_norm=1.0, *, target_start=1):
        """Une mise à jour next-token.

        target_start=1 entraîne tous les tokens suivants (pré-entraînement LM).
        Pour le supervised fine-tuning, target_start=len(prompt_ids) masque les
        labels du prompt et ne rétropropage que sur la réponse/completion.
        """
        ids = list(token_ids)
        if len(ids) < 2:
            raise ValueError("Il faut au moins deux tokens pour entraîner le modèle.")
        if isinstance(target_start, bool) or not isinstance(target_start, int) or not 1 <= target_start < len(ids):
            raise ValueError("target_start doit désigner un token cible existant (>= 1).")
        if len(ids) > self.max_seq_len + 1:
            trim = len(ids) - (self.max_seq_len + 1)
            ids = ids[trim:]
            target_start = max(1, target_start - trim)
        logits_all = self.forward(ids[:-1])
        first_prediction_row = target_start - 1
        if first_prediction_row:
            logits = logits_all.transpose().slice_cols(first_prediction_row, len(ids) - target_start).transpose()
        else:
            logits = logits_all
        loss = nn.cross_entropy_loss_ids(logits, ids[target_start:])
        optimizer.zero_grad()
        loss.backward()
        if hasattr(optimizer, "clip_grad_norm") and max_grad_norm is not None:
            optimizer.clip_grad_norm(max_grad_norm)
        optimizer.step()
        value = float(loss.data)
        loss.free_graph()
        return value

    def fit(self, token_ids, optimizer, epochs=1, seq_len=None,
            steps_per_epoch=None, seed=42, max_grad_norm=1.0):
        """Entraîne sur un flux tokenisé avec fenêtres aléatoires contiguës."""
        import random
        ids = list(token_ids)
        if len(ids) < 2:
            raise ValueError("Le corpus doit contenir au moins deux tokens.")
        window = min(int(seq_len or self.max_seq_len), self.max_seq_len)
        if window < 1:
            raise ValueError("seq_len doit être positif.")
        rng = random.Random(seed)
        max_start = max(0, len(ids) - (window + 1))
        steps = int(steps_per_epoch or max(1, (len(ids) - 1) // window))
        history = []
        for _ in range(int(epochs)):
            losses = []
            for _ in range(steps):
                start = rng.randint(0, max_start) if max_start else 0
                sample = ids[start:start + window + 1]
                if len(sample) < 2:
                    continue
                losses.append(self.train_step(sample, optimizer, max_grad_norm))
            history.append(sum(losses) / len(losses) if losses else float("nan"))
        return history

    def generate(self, prompt_ids, max_new_tokens=40, temperature=0.8,
                 top_k=40, repetition_penalty=1.1, seed=None):
        """Génération autoregressive avec fenêtre de contexte bornée.

        Le graphe C de chaque passe est immédiatement libéré. Ce premier
        moteur GPT recalcule le préfixe complet (pas encore de KV cache).
        """
        import random
        tokens = list(prompt_ids)
        if not tokens:
            raise ValueError("Fournis au moins un token de départ.")
        if any(isinstance(t, bool) or not isinstance(t, int) or
               t < 0 or t >= self.vocab_size for t in tokens):
            raise ValueError("Un token ID du prompt est hors du vocabulaire du modèle.")
        import math
        if (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or
                max_new_tokens < 0 or not math.isfinite(float(temperature)) or
                temperature < 0 or not math.isfinite(float(repetition_penalty)) or
                repetition_penalty <= 0):
            raise ValueError("Paramètres de génération invalides.")
        if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0):
            raise ValueError("top_k doit être un entier positif ou None.")
        rng = random.Random(seed)
        for _ in range(int(max_new_tokens)):
            context = tokens[-self.max_seq_len:]
            logits = self.forward(context)
            scores = list(logits.data[-1])
            logits.free_graph()

            if repetition_penalty != 1.0:
                for token in set(tokens[-32:]):
                    scores[token] = (scores[token] / repetition_penalty if scores[token] > 0
                                     else scores[token] * repetition_penalty)

            if temperature == 0:
                next_token = max(range(self.vocab_size), key=lambda i: scores[i])
            else:
                scaled = [s / temperature for s in scores]
                if top_k and top_k < self.vocab_size:
                    keep = set(sorted(range(self.vocab_size), key=lambda i: scaled[i], reverse=True)[:top_k])
                else:
                    keep = None
                maximum = max(scaled[i] for i in (keep if keep is not None else range(self.vocab_size)))
                probs = [0.0] * self.vocab_size
                total = 0.0
                for i, score in enumerate(scaled):
                    if keep is None or i in keep:
                        probs[i] = pow(2.718281828459045, score - maximum)
                        total += probs[i]
                threshold = rng.random() * total
                next_token = self.vocab_size - 1
                cumulative = 0.0
                for i, probability in enumerate(probs):
                    cumulative += probability
                    if cumulative >= threshold:
                        next_token = i
                        break
            tokens.append(next_token)
        return tokens

    def state_dict(self):
        return {
            "format": "G-TransformerLM-v1",
            "config": {
                "vocab_size": self.vocab_size, "d_model": self.d_model,
                "n_heads": self.n_heads, "n_layers": self.n_layers,
                "max_seq_len": self.max_seq_len, "ff_mult": self.ff_mult,
            },
            "parameters": [p.flat_data() for p in self.parameters()],
        }

    def load_state_dict(self, state):
        if state.get("format") != "G-TransformerLM-v1":
            raise ValueError("Format de checkpoint Transformer non reconnu.")
        expected_config = {
            "vocab_size": self.vocab_size, "d_model": self.d_model,
            "n_heads": self.n_heads, "n_layers": self.n_layers,
            "max_seq_len": self.max_seq_len, "ff_mult": self.ff_mult,
        }
        if state.get("config") != expected_config:
            raise ValueError("La configuration du checkpoint ne correspond pas au modèle.")
        params = self.parameters()
        values = state.get("parameters", [])
        if len(values) != len(params):
            raise ValueError("Nombre de paramètres du checkpoint incorrect.")
        for parameter, flat in zip(params, values):
            parameter.set_flat_data(flat)

    def save(self, path):
        import json
        import os
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(self.state_dict(), stream, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def load(self, path):
        import json
        with open(path, encoding="utf-8") as stream:
            self.load_state_dict(json.load(stream))

    def free(self):
        self.embedding.free()
        self.position_embedding.free()
        for block in self.blocks:
            block.free()
        self.final_norm.free()
