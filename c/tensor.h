#ifndef GHOSTLIB_TENSOR_H
#define GHOSTLIB_TENSOR_H

#if defined(_WIN32) && defined(GHOST_BUILD_DLL)
#define GHOST_API __declspec(dllexport)
#else
#define GHOST_API
#endif

/*
 * G - tensor.h
 * ----------------------------------------------------------
 * Coeur du moteur de tenseurs + autograd, en C pur (aucune
 * dependance externe, portable sur ARM32/ARM64/x86).
 *
 * Un Tensor est un tableau 2D (matrice) de float32 :
 *   - rows x cols
 *   - data[i*cols + j] = element (i, j)
 *
 * Chaque Tensor peut faire partie d'un graphe de calcul :
 *   - si requires_grad == 1, il accumule son gradient dans `grad`
 *   - op / parent_a / parent_b decrivent comment il a ete produit
 *   - tensor_backward() remonte le graphe et calcule tous les
 *     gradients via la regle de la chaine (backprop manuelle).
 *
 * Ce n'est PAS un moteur generique a N dimensions : on reste en
 * 2D (vecteurs = matrices a 1 ligne ou 1 colonne) pour rester
 * simple, rapide et fiable sur du materiel contraint (telephone).
 */

typedef enum {
    OP_LEAF,      /* tenseur cree directement par l'utilisateur */
    OP_ADD,
    OP_SUB,
    OP_MUL,       /* multiplication elementwise */
    OP_MATMUL,    /* produit matriciel */
    OP_RELU,
    OP_SIGMOID,
    OP_TANH,
    OP_SUM,       /* reduction -> scalaire (tenseur 1x1) */
    OP_SCALE,     /* multiplication par une constante float */
    OP_CROSS_ENTROPY, /* softmax + entropie croisee fusionnes, stable numeriquement */
    OP_CROSS_ENTROPY_IDS, /* cible token sous forme d'identifiant */
    OP_EMBEDDING,     /* extraction de lignes d'une table d'embeddings */
    OP_TRANSPOSE,
    OP_SOFTMAX,       /* softmax ligne par ligne, causal optionnel */
    OP_LAYER_NORM,    /* normalisation par ligne avec gamma et beta */
    OP_SLICE_COLS,
    OP_CONCAT_COLS,
    OP_CONCAT_ROWS
} TensorOp;

typedef struct Tensor {
    float *data;          /* rows * cols valeurs, row-major */
    float *grad;          /* meme taille que data, ou NULL si requires_grad == 0 */
    int rows;
    int cols;
    int requires_grad;

    /* graphe de calcul (pour l'autograd) */
    TensorOp op;
    struct Tensor *parent_a;
    struct Tensor *parent_b;
    struct Tensor *parent_c;   /* utilise par OP_LAYER_NORM */

    /* comptage de reference simplifie : on libere explicitement */
    int visited;          /* utilise en interne pour le tri topologique */
    float scalar;         /* utilise seulement par OP_SCALE */
    int is_protected;     /* si 1, tensor_free_graph() ne libere JAMAIS ce noeud (voir dense_create) */
    int *indices;         /* indices copies par OP_EMBEDDING */
    int index_count;
} Tensor;

/* ---- creation / destruction ---- */

/* Cree un tenseur rempli de zeros. */
GHOST_API Tensor *tensor_zeros(int rows, int cols, int requires_grad);

/* Cree un tenseur a partir d'un tableau de floats fourni (copie les valeurs). */
GHOST_API Tensor *tensor_from_array(const float *values, int rows, int cols, int requires_grad);

/* Cree un tenseur avec une distribution normale He (variance 2/fan_in). */
GHOST_API Tensor *tensor_randn(int rows, int cols, int requires_grad, unsigned int seed);

/* Libere un seul tenseur (ne touche pas a ses parents). */
GHOST_API void tensor_free(Tensor *t);

/* Libere un tenseur ET tout le sous-graphe de calcul qui a servi a le produire
 * (utile en fin d'entrainement pour eviter les fuites memoire). Ne libere pas
 * les feuilles marquees comme parametres persistants (voir tensor_detach). */
GHOST_API void tensor_free_graph(Tensor *t);

/* Comme tensor_free_graph, mais NE libere PAS les feuilles (op == OP_LEAF) :
 * les poids/biais persistants d'un modele, ou toute entree que l'appelant
 * veut liberer lui-meme (via tensor_free). Libere uniquement les tenseurs
 * INTERMEDIAIRES produits par les operations (add, matmul, relu, etc).
 *
 * POURQUOI CETTE FONCTION EXISTE : chaque forward() alloue en C de nouveaux
 * tenseurs intermediaires qui ne sont jamais liberes automatiquement (il n'y
 * a pas de ramasse-miettes cote C, et le wrapper Python ne les libere pas
 * tout seul). Sans appeler cette fonction apres chaque forward(+backward),
 * la memoire augmente sans limite -- negligeable sur de petits tenseurs
 * (ex: XOR a 2 entrees), mais fatal sur des etats plus grands (ex: un
 * gridworld encode en 176 valeurs) ou beaucoup d'iterations. */
GHOST_API void tensor_free_graph_keep_leaves(Tensor *t);

/* ---- operations (construisent le graphe si un des operandes requires_grad) ---- */

GHOST_API Tensor *tensor_add(Tensor *a, Tensor *b);
GHOST_API Tensor *tensor_sub(Tensor *a, Tensor *b);
GHOST_API Tensor *tensor_mul(Tensor *a, Tensor *b);      /* elementwise */
GHOST_API Tensor *tensor_matmul(Tensor *a, Tensor *b);   /* produit matriciel classique */
GHOST_API Tensor *tensor_relu(Tensor *a);
GHOST_API Tensor *tensor_sigmoid(Tensor *a);
GHOST_API Tensor *tensor_tanh(Tensor *a);
GHOST_API Tensor *tensor_sum(Tensor *a);                 /* -> tenseur 1x1 */
GHOST_API Tensor *tensor_scale(Tensor *a, float scalar); /* a * scalar (constante non-apprenable) */

/* Operations natives pour les modèles de langage. */
GHOST_API Tensor *tensor_embedding(Tensor *weight, const int *indices, int count);
GHOST_API Tensor *tensor_transpose(Tensor *a);
GHOST_API Tensor *tensor_softmax(Tensor *a, int causal);
GHOST_API Tensor *tensor_layer_norm(Tensor *x, Tensor *gamma, Tensor *beta, float eps);
GHOST_API Tensor *tensor_detach(Tensor *a);
GHOST_API Tensor *tensor_slice_cols(Tensor *a, int start, int count);
GHOST_API Tensor *tensor_concat_cols(Tensor *a, Tensor *b);
GHOST_API Tensor *tensor_concat_rows(Tensor *a, Tensor *b);

/* Softmax + entropie croisee, fusionnes (comme torch.nn.CrossEntropyLoss) :
 * plus stable numeriquement et plus simple a deriver que softmax puis log puis
 * mse separement. `logits` = scores bruts du reseau (PAS deja passes par une
 * activation), `target` = vecteurs one-hot de meme forme. Renvoie un scalaire
 * (1x1), moyenne sur les lignes (le batch). */
GHOST_API Tensor *tensor_cross_entropy(Tensor *logits, Tensor *target);
GHOST_API Tensor *tensor_cross_entropy_ids(Tensor *logits, const int *target_ids, int count);

/* ---- autograd ---- */

/* Met tous les gradients du sous-graphe a zero. */
GHOST_API void tensor_zero_grad(Tensor *t);

/* Calcule dt/dparametres pour tout le graphe en amont de t.
 * t doit etre un scalaire (1x1), typiquement une loss. */
GHOST_API void tensor_backward(Tensor *t);

/* ---- utilitaires ---- */

GHOST_API void tensor_print(const Tensor *t, const char *label);

#endif /* GHOSTLIB_TENSOR_H */
