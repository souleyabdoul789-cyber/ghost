#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <stdint.h>
#include "tensor.h"

/* ============================================================
 * Creation / destruction
 * ============================================================ */

static Tensor *tensor_alloc(int rows, int cols, int requires_grad) {
    size_t count;
    if (rows <= 0 || cols <= 0 || (size_t)rows > SIZE_MAX / (size_t)cols) {
        fprintf(stderr, "Ghost: invalid tensor shape (%d x %d)\n", rows, cols);
        return NULL;
    }
    count = (size_t)rows * (size_t)cols;
    if (count > SIZE_MAX / sizeof(float) || count > INT_MAX) return NULL;

    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if (!t) return NULL;
    memset(t, 0, sizeof(*t));
    t->rows = rows;
    t->cols = cols;
    t->requires_grad = requires_grad;
    t->op = OP_LEAF;
    t->data = (float *)calloc(count, sizeof(float));
    t->grad = requires_grad ? (float *)calloc(count, sizeof(float)) : NULL;
    if (!t->data || (requires_grad && !t->grad)) {
        free(t->data);
        free(t->grad);
        free(t);
        return NULL;
    }
    return t;
}

Tensor *tensor_zeros(int rows, int cols, int requires_grad) {
    return tensor_alloc(rows, cols, requires_grad);
}

Tensor *tensor_from_array(const float *values, int rows, int cols, int requires_grad) {
    if (!values) return NULL;
    Tensor *t = tensor_alloc(rows, cols, requires_grad);
    if (!t) return NULL;
    memcpy(t->data, values, sizeof(float) * (size_t)rows * (size_t)cols);
    return t;
}

Tensor *tensor_randn(int rows, int cols, int requires_grad, unsigned int seed) {
    Tensor *t = tensor_alloc(rows, cols, requires_grad);
    if (!t) return NULL;
    /* Pour une matrice W (fan_in x fan_out), fan_in est le nombre de lignes,
     * pas le nombre de colonnes. L'ancienne implementation utilisait cols. */
    int fan_in = rows;
    float scale = sqrtf(2.0f / (float)fan_in);
    /* BUG CRITIQUE CORRIGE : le xorshift ci-dessous a un point fixe en 0 (0 ^ (0<<13)
     * ^ (0>>17) ^ (0<<5) = 0 indefiniment) -- avec seed=0, TOUS les poids etaient
     * silencieusement initialises a la MEME valeur negative (-scale), ce qui tue tous
     * les neurones ReLU des l'initialisation (sortie 0 pour n'importe quelle entree).
     * On ecarte donc explicitement seed=0 (et tout seed qui degenererait a 0 en cours
     * de route ne peut pas se produire : l'etat ne peut revenir a 0 que s'il y est deja). */
    unsigned int state = (seed == 0) ? 0x9E3779B9u : seed;
    size_t count = (size_t)rows * (size_t)cols;
    for (size_t i = 0; i < count; i += 2) {
        /* Xorshift32 + Box-Muller : tensor_randn produit maintenant une
         * vraie distribution normale, sans état global libc. */
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        float u1 = ((float)state + 1.0f) / 4294967297.0f;
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        float u2 = ((float)state + 1.0f) / 4294967297.0f;
        if (u1 < 1.0e-7f) u1 = 1.0e-7f;
        float radius = sqrtf(-2.0f * logf(u1)) * scale;
        float angle = 6.2831853071795864769f * u2;
        t->data[i] = radius * cosf(angle);
        if (i + 1 < count) t->data[i + 1] = radius * sinf(angle);
    }
    return t;
}

void tensor_free(Tensor *t) {
    if (!t) return;
    free(t->data);
    free(t->grad);
    free(t->indices);
    free(t);
}

/* Collects every node reachable from root (via parent_a/parent_b) into a
 * malloc'd array, in post-order (children before parents). Marks visited=1
 * on each node as it is added, to avoid duplicates in diamond graphs. */
static int build_topo(Tensor *node, Tensor ***order, int *count, int *capacity) {
    if (!node || node->visited) return 1;
    node->visited = 1;
    if (!build_topo(node->parent_a, order, count, capacity) ||
        !build_topo(node->parent_b, order, count, capacity) ||
        !build_topo(node->parent_c, order, count, capacity)) return 0;
    if (*count >= *capacity) {
        if (*capacity > INT_MAX / 2) return 0;
        int new_capacity = *capacity * 2;
        Tensor **grown = (Tensor **)realloc(*order, sizeof(Tensor *) * (size_t)new_capacity);
        if (!grown) return 0;
        *order = grown;
        *capacity = new_capacity;
    }
    (*order)[(*count)++] = node;
    return 1;
}

static void reset_visited(Tensor *node) {
    if (!node || !node->visited) return;
    node->visited = 0;
    reset_visited(node->parent_a);
    reset_visited(node->parent_b);
    reset_visited(node->parent_c);
}

static Tensor **topo_sort(Tensor *root, int *out_count) {
    int capacity = 16;
    Tensor **order = (Tensor **)malloc(sizeof(Tensor *) * capacity);
    int count = 0;
    if (!order || !build_topo(root, &order, &count, &capacity)) {
        free(order);
        reset_visited(root);
        *out_count = 0;
        return NULL;
    }
    /* reset visited flags for future calls */
    for (int i = 0; i < count; i++) order[i]->visited = 0;
    *out_count = count;
    return order;
}

void tensor_free_graph(Tensor *t) {
    if (!t) return;
    /* BUG CRITIQUE CORRIGE : cette fonction ne liberait jamais rien nulle part dans
     * le code -- aucun appelant (Python ou C) ne l'invoquait, donc chaque tenseur
     * intermediaire cree par une operation (matmul, add, relu, ...) restait alloue
     * pour toujours. Verifie empiriquement : ~400 Mo de fuite pour 150 episodes
     * d'entrainement DQN. Desormais utilisee par les boucles d'entrainement de G.rl
     * apres chaque forward/backward, elle libere tout le sous-graphe EPHEMERE
     * (entrees, resultats intermediaires) tout en preservant les tenseurs marques
     * is_protected=1 (les poids/biais des couches, crees par dense_create) : sans
     * cette protection, liberer le graphe d'une loss aurait aussi detruit les poids
     * du modele, puisqu'ils font partie du meme graphe (parents des noeuds calcules). */
    int count;
    Tensor **order = topo_sort(t, &count);
    if (!order) return;
    for (int i = 0; i < count; i++) {
        if (!order[i]->is_protected) tensor_free(order[i]);
    }
    free(order);
}

void tensor_free_graph_keep_leaves(Tensor *t) {
    if (!t) return;
    int count;
    Tensor **order = topo_sort(t, &count);
    if (!order) return;
    for (int i = 0; i < count; i++) {
        if (order[i]->op != OP_LEAF) tensor_free(order[i]);
    }
    free(order);
}

/* ============================================================
 * Forward ops
 * ============================================================ */

static Tensor *make_result(int rows, int cols, int rg_a, int rg_b, TensorOp op,
                            Tensor *a, Tensor *b) {
    int requires_grad = rg_a || rg_b;
    Tensor *out = tensor_alloc(rows, cols, requires_grad);
    if (!out) return NULL;
    out->op = op;
    out->parent_a = a;
    out->parent_b = b;
    return out;
}

Tensor *tensor_add(Tensor *a, Tensor *b) {
    if (!a || !b) return NULL;
    /* supports same-shape add, and broadcasting b as a (1 x cols) bias row */
    if (a->rows == b->rows && a->cols == b->cols) {
        Tensor *out = make_result(a->rows, a->cols, a->requires_grad, b->requires_grad, OP_ADD, a, b);
        if (!out) return NULL;
        for (int i = 0; i < a->rows * a->cols; i++) out->data[i] = a->data[i] + b->data[i];
        return out;
    }
    if (b->rows == 1 && b->cols == a->cols) {
        Tensor *out = make_result(a->rows, a->cols, a->requires_grad, b->requires_grad, OP_ADD, a, b);
        if (!out) return NULL;
        for (int i = 0; i < a->rows; i++)
            for (int j = 0; j < a->cols; j++)
                out->data[i * a->cols + j] = a->data[i * a->cols + j] + b->data[j];
        return out;
    }
    fprintf(stderr, "G: tensor_add shape mismatch (%dx%d) + (%dx%d)\n",
            a->rows, a->cols, b->rows, b->cols);
    return NULL;
}

Tensor *tensor_sub(Tensor *a, Tensor *b) {
    if (!a || !b) return NULL;
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "G: tensor_sub shape mismatch (%dx%d) - (%dx%d)\n",
                a->rows, a->cols, b->rows, b->cols);
        return NULL;
    }
    Tensor *out = make_result(a->rows, a->cols, a->requires_grad, b->requires_grad, OP_SUB, a, b);
    if (!out) return NULL;
    for (int i = 0; i < a->rows * a->cols; i++) out->data[i] = a->data[i] - b->data[i];
    return out;
}

Tensor *tensor_mul(Tensor *a, Tensor *b) {
    if (!a || !b) return NULL;
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "G: tensor_mul shape mismatch (%dx%d) * (%dx%d)\n",
                a->rows, a->cols, b->rows, b->cols);
        return NULL;
    }
    Tensor *out = make_result(a->rows, a->cols, a->requires_grad, b->requires_grad, OP_MUL, a, b);
    if (!out) return NULL;
    for (int i = 0; i < a->rows * a->cols; i++) out->data[i] = a->data[i] * b->data[i];
    return out;
}

Tensor *tensor_matmul(Tensor *a, Tensor *b) {
    if (!a || !b) return NULL;
    if (a->cols != b->rows) {
        fprintf(stderr, "G: tensor_matmul shape mismatch (%dx%d) x (%dx%d)\n",
                a->rows, a->cols, b->rows, b->cols);
        return NULL;
    }
    Tensor *out = make_result(a->rows, b->cols, a->requires_grad, b->requires_grad, OP_MATMUL, a, b);
    if (!out) return NULL;
    /* i-k-j : accès contigu à B et à la ligne de sortie. Plus efficace
     * que i-j-k sur CPU sans BLAS, notamment sur ARM mobile. */
    for (int i = 0; i < a->rows; ++i) {
        float *out_row = out->data + (size_t)i * b->cols;
        const float *a_row = a->data + (size_t)i * a->cols;
        for (int k = 0; k < a->cols; ++k) {
            float aik = a_row[k];
            const float *b_row = b->data + (size_t)k * b->cols;
            for (int j = 0; j < b->cols; ++j)
                out_row[j] += aik * b_row[j];
        }
    }
    return out;
}

Tensor *tensor_relu(Tensor *a) {
    if (!a) return NULL;
    Tensor *out = make_result(a->rows, a->cols, a->requires_grad, 0, OP_RELU, a, NULL);
    if (!out) return NULL;
    for (int i = 0; i < a->rows * a->cols; i++)
        out->data[i] = a->data[i] > 0.0f ? a->data[i] : 0.0f;
    return out;
}

Tensor *tensor_sigmoid(Tensor *a) {
    if (!a) return NULL;
    Tensor *out = make_result(a->rows, a->cols, a->requires_grad, 0, OP_SIGMOID, a, NULL);
    if (!out) return NULL;
    for (int i = 0; i < a->rows * a->cols; i++)
        out->data[i] = 1.0f / (1.0f + expf(-a->data[i]));
    return out;
}

Tensor *tensor_tanh(Tensor *a) {
    if (!a) return NULL;
    Tensor *out = make_result(a->rows, a->cols, a->requires_grad, 0, OP_TANH, a, NULL);
    if (!out) return NULL;
    for (int i = 0; i < a->rows * a->cols; i++)
        out->data[i] = tanhf(a->data[i]);
    return out;
}

Tensor *tensor_sum(Tensor *a) {
    if (!a) return NULL;
    Tensor *out = make_result(1, 1, a->requires_grad, 0, OP_SUM, a, NULL);
    if (!out) return NULL;
    float s = 0.0f;
    for (int i = 0; i < a->rows * a->cols; i++) s += a->data[i];
    out->data[0] = s;
    return out;
}

Tensor *tensor_scale(Tensor *a, float scalar) {
    if (!a) return NULL;
    Tensor *out = make_result(a->rows, a->cols, a->requires_grad, 0, OP_SCALE, a, NULL);
    if (!out) return NULL;
    out->scalar = scalar;
    for (int i = 0; i < a->rows * a->cols; i++) out->data[i] = a->data[i] * scalar;
    return out;
}

Tensor *tensor_cross_entropy(Tensor *logits, Tensor *target) {
    if (!logits || !target || logits->rows <= 0 || logits->cols <= 0) return NULL;
    if (logits->rows != target->rows || logits->cols != target->cols) {
        fprintf(stderr, "ghostlib: tensor_cross_entropy shape mismatch (%dx%d) vs (%dx%d)\n",
                logits->rows, logits->cols, target->rows, target->cols);
        return NULL;
    }
    Tensor *out = make_result(1, 1, logits->requires_grad, 0, OP_CROSS_ENTROPY, logits, target);
    if (!out) return NULL;
    int rows = logits->rows, cols = logits->cols;
    float total = 0.0f;
    for (int i = 0; i < rows; i++) {
        float maxv = logits->data[i * cols];
        for (int j = 1; j < cols; j++) {
            float v = logits->data[i * cols + j];
            if (v > maxv) maxv = v;
        }
        float sumexp = 0.0f;
        for (int j = 0; j < cols; j++)
            sumexp += expf(logits->data[i * cols + j] - maxv);
        float logsumexp = logf(sumexp) + maxv;
        /* target est one-hot : sum_j(target_j * logits_j) = logits[classe_cible] */
        float dot = 0.0f;
        for (int j = 0; j < cols; j++)
            dot += target->data[i * cols + j] * logits->data[i * cols + j];
        total += (logsumexp - dot);
    }
    out->data[0] = total / (float)rows;
    return out;
}

Tensor *tensor_cross_entropy_ids(Tensor *logits, const int *target_ids, int count) {
    if (!logits || !target_ids || count != logits->rows || logits->cols <= 0) return NULL;
    Tensor *out = tensor_alloc(1, 1, logits->requires_grad);
    if (!out) return NULL;
    out->op = OP_CROSS_ENTROPY_IDS;
    out->parent_a = logits;
    out->indices = (int *)malloc(sizeof(int) * (size_t)count);
    if (!out->indices) { tensor_free(out); return NULL; }
    out->index_count = count;
    memcpy(out->indices, target_ids, sizeof(int) * (size_t)count);
    float total = 0.0f;
    for (int i = 0; i < count; ++i) {
        int target = target_ids[i];
        if (target < 0 || target >= logits->cols) { tensor_free(out); return NULL; }
        float maxv = logits->data[(size_t)i * logits->cols];
        for (int j = 1; j < logits->cols; ++j) {
            float v = logits->data[(size_t)i * logits->cols + j];
            if (v > maxv) maxv = v;
        }
        float sumexp = 0.0f;
        for (int j = 0; j < logits->cols; ++j)
            sumexp += expf(logits->data[(size_t)i * logits->cols + j] - maxv);
        total += logf(sumexp) + maxv - logits->data[(size_t)i * logits->cols + target];
    }
    out->data[0] = total / (float)count;
    return out;
}

/* Lookup natif : évite la matrice one-hot de taille (tokens x vocabulaire). */
Tensor *tensor_embedding(Tensor *weight, const int *indices, int count) {
    if (!weight || !indices || count <= 0) return NULL;
    Tensor *out = tensor_alloc(count, weight->cols, weight->requires_grad);
    if (!out) return NULL;
    out->indices = (int *)malloc(sizeof(int) * (size_t)count);
    if (!out->indices) { tensor_free(out); return NULL; }
    out->index_count = count;
    out->op = OP_EMBEDDING;
    out->parent_a = weight;
    memcpy(out->indices, indices, sizeof(int) * (size_t)count);
    for (int i = 0; i < count; ++i) {
        int token = indices[i];
        if (token < 0 || token >= weight->rows) {
            fprintf(stderr, "G: embedding token id %d hors limites [0, %d)\n", token, weight->rows);
            tensor_free(out);
            return NULL;
        }
        memcpy(out->data + (size_t)i * weight->cols,
               weight->data + (size_t)token * weight->cols,
               sizeof(float) * (size_t)weight->cols);
    }
    return out;
}

Tensor *tensor_transpose(Tensor *a) {
    if (!a) return NULL;
    Tensor *out = make_result(a->cols, a->rows, a->requires_grad, 0, OP_TRANSPOSE, a, NULL);
    if (!out) return NULL;
    for (int i = 0; i < a->rows; ++i)
        for (int j = 0; j < a->cols; ++j)
            out->data[(size_t)j * a->rows + i] = a->data[(size_t)i * a->cols + j];
    return out;
}

/* Softmax row-wise, calculé avec log-sum-exp et masque causal facultatif. */
Tensor *tensor_softmax(Tensor *a, int causal) {
    if (!a || (causal && a->rows != a->cols)) return NULL;
    Tensor *out = make_result(a->rows, a->cols, a->requires_grad, 0, OP_SOFTMAX, a, NULL);
    if (!out) return NULL;
    out->scalar = causal ? 1.0f : 0.0f;
    for (int i = 0; i < a->rows; ++i) {
        int last = causal ? i : a->cols - 1;
        float maxv = a->data[(size_t)i * a->cols];
        for (int j = 1; j <= last; ++j) {
            float v = a->data[(size_t)i * a->cols + j];
            if (v > maxv) maxv = v;
        }
        float sum = 0.0f;
        for (int j = 0; j <= last; ++j)
            sum += expf(a->data[(size_t)i * a->cols + j] - maxv);
        for (int j = 0; j < a->cols; ++j) {
            if (j <= last)
                out->data[(size_t)i * a->cols + j] =
                    expf(a->data[(size_t)i * a->cols + j] - maxv) / sum;
            else
                out->data[(size_t)i * a->cols + j] = 0.0f;
        }
    }
    return out;
}

/* LayerNorm indépendante pour chaque ligne : (x-mean)/sqrt(var+eps)*gamma+beta. */
Tensor *tensor_layer_norm(Tensor *x, Tensor *gamma, Tensor *beta, float eps) {
    if (!x || !gamma || !beta || gamma->rows != 1 || beta->rows != 1 ||
        gamma->cols != x->cols || beta->cols != x->cols ||
        !(eps > 0.0f) || !isfinite(eps)) {
        fprintf(stderr, "G: tensor_layer_norm shape/epsilon invalide\n");
        return NULL;
    }
    Tensor *out = tensor_alloc(x->rows, x->cols,
        x->requires_grad || gamma->requires_grad || beta->requires_grad);
    if (!out) return NULL;
    out->op = OP_LAYER_NORM;
    out->parent_a = x;
    out->parent_b = gamma;
    out->parent_c = beta;
    out->scalar = eps;
    for (int i = 0; i < x->rows; ++i) {
        float mean = 0.0f;
        for (int j = 0; j < x->cols; ++j) mean += x->data[(size_t)i * x->cols + j];
        mean /= (float)x->cols;
        float var = 0.0f;
        for (int j = 0; j < x->cols; ++j) {
            float d = x->data[(size_t)i * x->cols + j] - mean;
            var += d * d;
        }
        var /= (float)x->cols;
        float inv = 1.0f / sqrtf(var + eps);
        for (int j = 0; j < x->cols; ++j) {
            float norm = (x->data[(size_t)i * x->cols + j] - mean) * inv;
            out->data[(size_t)i * x->cols + j] = norm * gamma->data[j] + beta->data[j];
        }
    }
    return out;
}

Tensor *tensor_detach(Tensor *a) {
    if (!a) return NULL;
    return tensor_from_array(a->data, a->rows, a->cols, 0);
}

Tensor *tensor_slice_cols(Tensor *a, int start, int count) {
    if (!a || start < 0 || count <= 0 || start > a->cols - count) return NULL;
    Tensor *out = make_result(a->rows, count, a->requires_grad, 0, OP_SLICE_COLS, a, NULL);
    if (!out) return NULL;
    out->scalar = (float)start;
    for (int i = 0; i < a->rows; ++i)
        memcpy(out->data + (size_t)i * count,
               a->data + (size_t)i * a->cols + start,
               sizeof(float) * (size_t)count);
    return out;
}

Tensor *tensor_concat_cols(Tensor *a, Tensor *b) {
    if (!a || !b || a->rows != b->rows || a->cols > INT_MAX - b->cols) return NULL;
    Tensor *out = make_result(a->rows, a->cols + b->cols,
        a->requires_grad, b->requires_grad, OP_CONCAT_COLS, a, b);
    if (!out) return NULL;
    for (int i = 0; i < a->rows; ++i) {
        memcpy(out->data + (size_t)i * out->cols,
               a->data + (size_t)i * a->cols, sizeof(float) * (size_t)a->cols);
        memcpy(out->data + (size_t)i * out->cols + a->cols,
               b->data + (size_t)i * b->cols, sizeof(float) * (size_t)b->cols);
    }
    return out;
}

Tensor *tensor_concat_rows(Tensor *a, Tensor *b) {
    if (!a || !b || a->cols != b->cols || a->rows > INT_MAX - b->rows) return NULL;
    Tensor *out = make_result(a->rows + b->rows, a->cols,
        a->requires_grad, b->requires_grad, OP_CONCAT_ROWS, a, b);
    if (!out) return NULL;
    size_t a_count = (size_t)a->rows * (size_t)a->cols;
    size_t b_count = (size_t)b->rows * (size_t)b->cols;
    memcpy(out->data, a->data, sizeof(float) * a_count);
    memcpy(out->data + a_count, b->data, sizeof(float) * b_count);
    return out;
}

/* ============================================================
 * Autograd
 * ============================================================ */

void tensor_zero_grad(Tensor *t) {
    if (!t) return;
    int count;
    Tensor **order = topo_sort(t, &count);
    if (!order) return;
    for (int i = 0; i < count; i++) {
        Tensor *n = order[i];
        if (n->requires_grad && n->grad)
            memset(n->grad, 0, sizeof(float) * n->rows * n->cols);
    }
    free(order);
}

static void accumulate(Tensor *dst, int idx, float value) {
    if (dst && dst->requires_grad && dst->grad) dst->grad[idx] += value;
}

void tensor_backward(Tensor *t) {
    if (!t) return;
    if (t->rows != 1 || t->cols != 1) {
        fprintf(stderr, "G: tensor_backward requires a scalar (1x1) tensor\n");
        return;
    }
    int count;
    Tensor **order = topo_sort(t, &count);
    if (!order) return;

    if (!t->grad) t->grad = (float *)calloc(1, sizeof(float));
    if (!t->grad) { free(order); return; }
    t->grad[0] = 1.0f; /* dL/dL = 1 */

    for (int idx = count - 1; idx >= 0; idx--) {
        Tensor *n = order[idx];
        Tensor *a = n->parent_a;
        Tensor *b = n->parent_b;
        if (!n->grad) continue;

        switch (n->op) {
            case OP_LEAF:
                break;

            case OP_ADD:
                if (a->rows == n->rows && a->cols == n->cols) {
                    for (int i = 0; i < n->rows * n->cols; i++) accumulate(a, i, n->grad[i]);
                }
                if (b->rows == 1 && b->cols == n->cols && a->rows == n->rows) {
                    /* b was broadcast as a bias row: sum gradient over rows */
                    for (int j = 0; j < n->cols; j++) {
                        float s = 0.0f;
                        for (int i = 0; i < n->rows; i++) s += n->grad[i * n->cols + j];
                        accumulate(b, j, s);
                    }
                } else if (b->rows == n->rows && b->cols == n->cols) {
                    for (int i = 0; i < n->rows * n->cols; i++) accumulate(b, i, n->grad[i]);
                }
                break;

            case OP_SUB:
                for (int i = 0; i < n->rows * n->cols; i++) {
                    accumulate(a, i, n->grad[i]);
                    accumulate(b, i, -n->grad[i]);
                }
                break;

            case OP_MUL:
                for (int i = 0; i < n->rows * n->cols; i++) {
                    accumulate(a, i, n->grad[i] * b->data[i]);
                    accumulate(b, i, n->grad[i] * a->data[i]);
                }
                break;

            case OP_MATMUL:
                /* C = A(mxk) . B(kxn) ; dA = dC . B^T ; dB = A^T . dC */
                if (a->requires_grad) {
                    for (int i = 0; i < a->rows; i++)
                        for (int k = 0; k < a->cols; k++) {
                            float s = 0.0f;
                            for (int j = 0; j < n->cols; j++)
                                s += n->grad[i * n->cols + j] * b->data[k * b->cols + j];
                            accumulate(a, i * a->cols + k, s);
                        }
                }
                if (b->requires_grad) {
                    for (int k = 0; k < b->rows; k++)
                        for (int j = 0; j < b->cols; j++) {
                            float s = 0.0f;
                            for (int i = 0; i < a->rows; i++)
                                s += a->data[i * a->cols + k] * n->grad[i * n->cols + j];
                            accumulate(b, k * b->cols + j, s);
                        }
                }
                break;

            case OP_RELU:
                for (int i = 0; i < n->rows * n->cols; i++)
                    accumulate(a, i, n->data[i] > 0.0f ? n->grad[i] : 0.0f);
                break;

            case OP_SIGMOID:
                for (int i = 0; i < n->rows * n->cols; i++) {
                    float s = n->data[i];
                    accumulate(a, i, n->grad[i] * s * (1.0f - s));
                }
                break;

            case OP_TANH:
                for (int i = 0; i < n->rows * n->cols; i++) {
                    float th = n->data[i];
                    accumulate(a, i, n->grad[i] * (1.0f - th * th));
                }
                break;

            case OP_SUM:
                for (int i = 0; i < a->rows * a->cols; i++)
                    accumulate(a, i, n->grad[0]);
                break;

            case OP_SCALE:
                for (int i = 0; i < n->rows * n->cols; i++)
                    accumulate(a, i, n->grad[i] * n->scalar);
                break;

            case OP_EMBEDDING:
                if (a->requires_grad) {
                    for (int i = 0; i < n->index_count; ++i) {
                        int token = n->indices[i];
                        for (int j = 0; j < a->cols; ++j)
                            accumulate(a, token * a->cols + j,
                                n->grad[(size_t)i * n->cols + j]);
                    }
                }
                break;

            case OP_TRANSPOSE:
                for (int i = 0; i < a->rows; ++i)
                    for (int j = 0; j < a->cols; ++j)
                        accumulate(a, i * a->cols + j,
                            n->grad[(size_t)j * n->cols + i]);
                break;

            case OP_SLICE_COLS: {
                int start = (int)n->scalar;
                for (int i = 0; i < a->rows; ++i)
                    for (int j = 0; j < n->cols; ++j)
                        accumulate(a, i * a->cols + start + j,
                            n->grad[(size_t)i * n->cols + j]);
                break;
            }

            case OP_CONCAT_COLS:
                for (int i = 0; i < a->rows; ++i) {
                    for (int j = 0; j < a->cols; ++j)
                        accumulate(a, i * a->cols + j,
                            n->grad[(size_t)i * n->cols + j]);
                    for (int j = 0; j < b->cols; ++j)
                        accumulate(b, i * b->cols + j,
                            n->grad[(size_t)i * n->cols + a->cols + j]);
                }
                break;

            case OP_CONCAT_ROWS: {
                size_t a_count = (size_t)a->rows * (size_t)a->cols;
                size_t b_count = (size_t)b->rows * (size_t)b->cols;
                for (size_t i = 0; i < a_count; ++i)
                    accumulate(a, (int)i, n->grad[i]);
                for (size_t i = 0; i < b_count; ++i)
                    accumulate(b, (int)i, n->grad[a_count + i]);
                break;
            }

            case OP_SOFTMAX:
                for (int i = 0; i < n->rows; ++i) {
                    float dot = 0.0f;
                    for (int j = 0; j < n->cols; ++j)
                        dot += n->grad[(size_t)i * n->cols + j] * n->data[(size_t)i * n->cols + j];
                    for (int j = 0; j < n->cols; ++j) {
                        int idx = i * n->cols + j;
                        accumulate(a, idx, n->data[idx] * (n->grad[idx] - dot));
                    }
                }
                break;

            case OP_LAYER_NORM: {
                Tensor *gamma = b;
                Tensor *beta = n->parent_c;
                int width = a->cols;
                float inv_width = 1.0f / (float)width;
                for (int i = 0; i < a->rows; ++i) {
                    float mean = 0.0f;
                    for (int j = 0; j < width; ++j)
                        mean += a->data[(size_t)i * width + j];
                    mean *= inv_width;
                    float var = 0.0f;
                    for (int j = 0; j < width; ++j) {
                        float d = a->data[(size_t)i * width + j] - mean;
                        var += d * d;
                    }
                    var *= inv_width;
                    float inv = 1.0f / sqrtf(var + n->scalar);
                    float sum_dyg = 0.0f;
                    float sum_dyg_xhat = 0.0f;
                    for (int j = 0; j < width; ++j) {
                        int idx = i * width + j;
                        float xhat = (a->data[idx] - mean) * inv;
                        float dy = n->grad[idx];
                        float dyg = dy * gamma->data[j];
                        sum_dyg += dyg;
                        sum_dyg_xhat += dyg * xhat;
                        accumulate(gamma, j, dy * xhat);
                        accumulate(beta, j, dy);
                    }
                    for (int j = 0; j < width; ++j) {
                        int idx = i * width + j;
                        float xhat = (a->data[idx] - mean) * inv;
                        float dyg = n->grad[idx] * gamma->data[j];
                        float dx = inv * inv_width *
                            ((float)width * dyg - sum_dyg - xhat * sum_dyg_xhat);
                        accumulate(a, idx, dx);
                    }
                }
                break;
            }

            case OP_CROSS_ENTROPY: {
                /* dL/dlogits = (softmax(logits) - target) / nb_lignes --
                 * c'est pour cette simplicite (et la stabilite numerique du
                 * log-sum-exp) qu'on fusionne softmax+entropie croisee plutot
                 * que de les composer a partir d'operations separees. */
                int rows = a->rows, cols = a->cols;
                for (int i = 0; i < rows; i++) {
                    float maxv = a->data[i * cols];
                    for (int j = 1; j < cols; j++) {
                        float v = a->data[i * cols + j];
                        if (v > maxv) maxv = v;
                    }
                    float sumexp = 0.0f;
                    for (int j = 0; j < cols; j++)
                        sumexp += expf(a->data[i * cols + j] - maxv);
                    for (int j = 0; j < cols; j++) {
                        float softmax_ij = expf(a->data[i * cols + j] - maxv) / sumexp;
                        float grad_ij = (softmax_ij - b->data[i * cols + j]) / (float)rows;
                        accumulate(a, i * cols + j, n->grad[0] * grad_ij);
                    }
                }
                break;
            }

            case OP_CROSS_ENTROPY_IDS: {
                int rows = a->rows, cols = a->cols;
                for (int i = 0; i < rows; ++i) {
                    float maxv = a->data[(size_t)i * cols];
                    for (int j = 1; j < cols; ++j) {
                        float v = a->data[(size_t)i * cols + j];
                        if (v > maxv) maxv = v;
                    }
                    float sumexp = 0.0f;
                    for (int j = 0; j < cols; ++j)
                        sumexp += expf(a->data[(size_t)i * cols + j] - maxv);
                    int target = n->indices[i];
                    for (int j = 0; j < cols; ++j) {
                        float p = expf(a->data[(size_t)i * cols + j] - maxv) / sumexp;
                        if (j == target) p -= 1.0f;
                        accumulate(a, i * cols + j, n->grad[0] * p / (float)rows);
                    }
                }
                break;
            }
        }
    }
    free(order);
}

/* ============================================================
 * Utilities
 * ============================================================ */

void tensor_print(const Tensor *t, const char *label) {
    if (label) printf("%s ", label);
    printf("Tensor(%dx%d):\n", t->rows, t->cols);
    for (int i = 0; i < t->rows; i++) {
        printf("  [");
        for (int j = 0; j < t->cols; j++) {
            printf("%8.4f", t->data[i * t->cols + j]);
            if (j < t->cols - 1) printf(", ");
        }
        printf("]\n");
    }
}
