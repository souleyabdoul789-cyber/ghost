#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "nn.h"

/* ============================================================
 * Dense layer
 * ============================================================ */

Dense *dense_create(int in_features, int out_features, Activation activation, unsigned int seed) {
    if (in_features <= 0 || out_features <= 0 || activation < ACT_NONE || activation > ACT_TANH) return NULL;
    Dense *layer = (Dense *)malloc(sizeof(Dense));
    if (!layer) return NULL;
    layer->weights = tensor_randn(in_features, out_features, 1, seed);
    layer->bias = tensor_zeros(1, out_features, 1);
    if (!layer->weights || !layer->bias) {
        tensor_free(layer->weights);
        tensor_free(layer->bias);
        free(layer);
        return NULL;
    }
    /* He pour ReLU ; Xavier normal pour les activations bornées et les
     * projections linéaires. tensor_randn fournit He normal. */
    if (activation != ACT_RELU) {
        float factor = sqrtf((float)in_features / (float)(in_features + out_features));
        size_t n = (size_t)in_features * (size_t)out_features;
        for (size_t i = 0; i < n; ++i) layer->weights->data[i] *= factor;
    }
    layer->weights->is_protected = 1; /* survit a tensor_free_graph() -- voir tensor.c */
    layer->bias->is_protected = 1;
    layer->activation = activation;
    layer->in_features = in_features;
    layer->out_features = out_features;
    return layer;
}

void dense_free(Dense *layer) {
    if (!layer) return;
    tensor_free(layer->weights);
    tensor_free(layer->bias);
    free(layer);
}

Tensor *dense_forward(Dense *layer, Tensor *x) {
    if (!layer || !x || x->cols != layer->in_features) return NULL;
    Tensor *z = tensor_matmul(x, layer->weights);
    if (!z) return NULL;
    Tensor *z_biased = tensor_add(z, layer->bias);
    if (!z_biased) {
        tensor_free_graph_keep_leaves(z);
        return NULL;
    }

    switch (layer->activation) {
        case ACT_RELU: {
            Tensor *out = tensor_relu(z_biased);
            if (!out) tensor_free_graph_keep_leaves(z_biased);
            return out;
        }
        case ACT_SIGMOID: {
            Tensor *out = tensor_sigmoid(z_biased);
            if (!out) tensor_free_graph_keep_leaves(z_biased);
            return out;
        }
        case ACT_TANH: {
            Tensor *out = tensor_tanh(z_biased);
            if (!out) tensor_free_graph_keep_leaves(z_biased);
            return out;
        }
        case ACT_NONE:
        default:          return z_biased;
    }
}

/* ============================================================
 * SGD optimizer
 * ============================================================ */

SGD *sgd_create(float lr) {
    if (!(lr > 0.0f) || !isfinite(lr)) return NULL;
    SGD *opt = (SGD *)malloc(sizeof(SGD));
    if (!opt) return NULL;
    opt->params = NULL;
    opt->n_params = 0;
    opt->lr = lr;
    return opt;
}

void sgd_free(SGD *opt) {
    if (!opt) return;
    free(opt->params); /* does NOT free the tensors themselves, they belong to layers */
    free(opt);
}

void sgd_add_param(SGD *opt, Tensor *param) {
    if (!opt || !param) return;
    for (int i = 0; i < opt->n_params; ++i) if (opt->params[i] == param) return;
    Tensor **items = (Tensor **)realloc(opt->params, sizeof(Tensor *) * (size_t)(opt->n_params + 1));
    if (!items) return;
    opt->params = items;
    opt->params[opt->n_params++] = param;
}

void sgd_step(SGD *opt) {
    if (!opt) return;
    for (int p = 0; p < opt->n_params; p++) {
        Tensor *t = opt->params[p];
        if (!t->requires_grad || !t->grad) continue;
        for (int i = 0; i < t->rows * t->cols; i++)
            t->data[i] -= opt->lr * t->grad[i];
    }
}

void sgd_zero_grad(SGD *opt) {
    if (!opt) return;
    for (int p = 0; p < opt->n_params; p++) {
        Tensor *t = opt->params[p];
        if (t->requires_grad && t->grad)
            for (int i = 0; i < t->rows * t->cols; i++) t->grad[i] = 0.0f;
    }
}

/* ============================================================
 * AdamW
 * ============================================================ */

AdamW *adamw_create(float lr, float beta1, float beta2, float eps, float weight_decay) {
    if (!(lr > 0.0f) || !(beta1 >= 0.0f && beta1 < 1.0f) ||
        !(beta2 >= 0.0f && beta2 < 1.0f) || !(eps > 0.0f) ||
        !(weight_decay >= 0.0f) || !isfinite(lr) || !isfinite(beta1) ||
        !isfinite(beta2) || !isfinite(eps) || !isfinite(weight_decay))
        return NULL;
    AdamW *opt = (AdamW *)calloc(1, sizeof(AdamW));
    if (!opt) return NULL;
    opt->lr = lr;
    opt->beta1 = beta1;
    opt->beta2 = beta2;
    opt->eps = eps;
    opt->weight_decay = weight_decay;
    return opt;
}

int adamw_add_param(AdamW *opt, Tensor *param) {
    if (!opt || !param || !param->requires_grad || !param->grad) return 0;
    for (int i = 0; i < opt->n_params; ++i)
        if (opt->params[i] == param) return 1;

    float *m = (float *)calloc((size_t)param->rows * param->cols, sizeof(float));
    float *v = (float *)calloc((size_t)param->rows * param->cols, sizeof(float));
    Tensor **params = (Tensor **)malloc(sizeof(Tensor *) * (size_t)(opt->n_params + 1));
    float **mom1 = (float **)malloc(sizeof(float *) * (size_t)(opt->n_params + 1));
    float **mom2 = (float **)malloc(sizeof(float *) * (size_t)(opt->n_params + 1));
    if (!m || !v || !params || !mom1 || !mom2) {
        free(m); free(v); free(params); free(mom1); free(mom2);
        return 0;
    }
    if (opt->n_params > 0) {
        memcpy(params, opt->params, sizeof(Tensor *) * (size_t)opt->n_params);
        memcpy(mom1, opt->m, sizeof(float *) * (size_t)opt->n_params);
        memcpy(mom2, opt->v, sizeof(float *) * (size_t)opt->n_params);
    }
    params[opt->n_params] = param;
    mom1[opt->n_params] = m;
    mom2[opt->n_params] = v;
    free(opt->params); free(opt->m); free(opt->v);
    opt->params = params; opt->m = mom1; opt->v = mom2;
    opt->n_params++;
    return 1;
}

void adamw_step(AdamW *opt) {
    if (!opt) return;
    opt->step++;
    float correction1 = 1.0f - powf(opt->beta1, (float)opt->step);
    float correction2 = 1.0f - powf(opt->beta2, (float)opt->step);
    for (int p = 0; p < opt->n_params; ++p) {
        Tensor *t = opt->params[p];
        if (!t || !t->grad || !t->data) continue;
        size_t n = (size_t)t->rows * (size_t)t->cols;
        for (size_t i = 0; i < n; ++i) {
            float g = t->grad[i];
            opt->m[p][i] = opt->beta1 * opt->m[p][i] + (1.0f - opt->beta1) * g;
            opt->v[p][i] = opt->beta2 * opt->v[p][i] + (1.0f - opt->beta2) * g * g;
            float m_hat = opt->m[p][i] / correction1;
            float v_hat = opt->v[p][i] / correction2;
            /* Decoupled weight decay : ne modifie pas le gradient Adam.
             * Par convention, on ne decay pas les vecteurs (bias, gamma,
             * beta de LayerNorm), seulement les matrices de poids. */
            float decay = (t->rows > 1 && t->cols > 1) ? opt->weight_decay : 0.0f;
            t->data[i] -= opt->lr * (m_hat / (sqrtf(v_hat) + opt->eps)
                                     + decay * t->data[i]);
        }
    }
}

void adamw_zero_grad(AdamW *opt) {
    if (!opt) return;
    for (int p = 0; p < opt->n_params; ++p) {
        Tensor *t = opt->params[p];
        if (t && t->grad)
            memset(t->grad, 0, sizeof(float) * (size_t)t->rows * (size_t)t->cols);
    }
}

float adamw_clip_grad_norm(AdamW *opt, float max_norm) {
    if (!opt || !(max_norm > 0.0f)) return 0.0f;
    double sum_sq = 0.0;
    for (int p = 0; p < opt->n_params; ++p) {
        Tensor *t = opt->params[p];
        if (!t || !t->grad) continue;
        size_t n = (size_t)t->rows * (size_t)t->cols;
        for (size_t i = 0; i < n; ++i) sum_sq += (double)t->grad[i] * (double)t->grad[i];
    }
    float norm = (float)sqrt(sum_sq);
    if (norm > max_norm) {
        float scale = max_norm / (norm + 1.0e-12f);
        for (int p = 0; p < opt->n_params; ++p) {
            Tensor *t = opt->params[p];
            if (!t || !t->grad) continue;
            size_t n = (size_t)t->rows * (size_t)t->cols;
            for (size_t i = 0; i < n; ++i) t->grad[i] *= scale;
        }
    }
    return norm;
}

void adamw_free(AdamW *opt) {
    if (!opt) return;
    for (int p = 0; p < opt->n_params; ++p) { free(opt->m[p]); free(opt->v[p]); }
    free(opt->params); free(opt->m); free(opt->v); free(opt);
}

/* ============================================================
 * Loss functions
 * ============================================================ */

Tensor *loss_mse(Tensor *pred, Tensor *target) {
    if (!pred || !target || pred->rows != target->rows || pred->cols != target->cols) return NULL;
    Tensor *diff = tensor_sub(pred, target);
    if (!diff) return NULL;
    Tensor *sq = tensor_mul(diff, diff);
    if (!sq) { tensor_free_graph_keep_leaves(diff); return NULL; }
    Tensor *total = tensor_sum(sq);
    if (!total) { tensor_free_graph_keep_leaves(sq); return NULL; }
    size_t n = (size_t)pred->rows * (size_t)pred->cols;
    Tensor *mean = tensor_scale(total, 1.0f / (float)n);
    if (!mean) tensor_free_graph_keep_leaves(total);
    return mean;
}
