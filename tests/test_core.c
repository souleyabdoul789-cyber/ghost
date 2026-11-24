#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include "nn.h"

static int nearf(float a, float b, float eps) { return fabsf(a - b) <= eps; }

static float layer_norm_objective(const float *x, const float *gamma,
                                  const float *beta, const float *weights) {
    float mean = (x[0] + x[1] + x[2]) / 3.0f;
    float variance = 0.0f;
    for (int i = 0; i < 3; ++i) {
        float d = x[i] - mean;
        variance += d * d;
    }
    variance /= 3.0f;
    float inv = 1.0f / sqrtf(variance + 1.0e-5f);
    float loss = 0.0f;
    for (int i = 0; i < 3; ++i)
        loss += (((x[i] - mean) * inv) * gamma[i] + beta[i]) * weights[i];
    return loss;
}

static void test_embedding_lookup_and_gradient(void) {
    const float wv[] = {1,2, 3,4, 5,6};
    const int ids[] = {2,1,2};
    Tensor *w = tensor_from_array(wv, 3, 2, 1);
    w->is_protected = 1;
    Tensor *e = tensor_embedding(w, ids, 3);
    assert(e && e->rows == 3 && e->cols == 2);
    assert(nearf(e->data[0], 5, 1e-6f) && nearf(e->data[1], 6, 1e-6f));
    Tensor *loss = tensor_sum(e);
    tensor_backward(loss);
    assert(nearf(w->grad[0], 0, 1e-6f));
    assert(nearf(w->grad[2], 1, 1e-6f));
    assert(nearf(w->grad[4], 2, 1e-6f));
    tensor_free_graph(loss);
    tensor_free(w);
}

static void test_cross_entropy_ids(void) {
    const float lv[] = {2,1,0, 0,1,2};
    const int ids[] = {0,2};
    Tensor *logits = tensor_from_array(lv, 2, 3, 1);
    Tensor *loss = tensor_cross_entropy_ids(logits, ids, 2);
    assert(loss && isfinite(loss->data[0]) && loss->data[0] > 0);
    tensor_backward(loss);
    for (int r = 0; r < 2; ++r) {
        float sum = 0;
        for (int c = 0; c < 3; ++c) sum += logits->grad[r * 3 + c];
        assert(nearf(sum, 0.0f, 1e-5f));
    }
    assert(logits->grad[0] < 0.0f && logits->grad[5] < 0.0f);
    tensor_free_graph(loss);
}

static void test_softmax_causal(void) {
    const float v[] = {1,2,3, 3,2,1, 0,0,0};
    Tensor *x = tensor_from_array(v, 3, 3, 1);
    Tensor *p = tensor_softmax(x, 1);
    assert(p);
    assert(nearf(p->data[0], 1.0f, 1e-6f));
    assert(nearf(p->data[1], 0.0f, 1e-6f));
    assert(nearf(p->data[2], 0.0f, 1e-6f));
    assert(nearf(p->data[3] + p->data[4], 1.0f, 1e-6f));
    assert(nearf(p->data[5], 0.0f, 1e-6f));
    Tensor *weights = tensor_from_array((float[]){1,2,3, 2,1,0, 3,2,1}, 3, 3, 0);
    Tensor *product = tensor_mul(p, weights);
    Tensor *loss = tensor_sum(product);
    tensor_backward(loss);
    for (int r = 0; r < 3; ++r) {
        float sum = 0;
        for (int c = 0; c < 3; ++c) sum += x->grad[r * 3 + c];
        assert(nearf(sum, 0.0f, 2e-5f));
    }
    tensor_free_graph(loss);
}

static void test_layer_norm(void) {
    const float xv[] = {1,2,4, 2,4,8};
    const float gv[] = {1,1,1};
    const float bv[] = {0,0,0};
    Tensor *x = tensor_from_array(xv, 2, 3, 1);
    Tensor *g = tensor_from_array(gv, 1, 3, 1);
    Tensor *b = tensor_from_array(bv, 1, 3, 1);
    g->is_protected = b->is_protected = 1;
    Tensor *y = tensor_layer_norm(x, g, b, 1e-5f);
    assert(y && y->rows == 2 && y->cols == 3);
    for (int r = 0; r < 2; ++r) {
        float mean = 0;
        for (int c = 0; c < 3; ++c) mean += y->data[r * 3 + c];
        assert(nearf(mean / 3.0f, 0.0f, 1e-5f));
    }
    Tensor *sq = tensor_mul(y, y);
    Tensor *loss = tensor_sum(sq);
    tensor_backward(loss);
    for (int i = 0; i < 6; ++i) assert(isfinite(x->grad[i]));
    for (int i = 0; i < 3; ++i) assert(isfinite(g->grad[i]) && isfinite(b->grad[i]));
    tensor_free_graph(loss);
    tensor_free(g); tensor_free(b);
}

static void test_layer_norm_finite_difference(void) {
    float xv[] = {0.2f, -1.1f, 2.3f};
    float gv[] = {1.2f, 0.7f, -0.5f};
    float bv[] = {0.1f, -0.2f, 0.3f};
    const float wv[] = {0.8f, -0.6f, 0.4f};
    Tensor *x = tensor_from_array(xv, 1, 3, 1);
    Tensor *g = tensor_from_array(gv, 1, 3, 1);
    Tensor *b = tensor_from_array(bv, 1, 3, 1);
    Tensor *w = tensor_from_array(wv, 1, 3, 0);
    Tensor *y = tensor_layer_norm(x, g, b, 1.0e-5f);
    Tensor *weighted = tensor_mul(y, w);
    Tensor *loss = tensor_sum(weighted);
    tensor_backward(loss);
    const float step = 1.0e-3f;
    for (int i = 0; i < 3; ++i) {
        float original = xv[i];
        xv[i] = original + step;
        float plus = layer_norm_objective(xv, gv, bv, wv);
        xv[i] = original - step;
        float minus = layer_norm_objective(xv, gv, bv, wv);
        xv[i] = original;
        assert(nearf(x->grad[i], (plus - minus) / (2.0f * step), 2.0e-3f));

        original = gv[i];
        gv[i] = original + step;
        plus = layer_norm_objective(xv, gv, bv, wv);
        gv[i] = original - step;
        minus = layer_norm_objective(xv, gv, bv, wv);
        gv[i] = original;
        assert(nearf(g->grad[i], (plus - minus) / (2.0f * step), 2.0e-3f));

        original = bv[i];
        bv[i] = original + step;
        plus = layer_norm_objective(xv, gv, bv, wv);
        bv[i] = original - step;
        minus = layer_norm_objective(xv, gv, bv, wv);
        bv[i] = original;
        assert(nearf(b->grad[i], (plus - minus) / (2.0f * step), 2.0e-3f));
    }
    tensor_free_graph_keep_leaves(loss);
    tensor_free(x); tensor_free(g); tensor_free(b); tensor_free(w);
}

static void test_transpose(void) {
    const float xv[] = {1,2,3,4,5,6};
    Tensor *x = tensor_from_array(xv, 2, 3, 1);
    Tensor *t = tensor_transpose(x);
    assert(t && t->rows == 3 && t->cols == 2);
    assert(nearf(t->data[0], 1, 1e-6f) && nearf(t->data[1], 4, 1e-6f));
    Tensor *loss = tensor_sum(t);
    tensor_backward(loss);
    for (int i = 0; i < 6; ++i) assert(nearf(x->grad[i], 1, 1e-6f));
    tensor_free_graph(loss);
}

static void test_slice_concat(void) {
    const float xv[] = {1,2,3,4,5,6};
    Tensor *x = tensor_from_array(xv, 2, 3, 1);
    Tensor *left = tensor_slice_cols(x, 0, 1);
    Tensor *right = tensor_slice_cols(x, 1, 2);
    Tensor *joined = tensor_concat_cols(left, right);
    assert(joined && joined->rows == 2 && joined->cols == 3);
    for (int i = 0; i < 6; ++i) assert(nearf(joined->data[i], xv[i], 1e-6f));
    Tensor *loss = tensor_sum(joined);
    tensor_backward(loss);
    for (int i = 0; i < 6; ++i) assert(nearf(x->grad[i], 1.0f, 1e-6f));
    tensor_free_graph(loss);
}

static void test_concat_rows_gradient(void) {
    Tensor *a = tensor_from_array((float[]){1,2,3,4}, 2, 2, 1);
    Tensor *b = tensor_from_array((float[]){5,6}, 1, 2, 1);
    Tensor *joined = tensor_concat_rows(a, b);
    assert(joined && joined->rows == 3 && joined->cols == 2);
    assert(nearf(joined->data[0], 1, 1e-6f));
    assert(nearf(joined->data[5], 6, 1e-6f));
    Tensor *loss = tensor_sum(joined);
    tensor_backward(loss);
    for (int i = 0; i < 4; ++i) assert(nearf(a->grad[i], 1.0f, 1e-6f));
    for (int i = 0; i < 2; ++i) assert(nearf(b->grad[i], 1.0f, 1e-6f));
    tensor_free_graph(loss);
}

static void test_adamw(void) {
    Tensor *p = tensor_from_array((float[]){1,-2}, 1, 2, 1);
    AdamW *opt = adamw_create(0.1f, 0.9f, 0.999f, 1e-8f, 0.1f);
    assert(opt && adamw_add_param(opt, p));
    p->grad[0] = 10.0f; p->grad[1] = -10.0f;
    float norm = adamw_clip_grad_norm(opt, 1.0f);
    assert(norm > 14.0f && norm < 15.0f);
    assert(fabsf(p->grad[0]) < 1.0f);
    adamw_step(opt);
    assert(p->data[0] < 1.0f && p->data[1] > -2.0f);
    assert(opt->step == 1);
    adamw_zero_grad(opt);
    assert(nearf(p->grad[0], 0, 1e-7f));
    adamw_free(opt); tensor_free(p);
}

static void test_adamw_decay_excludes_vectors(void) {
    Tensor *matrix = tensor_from_array((float[]){1, 2, 3, 4}, 2, 2, 1);
    Tensor *vector = tensor_from_array((float[]){1, -2}, 1, 2, 1);
    AdamW *opt = adamw_create(0.1f, 0.9f, 0.999f, 1e-8f, 0.1f);
    assert(matrix && vector && opt);
    assert(adamw_add_param(opt, matrix) && adamw_add_param(opt, vector));
    adamw_step(opt); /* gradient nul : seul le weight decay des matrices agit */
    assert(nearf(matrix->data[0], 0.99f, 1e-6f));
    assert(nearf(vector->data[0], 1.0f, 1e-6f));
    assert(nearf(vector->data[1], -2.0f, 1e-6f));
    adamw_free(opt);
    tensor_free(matrix);
    tensor_free(vector);
    assert(adamw_create(NAN, 0.9f, 0.999f, 1e-8f, 0.01f) == NULL);
}

static void test_rng_and_detach(void) {
    Tensor *a = tensor_randn(128, 64, 0, 0);
    assert(a);
    float sum = 0, sum2 = 0;
    for (int i = 0; i < 128 * 64; ++i) { sum += a->data[i]; sum2 += a->data[i] * a->data[i]; }
    float mean = sum / (128.0f * 64.0f);
    float var = sum2 / (128.0f * 64.0f) - mean * mean;
    assert(fabsf(mean) < 0.03f);
    /* He normal variance ~= 2/fan_in = 2/128. */
    assert(var > 0.012f && var < 0.019f);
    Tensor *d = tensor_detach(a);
    assert(d && !d->requires_grad && d->op == OP_LEAF);
    assert(nearf(d->data[100], a->data[100], 1e-7f));
    tensor_free(d); tensor_free(a);
}

int main(void) {
    assert(tensor_zeros(0, 2, 0) == NULL);
    test_embedding_lookup_and_gradient();
    test_cross_entropy_ids();
    test_softmax_causal();
    test_layer_norm();
    test_layer_norm_finite_difference();
    test_transpose();
    test_slice_concat();
    test_concat_rows_gradient();
    test_adamw();
    test_adamw_decay_excludes_vectors();
    test_rng_and_detach();
    puts("ghost native core: 11 test groups passed");
    return 0;
}
