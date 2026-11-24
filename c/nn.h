#ifndef GHOSTLIB_NN_H
#define GHOSTLIB_NN_H

#if defined(_WIN32) && defined(GHOST_BUILD_DLL)
#define GHOST_API __declspec(dllexport)
#else
#define GHOST_API
#endif

#include "tensor.h"

/* ============================================================
 * Couche dense (fully-connected) : y = activation(x . W + b)
 * ============================================================ */

typedef enum {
    ACT_NONE,
    ACT_RELU,
    ACT_SIGMOID,
    ACT_TANH
} Activation;

typedef struct {
    Tensor *weights; /* in_features x out_features, requires_grad=1 */
    Tensor *bias;    /* 1 x out_features, requires_grad=1 */
    Activation activation;
    int in_features;
    int out_features;
} Dense;

GHOST_API Dense *dense_create(int in_features, int out_features, Activation activation, unsigned int seed);
GHOST_API void dense_free(Dense *layer); /* frees weights/bias too */

/* Forward pass. Input x must be (batch x in_features). Returns a new Tensor
 * that is part of the autograd graph (call tensor_backward on the loss, then
 * read layer->weights->grad / layer->bias->grad). */
GHOST_API Tensor *dense_forward(Dense *layer, Tensor *x);

/* ============================================================
 * SGD optimizer
 * ============================================================ */

typedef struct {
    Tensor **params;   /* array of pointers to parameter tensors (weights/biases) */
    int n_params;
    float lr;
} SGD;

GHOST_API SGD *sgd_create(float lr);
GHOST_API void sgd_free(SGD *opt);
/* Registers a parameter tensor to be updated by this optimizer. */
GHOST_API void sgd_add_param(SGD *opt, Tensor *param);
/* Applies param.data -= lr * param.grad for every registered parameter. */
GHOST_API void sgd_step(SGD *opt);
/* Zeros the .grad of every registered parameter (call before each backward). */
GHOST_API void sgd_zero_grad(SGD *opt);

/* AdamW : optimiseur adapte a l'entraînement des Transformers. */
typedef struct {
    Tensor **params;
    float **m;
    float **v;
    int n_params;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    unsigned long long step;
} AdamW;

GHOST_API AdamW *adamw_create(float lr, float beta1, float beta2, float eps, float weight_decay);
GHOST_API int adamw_add_param(AdamW *opt, Tensor *param);
GHOST_API void adamw_step(AdamW *opt);
GHOST_API void adamw_zero_grad(AdamW *opt);
GHOST_API void adamw_free(AdamW *opt);
GHOST_API float adamw_clip_grad_norm(AdamW *opt, float max_norm);

/* ============================================================
 * Pertes (loss functions) -- retournent un tenseur scalaire (1x1)
 * ============================================================ */

/* Mean Squared Error : mean((pred - target)^2) */
GHOST_API Tensor *loss_mse(Tensor *pred, Tensor *target);

#endif /* GHOSTLIB_NN_H */
