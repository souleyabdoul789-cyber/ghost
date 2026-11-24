"""
ghost.nn
----------------------------------------------------------------
Couches de reseau de neurones et optimiseur, construits sur le
moteur de tenseurs C (voir ghost.tensor).
"""

import ctypes
import math
from .tensor import Tensor, _lib, _PTensor

# ---------------------------------------------------------------
# Activation (doit correspondre a l'enum Activation dans nn.h)
# ---------------------------------------------------------------
ACT_NONE = 0
ACT_RELU = 1
ACT_SIGMOID = 2
ACT_TANH = 3


class _CDense(ctypes.Structure):
    _fields_ = [
        ("weights", _PTensor),
        ("bias", _PTensor),
        ("activation", ctypes.c_int),
        ("in_features", ctypes.c_int),
        ("out_features", ctypes.c_int),
    ]


_PDense = ctypes.POINTER(_CDense)

_lib.dense_create.restype = _PDense
_lib.dense_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
_lib.dense_free.argtypes = [_PDense]
_lib.dense_forward.restype = _PTensor
_lib.dense_forward.argtypes = [_PDense, _PTensor]

_lib.loss_mse.restype = _PTensor
_lib.loss_mse.argtypes = [_PTensor, _PTensor]


class Dense:
    """Couche entierement connectee : y = activation(x . W + b)

    Exemple :
        layer = Dense(4, 8, activation="relu")
        out = layer(x)   # x doit etre un Tensor (batch x 4)
    """

    _ACTS = {"none": ACT_NONE, "relu": ACT_RELU, "sigmoid": ACT_SIGMOID, "tanh": ACT_TANH}

    def __init__(self, in_features, out_features, activation="none", seed=42):
        if activation not in self._ACTS:
            raise ValueError(f"activation doit etre l'une de {list(self._ACTS)}")
        if (isinstance(in_features, bool) or isinstance(out_features, bool) or
                not isinstance(in_features, int) or not isinstance(out_features, int) or
                in_features <= 0 or out_features <= 0):
            raise ValueError("in_features et out_features doivent être des entiers positifs.")
        self._ptr = _lib.dense_create(in_features, out_features, self._ACTS[activation], seed)
        if not self._ptr:
            raise MemoryError("Le moteur C n'a pas pu créer la couche Dense.")
        self.in_features = in_features
        self.out_features = out_features

    @property
    def weights(self):
        return Tensor(_ptr=self._ptr.contents.weights)

    @property
    def bias(self):
        return Tensor(_ptr=self._ptr.contents.bias)

    def parameters(self):
        return [self.weights, self.bias]

    def __call__(self, x: Tensor) -> Tensor:
        return Tensor(_ptr=_lib.dense_forward(self._ptr, x._ptr))

    def free(self):
        if getattr(self, "_ptr", None):
            _lib.dense_free(self._ptr)
            self._ptr = None


class Sequential:
    """Empile plusieurs couches. Exemple :

        model = Sequential([
            Dense(4, 16, activation="relu"),
            Dense(16, 1, activation="sigmoid"),
        ])
        out = model(x)
    """

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def state_dict(self):
        """Renvoie les poids/biais de chaque couche, sous forme de listes Python
        (serialisable en JSON)."""
        return [{"weights": layer.weights.data, "bias": layer.bias.data} for layer in self.layers]

    def load_state_dict(self, state):
        """Recharge des poids/biais precedemment obtenus via state_dict().
        L'architecture (nombre de couches, tailles) doit deja correspondre."""
        if len(state) != len(self.layers):
            raise ValueError("Le nombre de couches du checkpoint ne correspond pas.")
        for layer, s in zip(self.layers, state):
            layer.weights.set_data(s["weights"])
            layer.bias.set_data(s["bias"])

    def free(self):
        """Libère les paramètres natifs de toutes les couches du modèle."""
        for layer in self.layers:
            layer.free()

    def save(self, path):
        import json
        with open(path, "w") as f:
            json.dump(self.state_dict(), f)

    def load(self, path):
        import json
        with open(path) as f:
            state = json.load(f)
        self.load_state_dict(state)


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    if pred.shape != target.shape:
        raise ValueError("mse_loss exige des tenseurs de même forme.")
    return Tensor(_ptr=_lib.loss_mse(pred._ptr, target._ptr))


def cross_entropy_loss(logits: Tensor, target: Tensor) -> Tensor:
    """logits = scores BRUTS du reseau (derniere couche en activation="none" --
    PAS de sigmoid/softmax avant, le softmax est fait DANS cette fonction).
    target = Tensor one-hot de meme forme. A utiliser pour de la classification
    multi-classes (ex: predire le prochain token) -- contrairement a mse_loss,
    gere correctement un vocabulaire avec des classes de frequences differentes."""
    return logits.cross_entropy(target)


def cross_entropy_loss_ids(logits: Tensor, target_ids) -> Tensor:
    """Cross-entropy native avec classes sous forme d'IDs (modèles de langage)."""
    return logits.cross_entropy_ids(target_ids)


def forward_values(model, x_values):
    """Fait un forward SANS garder le graphe (pas de backward prevu) et
    renvoie directement les valeurs Python -- nettoie la memoire C au
    passage (tenseur d'entree + tous les intermediaires). A utiliser pour
    toute inference (ex: choisir une action) ou toute evaluation d'un
    reseau cible, plutot que `model(Tensor(x_values)).data` qui fuit."""
    x = Tensor(x_values)
    out = model(x)
    values = out.data
    out.free_intermediates()
    x.free()
    return values


# ---------------------------------------------------------------
# SGD
# ---------------------------------------------------------------

class _CSGD(ctypes.Structure):
    _fields_ = [
        ("params", ctypes.POINTER(_PTensor)),
        ("n_params", ctypes.c_int),
        ("lr", ctypes.c_float),
    ]


_PSGD = ctypes.POINTER(_CSGD)

_lib.sgd_create.restype = _PSGD
_lib.sgd_create.argtypes = [ctypes.c_float]
_lib.sgd_add_param.argtypes = [_PSGD, _PTensor]
_lib.sgd_step.argtypes = [_PSGD]
_lib.sgd_zero_grad.argtypes = [_PSGD]
_lib.sgd_free.argtypes = [_PSGD]


class _CAdamW(ctypes.Structure):
    _fields_ = [
        ("params", ctypes.POINTER(_PTensor)),
        ("m", ctypes.POINTER(ctypes.POINTER(ctypes.c_float))),
        ("v", ctypes.POINTER(ctypes.POINTER(ctypes.c_float))),
        ("n_params", ctypes.c_int),
        ("lr", ctypes.c_float),
        ("beta1", ctypes.c_float),
        ("beta2", ctypes.c_float),
        ("eps", ctypes.c_float),
        ("weight_decay", ctypes.c_float),
        ("step", ctypes.c_ulonglong),
    ]


_PAdamW = ctypes.POINTER(_CAdamW)
_lib.adamw_create.restype = _PAdamW
_lib.adamw_create.argtypes = [ctypes.c_float] * 5
_lib.adamw_add_param.restype = ctypes.c_int
_lib.adamw_add_param.argtypes = [_PAdamW, _PTensor]
_lib.adamw_step.argtypes = [_PAdamW]
_lib.adamw_zero_grad.argtypes = [_PAdamW]
_lib.adamw_free.argtypes = [_PAdamW]
_lib.adamw_clip_grad_norm.restype = ctypes.c_float
_lib.adamw_clip_grad_norm.argtypes = [_PAdamW, ctypes.c_float]


class AdamW:
    """AdamW natif C avec moments persistants et clipping global optionnel."""

    def __init__(self, parameters, lr=3e-4, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01):
        if len(betas) != 2:
            raise ValueError("betas doit contenir exactement deux valeurs.")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError("betas doivent être dans [0, 1).")
        if not all(math.isfinite(float(value)) for value in (lr, eps, weight_decay, *betas)):
            raise ValueError("Les hyperparamètres AdamW doivent être finis.")
        self._ptr = _lib.adamw_create(
            ctypes.c_float(lr), ctypes.c_float(betas[0]), ctypes.c_float(betas[1]),
            ctypes.c_float(eps), ctypes.c_float(weight_decay))
        if not self._ptr:
            raise ValueError("Hyperparamètres AdamW invalides ou allocation impossible.")
        for parameter in parameters:
            if not _lib.adamw_add_param(self._ptr, parameter._ptr):
                self.free()
                raise MemoryError("Impossible d'enregistrer un paramètre AdamW.")

    def zero_grad(self):
        _lib.adamw_zero_grad(self._ptr)

    def clip_grad_norm(self, max_norm=1.0):
        return float(_lib.adamw_clip_grad_norm(self._ptr, ctypes.c_float(max_norm)))

    def step(self):
        _lib.adamw_step(self._ptr)

    def state_dict(self):
        """Sérialise les moments AdamW afin de reprendre un entraînement."""
        if not self._ptr:
            raise RuntimeError("Optimiseur déjà libéré.")
        state = self._ptr.contents
        parameters = []
        for index in range(state.n_params):
            parameter = state.params[index].contents
            count = parameter.rows * parameter.cols
            parameters.append({
                "shape": [parameter.rows, parameter.cols],
                "m": [float(state.m[index][j]) for j in range(count)],
                "v": [float(state.v[index][j]) for j in range(count)],
            })
        return {
            "format": "ghost-adamw-v1", "step": int(state.step),
            "lr": float(state.lr), "beta1": float(state.beta1),
            "beta2": float(state.beta2), "eps": float(state.eps),
            "weight_decay": float(state.weight_decay), "parameters": parameters,
        }

    def load_state_dict(self, saved):
        """Recharge l'état AdamW après validation de l'architecture et des hyperparamètres."""
        if not self._ptr:
            raise RuntimeError("Optimiseur déjà libéré.")
        state = self._ptr.contents
        if saved.get("format") != "ghost-adamw-v1":
            raise ValueError("Format d'état AdamW non reconnu.")
        expected = (float(state.lr), float(state.beta1), float(state.beta2),
                    float(state.eps), float(state.weight_decay))
        received = tuple(float(saved[key]) for key in ("lr", "beta1", "beta2", "eps", "weight_decay"))
        if any(abs(a - b) > 1e-12 for a, b in zip(expected, received)):
            raise ValueError("Les hyperparamètres de l'état AdamW ne correspondent pas.")
        step = saved.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("Compteur d'étapes AdamW invalide.")
        parameters = saved.get("parameters", [])
        if len(parameters) != state.n_params:
            raise ValueError("Nombre de paramètres AdamW incorrect.")
        staged = []
        for index, item in enumerate(parameters):
            parameter = state.params[index].contents
            count = parameter.rows * parameter.cols
            if item.get("shape") != [parameter.rows, parameter.cols]:
                raise ValueError(f"Forme du paramètre AdamW {index} incompatible.")
            m, v = item.get("m", []), item.get("v", [])
            if len(m) != count or len(v) != count:
                raise ValueError(f"Moments AdamW {index} de longueur incorrecte.")
            if not all(math.isfinite(float(x)) for x in m) or not all(math.isfinite(float(x)) for x in v):
                raise ValueError("L'état AdamW contient une valeur non finie.")
            staged.append((m, v))
        for index, (m, v) in enumerate(staged):
            parameter = state.params[index].contents
            count = parameter.rows * parameter.cols
            for j in range(count):
                state.m[index][j] = float(m[j])
                state.v[index][j] = float(v[j])
        state.step = step

    def free(self):
        if getattr(self, "_ptr", None):
            _lib.adamw_free(self._ptr)
            self._ptr = None


class SGD:
    """Optimiseur simple : param -= lr * param.grad pour chaque parametre enregistre."""

    def __init__(self, parameters, lr=0.01):
        if not math.isfinite(float(lr)) or lr <= 0:
            raise ValueError("lr doit être un nombre fini strictement positif.")
        self._ptr = _lib.sgd_create(ctypes.c_float(lr))
        if not self._ptr:
            raise MemoryError("Le moteur C n'a pas pu créer SGD.")
        for p in parameters:
            _lib.sgd_add_param(self._ptr, p._ptr)

    def step(self):
        _lib.sgd_step(self._ptr)

    def zero_grad(self):
        _lib.sgd_zero_grad(self._ptr)

    def free(self):
        if getattr(self, "_ptr", None):
            _lib.sgd_free(self._ptr)
            self._ptr = None
