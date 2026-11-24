"""
ghost.tensor
----------------------------------------------------------------
Binding Python (ctypes) autour du moteur C de ghost.

Ce module ne reimplemente aucun calcul : chaque operation appelle
directement la lib C compilee (libghost.so). C'est ce qui rend
G utilisable sur des machines contraintes (telephone ARM
via Termux) sans dependre de PyTorch/TensorFlow.
"""

import ctypes
import os

# ---------------------------------------------------------------
# Chargement de la librairie partagee
# ---------------------------------------------------------------
def _native_library_path():
    import platform
    system = platform.system().lower()
    filename = "ghost.dll" if system == "windows" else ("libghost.dylib" if system == "darwin" else "libghost.so")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


_LIB_PATH = _native_library_path()
if not os.path.isfile(_LIB_PATH):
    raise ImportError(
        f"Moteur natif ghost introuvable : {_LIB_PATH}. "
        "Compile-le avec `python build_native.py` ou `./build.sh`."
    )
_lib = ctypes.CDLL(_LIB_PATH)


class _CTensor(ctypes.Structure):
    """Miroir de la struct Tensor definie dans c/tensor.h.
    L'ordre et les types DOIVENT correspondre exactement au C."""
    pass


_CTensor._fields_ = [
    ("data", ctypes.POINTER(ctypes.c_float)),
    ("grad", ctypes.POINTER(ctypes.c_float)),
    ("rows", ctypes.c_int),
    ("cols", ctypes.c_int),
    ("requires_grad", ctypes.c_int),
    ("op", ctypes.c_int),
    ("parent_a", ctypes.POINTER(_CTensor)),
    ("parent_b", ctypes.POINTER(_CTensor)),
    ("parent_c", ctypes.POINTER(_CTensor)),
    ("visited", ctypes.c_int),
    ("scalar", ctypes.c_float),
    ("is_protected", ctypes.c_int),
    ("indices", ctypes.POINTER(ctypes.c_int)),
    ("index_count", ctypes.c_int),
]

_PTensor = ctypes.POINTER(_CTensor)

# ---------------------------------------------------------------
# Signatures des fonctions C (obligatoire pour eviter les crashs
# silencieux : ctypes suppose "int" par defaut sinon)
# ---------------------------------------------------------------
_lib.tensor_zeros.restype = _PTensor
_lib.tensor_zeros.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]

_lib.tensor_from_array.restype = _PTensor
_lib.tensor_from_array.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int]

_lib.tensor_randn.restype = _PTensor
_lib.tensor_randn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]

_lib.tensor_free.argtypes = [_PTensor]
_lib.tensor_free_graph.argtypes = [_PTensor]
_lib.tensor_free_graph_keep_leaves.argtypes = [_PTensor]

for name in ("tensor_add", "tensor_sub", "tensor_mul", "tensor_matmul"):
    fn = getattr(_lib, name)
    fn.restype = _PTensor
    fn.argtypes = [_PTensor, _PTensor]

for name in ("tensor_relu", "tensor_sigmoid", "tensor_tanh", "tensor_sum"):
    fn = getattr(_lib, name)
    fn.restype = _PTensor
    fn.argtypes = [_PTensor]

_lib.tensor_scale.restype = _PTensor
_lib.tensor_scale.argtypes = [_PTensor, ctypes.c_float]

_lib.tensor_cross_entropy.restype = _PTensor
_lib.tensor_cross_entropy.argtypes = [_PTensor, _PTensor]
_lib.tensor_cross_entropy_ids.restype = _PTensor
_lib.tensor_cross_entropy_ids.argtypes = [_PTensor, ctypes.POINTER(ctypes.c_int), ctypes.c_int]

_lib.tensor_embedding.restype = _PTensor
_lib.tensor_embedding.argtypes = [_PTensor, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
_lib.tensor_transpose.restype = _PTensor
_lib.tensor_transpose.argtypes = [_PTensor]
_lib.tensor_softmax.restype = _PTensor
_lib.tensor_softmax.argtypes = [_PTensor, ctypes.c_int]
_lib.tensor_layer_norm.restype = _PTensor
_lib.tensor_layer_norm.argtypes = [_PTensor, _PTensor, _PTensor, ctypes.c_float]
_lib.tensor_detach.restype = _PTensor
_lib.tensor_detach.argtypes = [_PTensor]
_lib.tensor_slice_cols.restype = _PTensor
_lib.tensor_slice_cols.argtypes = [_PTensor, ctypes.c_int, ctypes.c_int]
_lib.tensor_concat_cols.restype = _PTensor
_lib.tensor_concat_cols.argtypes = [_PTensor, _PTensor]
_lib.tensor_concat_rows.restype = _PTensor
_lib.tensor_concat_rows.argtypes = [_PTensor, _PTensor]

_lib.tensor_backward.argtypes = [_PTensor]
_lib.tensor_zero_grad.argtypes = [_PTensor]

_lib.tensor_print.argtypes = [_PTensor, ctypes.c_char_p]


class Tensor:
    """Wrapper Python confortable autour du Tensor C.

    Exemple :
        a = Tensor([[1.0, 2.0]])
        b = Tensor([[3.0, 4.0]], requires_grad=True)
        c = (a * b).sum()
        c.backward()
        print(b.grad)  # gradient de c par rapport a b
    """

    def __init__(self, values=None, rows=None, cols=None, requires_grad=False, _ptr=None):
        if _ptr is not None:
            if not _ptr:
                raise MemoryError("Le moteur C n'a pas pu creer le tenseur.")
            self._ptr = _ptr
            return

        if values is not None:
            flat, r, c = _flatten(values)
            arr = (ctypes.c_float * len(flat))(*flat)
            self._ptr = _lib.tensor_from_array(arr, r, c, int(requires_grad))
        elif rows is not None and cols is not None:
            self._ptr = _lib.tensor_zeros(rows, cols, int(requires_grad))
        else:
            raise ValueError("Fournir soit `values`, soit `rows` et `cols`.")
        if not self._ptr:
            raise MemoryError("Le moteur C n'a pas pu allouer le tenseur.")

    # -- proprietes --------------------------------------------------

    @property
    def shape(self):
        c = self._ptr.contents
        return (c.rows, c.cols)

    @property
    def data(self):
        c = self._ptr.contents
        n = c.rows * c.cols
        flat = [c.data[i] for i in range(n)]
        return _reshape(flat, c.rows, c.cols)

    @property
    def grad(self):
        c = self._ptr.contents
        if not c.grad:
            return None
        n = c.rows * c.cols
        flat = [c.grad[i] for i in range(n)]
        return _reshape(flat, c.rows, c.cols)

    # -- operations ----------------------------------------------------

    def __add__(self, other):
        return Tensor(_ptr=_lib.tensor_add(self._ptr, other._ptr))

    def __sub__(self, other):
        return Tensor(_ptr=_lib.tensor_sub(self._ptr, other._ptr))

    def __mul__(self, other):
        return Tensor(_ptr=_lib.tensor_mul(self._ptr, other._ptr))

    def matmul(self, other):
        return Tensor(_ptr=_lib.tensor_matmul(self._ptr, other._ptr))

    def relu(self):
        return Tensor(_ptr=_lib.tensor_relu(self._ptr))

    def sigmoid(self):
        return Tensor(_ptr=_lib.tensor_sigmoid(self._ptr))

    def tanh(self):
        return Tensor(_ptr=_lib.tensor_tanh(self._ptr))

    def sum(self):
        return Tensor(_ptr=_lib.tensor_sum(self._ptr))

    def scale(self, k):
        return Tensor(_ptr=_lib.tensor_scale(self._ptr, ctypes.c_float(k)))

    def cross_entropy(self, target):
        """self = logits BRUTS (pas d'activation avant). target = Tensor one-hot
        de meme forme. Softmax + entropie croisee fusionnes, stable numeriquement."""
        return Tensor(_ptr=_lib.tensor_cross_entropy(self._ptr, target._ptr))

    def cross_entropy_ids(self, target_ids):
        """Entropie croisee avec les classes cibles sous forme d'identifiants.
        Évite d'allouer une matrice one-hot de taille batch x vocabulaire."""
        ids = _validate_ids(target_ids)
        if len(ids) != self.shape[0]:
            raise ValueError("Il faut exactement un id cible par ligne de logits.")
        if any(token < 0 or token >= self.shape[1] for token in ids):
            raise ValueError("Un id cible est hors du nombre de classes des logits.")
        arr = (ctypes.c_int * len(ids))(*ids)
        return Tensor(_ptr=_lib.tensor_cross_entropy_ids(self._ptr, arr, len(ids)))

    def transpose(self):
        return Tensor(_ptr=_lib.tensor_transpose(self._ptr))

    def softmax(self, causal=False):
        """Softmax stable ligne par ligne. causal=True masque les positions futures."""
        return Tensor(_ptr=_lib.tensor_softmax(self._ptr, int(bool(causal))))

    def layer_norm(self, gamma, beta, eps=1e-5):
        """LayerNorm par ligne, avec paramètres gamma et beta entraînables."""
        return Tensor(_ptr=_lib.tensor_layer_norm(
            self._ptr, gamma._ptr, beta._ptr, ctypes.c_float(eps)))

    def detach(self):
        """Copie les données dans une feuille indépendante du graphe autograd."""
        return Tensor(_ptr=_lib.tensor_detach(self._ptr))

    def slice_cols(self, start, count):
        return Tensor(_ptr=_lib.tensor_slice_cols(self._ptr, int(start), int(count)))

    def concat_cols(self, other):
        """Concatène deux matrices par colonnes et conserve l'autograd."""
        return Tensor(_ptr=_lib.tensor_concat_cols(self._ptr, other._ptr))

    def concat_rows(self, other):
        """Concatène deux matrices par lignes et conserve l'autograd."""
        return Tensor(_ptr=_lib.tensor_concat_rows(self._ptr, other._ptr))

    def backward(self):
        _lib.tensor_backward(self._ptr)

    def zero_grad(self):
        _lib.tensor_zero_grad(self._ptr)

    def free_graph(self):
        """Libere ce tenseur ET tout le sous-graphe qui l'a produit.
        Ne pas appeler sur des tenseurs feuilles que tu veux garder
        (poids d'un layer par exemple) : utilise `free()` pour ceux-la."""
        _lib.tensor_free_graph(self._ptr)

    def free_intermediates(self):
        """Libere tous les tenseurs INTERMEDIAIRES du sous-graphe ayant
        produit ce tenseur (dont ce tenseur lui-meme, s'il n'est pas une
        feuille) -- garde les feuilles intactes (poids, biais, entrees).
        A appeler apres usage (et apres backward() si besoin) sur CHAQUE
        tenseur issu d'un forward(), sans quoi la memoire C grossit sans
        limite. Une fois appelee, ce Tensor (et tout intermediaire du
        graphe) ne doit plus etre utilise."""
        _lib.tensor_free_graph_keep_leaves(self._ptr)

    def free(self):
        _lib.tensor_free(self._ptr)

    def set_data(self, values):
        """Ecrase les valeurs du tenseur en place (memoire C partagee).
        Utilise pour synchroniser un reseau cible (DQN) ou charger un
        modele sauvegarde. La forme de `values` doit correspondre exactement."""
        flat, r, c = _flatten(values)
        if (r, c) != self.shape:
            raise ValueError(f"set_data: forme incompatible {self.shape} vs {(r, c)}")
        arr = (ctypes.c_float * len(flat))(*flat)
        ctypes.memmove(self._ptr.contents.data, arr, ctypes.sizeof(arr))

    def protect(self):
        """Marque ce tenseur comme PERSISTANT : free_graph() ne le liberera jamais,
        meme s'il est utilise comme parent dans un graphe de calcul ephemere.
        A utiliser sur tout poids cree hors de nn.Dense (qui le fait deja tout seul) --
        par exemple la table de ghost.seq.Embedding."""
        self._ptr.contents.is_protected = 1

    def flat_data(self):
        """Renvoie toujours une liste Python plate (longueur rows*cols),
        peu importe la forme. Utile pour serialiser/manipuler les poids
        sans se soucier du reshape (contrairement a `.data`)."""
        c = self._ptr.contents
        n = c.rows * c.cols
        return [c.data[i] for i in range(n)]

    def set_flat_data(self, flat):
        """Ecrase les valeurs a partir d'une liste plate (longueur rows*cols)."""
        c = self._ptr.contents
        n = c.rows * c.cols
        if len(flat) != n:
            raise ValueError(f"set_flat_data: attendu {n} valeurs, recu {len(flat)}")
        arr = (ctypes.c_float * n)(*flat)
        ctypes.memmove(c.data, arr, ctypes.sizeof(arr))

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.data})"


@classmethod
def _randn(cls, rows, cols, requires_grad=False, seed=42):
    return cls(_ptr=_lib.tensor_randn(rows, cols, int(requires_grad), seed))


Tensor.randn = _randn


def _flatten(values):
    """Accepte une liste 1D ou 2D et renvoie (liste_plate, rows, cols)."""
    if values is None or len(values) == 0:
        raise ValueError("Un tenseur ne peut pas être vide.")
    if isinstance(values[0], (list, tuple)):
        rows = len(values)
        cols = len(values[0])
        if cols == 0 or any(len(row) != cols for row in values):
            raise ValueError("Les lignes du tenseur doivent être non vides et de même longueur.")
        flat = [float(v) for row in values for v in row]
        return flat, rows, cols
    flat = [float(v) for v in values]
    return flat, 1, len(flat)


def _validate_ids(ids):
    if ids is None:
        raise ValueError("La séquence d'identifiants ne peut pas être vide.")
    result = []
    for value in ids:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("Les identifiants de tokens doivent être des entiers.")
        result.append(value)
    if not result:
        raise ValueError("La séquence d'identifiants ne peut pas être vide.")
    return result


def embedding_lookup(weight, token_ids):
    """Lookup d'embeddings natif : O(nombre_tokens x dimension), sans one-hot."""
    ids = _validate_ids(token_ids)
    if any(token < 0 or token >= weight.shape[0] for token in ids):
        raise ValueError("Un token ID est hors de la table d'embedding.")
    arr = (ctypes.c_int * len(ids))(*ids)
    return Tensor(_ptr=_lib.tensor_embedding(weight._ptr, arr, len(ids)))


def _reshape(flat, rows, cols):
    if rows == 1 and cols == 1:
        return flat[0]
    if rows == 1:
        return flat
    return [flat[i * cols:(i + 1) * cols] for i in range(rows)]
