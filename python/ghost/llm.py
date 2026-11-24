"""API publique des modèles de langage de ghost.

Le moteur et l'autograd sont exécutés en C ; ce module expose les
architectures de langage, tandis que ghost.text fournit le tokenizer BPE.
"""

from .seq import Embedding, LayerNorm, TransformerBlock, TransformerLanguageModel
from .text import Tokenizer

# Nom court pour les applications qui souhaitent simplement charger un GPT.
GPTModel = TransformerLanguageModel

__all__ = [
    "Tokenizer",
    "Embedding",
    "LayerNorm",
    "TransformerBlock",
    "TransformerLanguageModel",
    "GPTModel",
]
