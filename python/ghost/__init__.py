"""ghost — moteur IA natif et plateforme de recherche de G-SOCIETY.

L'API publique est volontairement en minuscules : `import ghost`.
"""

from .tensor import Tensor
from . import nn, rl, text, seq, llm, web, memory, agent, data, vision, browser, sandbox, motivation
from .text import Tokenizer

__all__ = [
    "Tensor", "Tokenizer", "nn", "rl", "text", "seq", "llm",
    "web", "memory", "agent", "data", "vision", "browser", "sandbox", "motivation",
]
__version__ = "0.6.2"
