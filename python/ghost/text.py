"""Tokenizer BPE UTF-8 byte-level de ghost.

Le tokenizer convertit du texte en identifiants entiers (et inversement).
Il ne chiffre pas les donnees : les IDs sont les symboles que le modele apprend
à prédire. Le vocabulaire de base contient les 256 valeurs d'octet, ce qui
permet de représenter tout texte UTF-8 sans token <unk>.

Cette implémentation privilégie la robustesse et une empreinte mémoire faible.
L'entraînement agrège les morceaux pré-tokenisés identiques afin d'éviter de
stocker chaque occurrence du corpus. Le calcul des fréquences de paires est
recalculé après chaque fusion : c'est fiable pour les petits/moyens vocabulaires,
mais une future version pourra remplacer ce noyau par une file de priorité
incrémentale pour les très grands corpus.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
import heapq
from pathlib import Path
from typing import Iterable, Iterator, Sequence

_PRETOKEN_PATTERN = re.compile(r""" ?\w+| ?[^\s\w]+|\s+""", re.UNICODE)
_FORMAT_VERSION = 2


class Tokenizer:
    """BPE au niveau des octets.

    Exemple::

        tokenizer = Tokenizer()
        tokenizer.train(["le robot marche", "le robot apprend"], vocab_size=320)
        tokenizer.add_special_tokens(["<|bos|>", "<|eos|>"])
        ids = tokenizer.encode("<|bos|>le robot<|eos|>")
        assert tokenizer.decode(ids) == "<|bos|>le robot<|eos|>"
        tokenizer.save("tokenizer.json")

    `train()` repart toujours d'un vocabulaire neuf. Ajouter les tokens spéciaux
    après `train()`, afin que leurs IDs restent après les IDs BPE appris.
    """

    def __init__(self):
        self.merges: list[tuple[int, int]] = []
        self.merge_rank: dict[tuple[int, int], int] = {}
        self.pair_to_new_id: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: dict[str, int] = {}
        self.special_tokens_inv: dict[int, str] = {}
        self.training_stats: dict[str, int] = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    def _reset(self) -> None:
        self.merges.clear()
        self.merge_rank.clear()
        self.pair_to_new_id.clear()
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_tokens.clear()
        self.special_tokens_inv.clear()
        self.training_stats = {}

    def add_special_tokens(self, tokens: Iterable[str]) -> dict[str, int]:
        """Ajoute des IDs réservés (BOS/EOS/PAD/UNK, etc.) après les fusions BPE.

        Retourne la table token -> ID. Les tokens déjà présents ne sont pas
        réattribués. Appeler cette méthode après `train()`.
        """
        for token in tokens:
            if not isinstance(token, str) or not token:
                raise ValueError("Chaque token spécial doit être une chaîne non vide.")
            if token not in self.special_tokens:
                token_id = 256 + len(self.merges) + len(self.special_tokens)
                self.special_tokens[token] = token_id
                self.special_tokens_inv[token_id] = token
        return dict(self.special_tokens)

    def token_to_id(self, token: str) -> int:
        """Retourne l'ID d'un token spécial enregistré."""
        try:
            return self.special_tokens[token]
        except KeyError as exc:
            raise KeyError(f"Token spécial absent du vocabulaire : {token}") from exc

    @staticmethod
    def _pretokenize(text: str) -> list[str]:
        return _PRETOKEN_PATTERN.findall(text)

    @staticmethod
    def _apply_merge(word: Sequence[int], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
        result: list[int] = []
        i = 0
        n = len(word)
        while i < n:
            if i + 1 < n and word[i] == pair[0] and word[i + 1] == pair[1]:
                result.append(new_id)
                i += 2
            else:
                result.append(word[i])
                i += 1
        return tuple(result)

    def train(
        self,
        texts: str | Iterable[str],
        vocab_size: int = 1000,
        *,
        min_frequency: int = 2,
        verbose: bool = False,
    ) -> "Tokenizer":
        """Apprend les fusions BPE sur un itérable de textes.

        Le corpus n'est pas concaténé en une énorme chaîne : seuls les morceaux
        pré-tokenisés uniques et leur fréquence sont conservés en mémoire.
        `vocab_size` inclut les 256 octets de base, mais pas les tokens spéciaux.
        Les égalités de fréquence sont départagées par les IDs pour obtenir des
        fusions reproductibles d'une exécution à l'autre.
        """
        if isinstance(texts, str):
            texts = (texts,)
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 256:
            raise ValueError("vocab_size doit être un entier >= 256.")
        if isinstance(min_frequency, bool) or not isinstance(min_frequency, int) or min_frequency < 1:
            raise ValueError("min_frequency doit être un entier >= 1.")

        self._reset()
        chunk_freq: Counter[str] = Counter()
        document_count = 0
        character_count = 0
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("Chaque élément du corpus doit être une chaîne Unicode.")
            document_count += 1
            character_count += len(text)
            chunk_freq.update(self._pretokenize(text))

        # Chaque pré-token unique devient une entrée pondérée. Les fréquences
        # des paires sont maintenues incrémentalement au lieu de rescanner tout
        # le corpus à chaque fusion. Le tas garde les anciennes entrées ; elles
        # sont écartées à l'extraction si leur fréquence est devenue obsolète.
        word_items = [(tuple(chunk.encode("utf-8")), int(frequency))
                      for chunk, frequency in chunk_freq.items() if chunk]
        words = [word for word, _frequency in word_items]
        frequencies = [frequency for _word, frequency in word_items]
        initial_unique_chunks = len(words)
        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_words: dict[tuple[int, int], set[int]] = defaultdict(set)
        for word_index, (word, frequency) in enumerate(zip(words, frequencies)):
            counts = Counter(zip(word, word[1:]))
            for pair, occurrences in counts.items():
                pair_counts[pair] += occurrences * frequency
                pair_words[pair].add(word_index)

        heap = [(-count, pair[0], pair[1]) for pair, count in pair_counts.items()]
        heapq.heapify(heap)
        merge_limit = vocab_size - 256

        for merge_index in range(merge_limit):
            best_pair = None
            best_frequency = 0
            while heap:
                negative_count, left, right = heapq.heappop(heap)
                pair = (left, right)
                current = pair_counts.get(pair, 0)
                if current == -negative_count and current > 0:
                    best_pair = pair
                    best_frequency = current
                    break
            if best_pair is None or best_frequency < min_frequency:
                break

            new_id = 256 + merge_index
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.merges.append(best_pair)
            self.merge_rank[best_pair] = merge_index
            self.pair_to_new_id[best_pair] = new_id

            affected = sorted(pair_words.get(best_pair, ()))
            for word_index in affected:
                old_word = words[word_index]
                frequency = frequencies[word_index]
                old_counts = Counter(zip(old_word, old_word[1:]))

                for pair, occurrences in old_counts.items():
                    pair_counts[pair] -= occurrences * frequency
                    pair_words[pair].discard(word_index)
                    if pair_counts[pair] <= 0:
                        pair_counts.pop(pair, None)
                        pair_words.pop(pair, None)
                    else:
                        heapq.heappush(heap, (-pair_counts[pair], pair[0], pair[1]))

                new_word = self._apply_merge(old_word, best_pair, new_id)
                words[word_index] = new_word
                new_counts = Counter(zip(new_word, new_word[1:]))
                for pair, occurrences in new_counts.items():
                    pair_counts[pair] += occurrences * frequency
                    pair_words[pair].add(word_index)
                    heapq.heappush(heap, (-pair_counts[pair], pair[0], pair[1]))

            if verbose and (merge_index % 100 == 0 or merge_index == merge_limit - 1):
                print(
                    f"BPE fusion {merge_index + 1}/{merge_limit}: "
                    f"{best_pair} -> {new_id} (fréquence={best_frequency})"
                )

        self.training_stats = {
            "documents": document_count,
            "characters": character_count,
            "unique_chunks": initial_unique_chunks,
            "merges_learned": len(self.merges),
            "vocab_size": self.vocab_size,
        }
        return self

    def train_file(
        self,
        path: str | Path,
        vocab_size: int = 1000,
        *,
        min_frequency: int = 2,
        encoding: str = "utf-8",
        verbose: bool = False,
    ) -> "Tokenizer":
        """Entraîne directement depuis un fichier texte lu ligne par ligne."""
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        with file_path.open("r", encoding=encoding, errors="strict", newline="") as stream:
            return self.train(stream, vocab_size, min_frequency=min_frequency, verbose=verbose)

    @staticmethod
    def _encode_plain(text: str, encoder) -> list[int]:
        ids: list[int] = []
        for chunk in _PRETOKEN_PATTERN.findall(text):
            ids.extend(encoder(chunk))
        return ids

    def _encode_chunk(self, chunk: str) -> list[int]:
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            best_rank = None
            best_pair = None
            for pair in zip(ids, ids[1:]):
                rank = self.merge_rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            ids = list(self._apply_merge(ids, best_pair, self.pair_to_new_id[best_pair]))
        return ids

    def encode(self, text: str, *, recognize_special_tokens: bool = True) -> list[int]:
        """Convertit Unicode en IDs. Les tokens spéciaux enregistrés sont reconnus.

        Tout texte ordinaire est encodé en UTF-8 avec fallback octet : aucun
        caractère inconnu ne provoque une erreur de vocabulaire.
        """
        if not isinstance(text, str):
            raise TypeError("encode() attend une chaîne Unicode.")
        if not text:
            return []
        if not recognize_special_tokens or not self.special_tokens:
            return self._encode_plain(text, self._encode_chunk)

        special_pattern = re.compile(
            "(" + "|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)) + ")"
        )
        result: list[int] = []
        for part in special_pattern.split(text):
            if not part:
                continue
            special_id = self.special_tokens.get(part)
            if special_id is not None:
                result.append(special_id)
            else:
                result.extend(self._encode_plain(part, self._encode_chunk))
        return result

    def encode_batch(self, texts: Iterable[str]) -> list[list[int]]:
        return [self.encode(text) for text in texts]

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = False) -> str:
        """Reconstruit le texte. Les octets sont concaténés avant décodage UTF-8."""
        parts: list[bytes] = []
        buffer = bytearray()
        for raw_id in ids:
            if isinstance(raw_id, bool) or not isinstance(raw_id, int):
                raise TypeError("Les IDs du tokenizer doivent être des entiers.")
            token_id = raw_id
            if token_id in self.special_tokens_inv:
                if buffer:
                    parts.append(bytes(buffer))
                    buffer.clear()
                if not skip_special_tokens:
                    parts.append(self.special_tokens_inv[token_id].encode("utf-8"))
            elif token_id in self.vocab:
                buffer.extend(self.vocab[token_id])
            else:
                raise ValueError(f"ID de token inconnu : {token_id}")
        if buffer:
            parts.append(bytes(buffer))
        return b"".join(parts).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        """Sauvegarde un tokenizer versionné, vérifiable et portable."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": _FORMAT_VERSION,
            "algorithm": "byte_bpe",
            "pretokenizer": "ghost_unicode_word_punctuation_v1",
            "merges": [list(pair) for pair in self.merges],
            "special_tokens": self.special_tokens,
            "training_stats": self.training_stats,
        }
        temporary = destination.with_name(destination.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
            stream.flush()
        temporary.replace(destination)

    def load(self, path: str | Path) -> "Tokenizer":
        """Charge un fichier BPE v1/v2 et vérifie les références de fusion."""
        with Path(path).open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict) or payload.get("algorithm", "byte_bpe") != "byte_bpe":
            raise ValueError("Format de tokenizer non pris en charge.")
        version = payload.get("format_version", 1)
        if version not in (1, 2):
            raise ValueError(f"Version de tokenizer non prise en charge : {version}")

        merges = payload.get("merges")
        if not isinstance(merges, list):
            raise ValueError("La liste des fusions BPE est invalide.")
        self._reset()
        for rank, raw_pair in enumerate(merges):
            if not isinstance(raw_pair, (list, tuple)) or len(raw_pair) != 2:
                raise ValueError(f"Fusion BPE invalide à l'index {rank}.")
            a, b = raw_pair
            max_available_id = 255 + rank
            if any(isinstance(value, bool) or not isinstance(value, int) for value in (a, b)):
                raise ValueError(f"IDs invalides dans la fusion BPE {rank}.")
            if not (0 <= a <= max_available_id and 0 <= b <= max_available_id):
                raise ValueError(f"Fusion BPE {rank} référence un token futur ou invalide.")
            pair = (a, b)
            new_id = 256 + rank
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
            self.merges.append(pair)
            self.merge_rank[pair] = rank
            self.pair_to_new_id[pair] = new_id

        specials = payload.get("special_tokens", {})
        if not isinstance(specials, dict):
            raise ValueError("La table des tokens spéciaux est invalide.")
        base = 256 + len(self.merges)
        expected_ids = set(range(base, base + len(specials)))
        normalized: dict[str, int] = {}
        for token, token_id in specials.items():
            if not isinstance(token, str) or not token:
                raise ValueError("Nom de token spécial invalide.")
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise ValueError("ID de token spécial invalide.")
            normalized[token] = token_id
        if set(normalized.values()) != expected_ids or len(set(normalized.values())) != len(normalized):
            raise ValueError("Les IDs des tokens spéciaux doivent être contigus après les fusions BPE.")
        self.special_tokens = normalized
        self.special_tokens_inv = {value: key for key, value in normalized.items()}
        self.training_stats = payload.get("training_stats", {}) if version >= 2 else {}
        if not isinstance(self.training_stats, dict):
            raise ValueError("Les statistiques du tokenizer sont invalides.")
        return self
