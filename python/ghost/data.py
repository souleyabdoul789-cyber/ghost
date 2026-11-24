"""ghost.data — préparation reproductible de corpus, sans charger tout le corpus en RAM.

Les documents conservent leur provenance et leur licence. La séparation train/val/test
se fait au niveau document, après déduplication, avant l'entraînement du tokenizer.
Les shards de tokens sont des entiers uint32 little-endian et sont lus par mmap.
"""
from __future__ import annotations

from array import array
from bisect import bisect_right
from dataclasses import asdict, dataclass, field
import csv
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import struct
import sys
import unicodedata
from typing import Iterable, Iterator, Optional


@dataclass(frozen=True)
class CorpusDocument:
    text: str
    source: str
    title: str = ""
    license: str = "unknown"
    language: str = "und"
    metadata: dict = field(default_factory=dict)
    sha256: str = ""


@dataclass(frozen=True)
class CorpusBuildReport:
    input_documents: int
    kept_documents: int
    exact_duplicates: int
    near_duplicates: int
    empty_documents: int
    split_counts: dict
    output_dir: str


def normalize_text(text: str, *, unicode_form: str = "NFC") -> str:
    """Nettoyage non destructif : conserve ponctuation, code, accents et paragraphes."""
    if not isinstance(text, str):
        raise TypeError("Le contenu du document doit être une chaîne Unicode.")
    text = unicodedata.normalize(unicode_form, text.replace("\r\n", "\n").replace("\r", "\n"))
    text = text.replace("\x00", "")
    lines = [line.rstrip() for line in text.split("\n")]
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{4,}", "\n\n\n", normalized)
    return normalized.strip()


def _word_shingles(text: str, n: int = 3) -> list[str]:
    words = re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)
    if not words:
        return []
    if len(words) < n:
        return [" ".join(words)]
    return [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]


def simhash64(text: str) -> int:
    """Empreinte SimHash 64 bits légère pour repérer les quasi-doublons textuels."""
    shingles = _word_shingles(text)
    if not shingles:
        return 0
    weights = [0] * 64
    for shingle in shingles:
        value = int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "little")
        for bit in range(64):
            weights[bit] += 1 if (value >> bit) & 1 else -1
    result = 0
    for bit, score in enumerate(weights):
        if score >= 0:
            result |= 1 << bit
    return result


def hamming_distance64(a: int, b: int) -> int:
    return (int(a) ^ int(b)).bit_count()


def detect_language(text: str) -> str:
    """Heuristique facultative, jamais utilisée comme preuve de langue."""
    sample = " " + " ".join(re.findall(r"[\wÀ-ÿ]+", text.casefold())[:300]) + " "
    fr = sum(sample.count(" " + w + " ") for w in ("le", "la", "les", "des", "une", "est", "avec", "pour", "dans"))
    en = sum(sample.count(" " + w + " ") for w in ("the", "and", "is", "with", "for", "this", "that", "from"))
    if fr >= 2 and fr > en:
        return "fr"
    if en >= 2 and en > fr:
        return "en"
    return "und"


class DatasetBuilder:
    """Ingestion TXT/Markdown/JSONL/JSON/CSV et constitution de splits stables."""

    TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst"}
    STRUCTURED_EXTENSIONS = {".jsonl", ".ndjson", ".json", ".csv"}

    def __init__(self, *, seed: int = 42, val_fraction: float = 0.05,
                 test_fraction: float = 0.05, near_duplicate_threshold: int = 3,
                 detect_lang: bool = True):
        if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
            raise ValueError("Les fractions val/test doivent être positives et leur somme < 1.")
        if not 0 <= near_duplicate_threshold <= 8:
            raise ValueError("near_duplicate_threshold doit être entre 0 et 8.")
        self.seed = int(seed)
        self.val_fraction = float(val_fraction)
        self.test_fraction = float(test_fraction)
        self.near_duplicate_threshold = int(near_duplicate_threshold)
        self.detect_lang = bool(detect_lang)

    @staticmethod
    def _paths(inputs: Iterable[str | os.PathLike]) -> list[Path]:
        result = []
        for item in inputs:
            path = Path(item).expanduser()
            if path.is_dir():
                result.extend(p for p in path.rglob("*") if p.is_file())
            elif path.is_file():
                result.append(path)
            else:
                raise FileNotFoundError(path)
        return sorted(set(p.resolve() for p in result), key=lambda p: str(p).lower())

    def _record(self, text, source, title="", license="unknown", language="", metadata=None):
        cleaned = normalize_text(str(text))
        if not cleaned:
            return None
        detected = (language or detect_language(cleaned)) if self.detect_lang else (language or "und")
        return CorpusDocument(
            text=cleaned, source=str(source), title=str(title or Path(str(source)).name),
            license=str(license or "unknown"),
            language=str(detected),
            metadata=dict(metadata or {}),
            sha256=hashlib.sha256(cleaned.encode("utf-8")).hexdigest(),
        )

    def _iter_file(self, path: Path) -> Iterator[CorpusDocument]:
        ext = path.suffix.lower()
        if ext in self.TEXT_EXTENSIONS:
            # Lecture par blocs de caractères : limite l'usage RAM même pour un
            # fichier dépourvu de lignes vides ou contenant de très longs paragraphes.
            limit = 64 * 1024
            buffer = ""
            index = 0
            with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
                while True:
                    chunk = stream.read(limit)
                    if not chunk:
                        break
                    buffer += chunk
                    while len(buffer) >= limit:
                        boundary = buffer.rfind("\n", 0, limit)
                        if boundary <= 0:
                            boundary = limit
                        else:
                            boundary += 1
                        part, buffer = buffer[:boundary], buffer[boundary:]
                        doc = self._record(part, f"{path}#part={index}", title=path.name)
                        if doc:
                            yield doc
                        index += 1
            if buffer:
                doc = self._record(buffer, f"{path}#part={index}", title=path.name)
                if doc:
                    yield doc
            return

        if ext in {".jsonl", ".ndjson"}:
            with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"JSONL invalide {path}:{line_number}: {exc}") from exc
                    if isinstance(item, str):
                        item = {"text": item}
                    if not isinstance(item, dict):
                        continue
                    body = item.get("text", item.get("content", item.get("body", "")))
                    doc = self._record(body, item.get("source", f"{path}:{line_number}"),
                                       item.get("title", ""), item.get("license", "unknown"),
                                       item.get("language", ""), item.get("metadata", {}))
                    if doc: yield doc
            return

        if ext == ".json":
            with path.open("r", encoding="utf-8-sig") as stream:
                payload = json.load(stream)
            items = payload if isinstance(payload, list) else [payload]
            for index, item in enumerate(items):
                if isinstance(item, str): item = {"text": item}
                if not isinstance(item, dict): continue
                body = item.get("text", item.get("content", item.get("body", "")))
                doc = self._record(body, item.get("source", f"{path}#{index}"),
                                   item.get("title", ""), item.get("license", "unknown"),
                                   item.get("language", ""), item.get("metadata", {}))
                if doc: yield doc
            return

        if ext == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if not reader.fieldnames or not ({"text", "content", "body"} & set(reader.fieldnames)):
                    raise ValueError(f"CSV {path} doit contenir une colonne text, content ou body.")
                for index, item in enumerate(reader, 2):
                    body = item.get("text", item.get("content", item.get("body", "")))
                    doc = self._record(body, item.get("source", f"{path}:{index}"),
                                       item.get("title", ""), item.get("license", "unknown"),
                                       item.get("language", ""), item)
                    if doc: yield doc
            return

        # Ignore explicit binary/non-corpus files; fail on unknown text-like extensions.
        if ext in self.TEXT_EXTENSIONS | self.STRUCTURED_EXTENSIONS:
            return

    @staticmethod
    def _split_for(digest: str, seed: int, val_fraction: float, test_fraction: float) -> str:
        key = f"{seed}:{digest}".encode("ascii")
        value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / float(1 << 64)
        if value < test_fraction:
            return "test"
        if value < test_fraction + val_fraction:
            return "validation"
        return "train"

    def build(self, inputs: Iterable[str | os.PathLike], output_dir: str | os.PathLike) -> CorpusBuildReport:
        paths = self._paths(inputs)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        split_paths = {name: output / f"{name}.jsonl" for name in ("train", "validation", "test")}
        counts = {name: 0 for name in split_paths}
        exact_seen: set[str] = set()
        signatures: dict[int, int] = {}
        buckets: dict[int, dict[int, set[int]]] = {i: {} for i in range(4)}
        totals = {"input": 0, "exact": 0, "near": 0, "empty": 0, "kept": 0}

        with open(output / "manifest.jsonl", "w", encoding="utf-8") as manifest, \
             split_paths["train"].open("w", encoding="utf-8") as train_out, \
             split_paths["validation"].open("w", encoding="utf-8") as val_out, \
             split_paths["test"].open("w", encoding="utf-8") as test_out:
            streams = {"train": train_out, "validation": val_out, "test": test_out}
            for path in paths:
                for doc in self._iter_file(path):
                    totals["input"] += 1
                    if not doc.text:
                        totals["empty"] += 1
                        continue
                    if doc.sha256 in exact_seen:
                        totals["exact"] += 1
                        continue
                    exact_seen.add(doc.sha256)
                    signature = simhash64(doc.text)
                    candidate_ids = set()
                    for part in range(4):
                        candidate_ids.update(buckets[part].get((signature >> (part * 16)) & 0xFFFF, set()))
                    if self.near_duplicate_threshold:
                        near = any(hamming_distance64(signature, signatures[candidate]) <= self.near_duplicate_threshold
                                   for candidate in candidate_ids)
                        if near:
                            totals["near"] += 1
                            continue
                    doc_id = len(signatures)
                    signatures[doc_id] = signature
                    for part in range(4):
                        buckets[part].setdefault((signature >> (part * 16)) & 0xFFFF, set()).add(doc_id)
                    split = self._split_for(doc.sha256, self.seed, self.val_fraction, self.test_fraction)
                    payload = asdict(doc)
                    payload["split"] = split
                    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                    streams[split].write(line + "\n")
                    manifest.write(line + "\n")
                    counts[split] += 1
                    totals["kept"] += 1

        report = CorpusBuildReport(
            input_documents=totals["input"], kept_documents=totals["kept"],
            exact_duplicates=totals["exact"], near_duplicates=totals["near"],
            empty_documents=totals["empty"], split_counts=counts, output_dir=str(output.resolve()),
        )
        meta = {"format": "ghost-corpus-v1", "seed": self.seed,
                "val_fraction": self.val_fraction, "test_fraction": self.test_fraction,
                "near_duplicate_threshold": self.near_duplicate_threshold,
                "report": asdict(report)}
        temp = output / "dataset.json.tmp"
        temp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, output / "dataset.json")
        return report


def iter_split(dataset_dir: str | os.PathLike, split: str = "train") -> Iterator[CorpusDocument]:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split doit être train, validation ou test.")
    path = Path(dataset_dir) / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip(): continue
            item = json.loads(line)
            yield CorpusDocument(
                text=item["text"], source=item["source"], title=item.get("title", ""),
                license=item.get("license", "unknown"), language=item.get("language", "und"),
                metadata=item.get("metadata", {}), sha256=item.get("sha256", ""),
            )


def write_token_shards(documents: Iterable[CorpusDocument], tokenizer, output_dir: str | os.PathLike,
                       *, eos_id: Optional[int] = None, tokens_per_shard: int = 1_000_000) -> dict:
    """Encode en shards uint32 avec EOS inter-document ; mémoire bornée par un shard."""
    if tokens_per_shard < 2:
        raise ValueError("tokens_per_shard doit être >= 2.")
    if eos_id is not None and (isinstance(eos_id, bool) or not 0 <= int(eos_id) < tokenizer.vocab_size):
        raise ValueError("eos_id hors vocabulaire.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    buffer = array("I")
    total = 0
    doc_count = 0

    def flush():
        nonlocal buffer
        if not buffer:
            return
        if sys.byteorder != "little":
            buffer.byteswap()
        path = output / f"tokens-{len(shard_paths):05d}.bin"
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            stream.write(buffer.tobytes())
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        shard_paths.append({"file": path.name, "tokens": len(buffer)})
        buffer = array("I")

    for document in documents:
        ids = tokenizer.encode(document.text)
        if eos_id is not None:
            ids.append(int(eos_id))
        for token in ids:
            if token < 0 or token > 0xFFFFFFFF:
                raise ValueError("Token ID non représentable en uint32.")
            buffer.append(token)
            total += 1
            if len(buffer) >= tokens_per_shard:
                flush()
        doc_count += 1
    flush()
    manifest = {"format": "ghost-token-shards-v1", "dtype": "uint32-le",
                "total_tokens": total, "documents": doc_count, "shards": shard_paths}
    temp = output / "tokens.json.tmp"
    temp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temp, output / "tokens.json")
    return manifest


class TokenShardDataset:
    """Flux mmap de tokens : lecture de fenêtres sans charger les shards en RAM."""
    def __init__(self, directory: str | os.PathLike):
        self.directory = Path(directory)
        manifest_path = self.directory / "tokens.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "ghost-token-shards-v1" or manifest.get("dtype") != "uint32-le":
            raise ValueError("Manifest de shards Ghost non reconnu.")
        self._files = []
        self._maps = []
        self._starts = []
        self._counts = []
        self.total_tokens = 0
        try:
            for item in manifest.get("shards", []):
                path = self.directory / item["file"]
                handle = path.open("rb")
                size = os.fstat(handle.fileno()).st_size
                if size % 4:
                    handle.close()
                    raise ValueError(f"Shard tronqué (taille non multiple de 4) : {path}")
                count = size // 4
                if count == 0:
                    handle.close()
                    continue
                mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
                self._starts.append(self.total_tokens)
                self._counts.append(count)
                self.total_tokens += count
                self._files.append(handle)
                self._maps.append(mapping)
            if self.total_tokens != int(manifest.get("total_tokens", -1)):
                raise ValueError("Le total du manifest ne correspond pas aux shards.")
        except Exception:
            self.close()
            raise

    def __len__(self):
        return self.total_tokens

    def read_window(self, start: int, length: int) -> list[int]:
        if isinstance(start, bool) or isinstance(length, bool) or start < 0 or length < 1 or start + length > self.total_tokens:
            raise IndexError("Fenêtre de tokens hors limites.")
        result = []
        shard_index = bisect_right(self._starts, start) - 1
        cursor = start
        remaining = length
        while remaining:
            local = cursor - self._starts[shard_index]
            take = min(remaining, self._counts[shard_index] - local)
            raw = self._maps[shard_index]
            values = struct.unpack_from("<" + "I" * take, raw, local * 4)
            result.extend(values)
            cursor += take
            remaining -= take
            shard_index += 1 if remaining else 0
        return list(result)

    def sample_window(self, length: int, rng=None) -> list[int]:
        import random
        rng = rng or random
        if length < 1 or length > self.total_tokens:
            raise ValueError("Longueur de fenêtre incompatible avec le nombre de tokens.")
        return self.read_window(rng.randint(0, self.total_tokens - length), length)

    def close(self):
        for mapping in getattr(self, "_maps", []):
            try: mapping.close()
            except Exception: pass
        for handle in getattr(self, "_files", []):
            try: handle.close()
            except Exception: pass
        self._maps = []
        self._files = []

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, traceback): self.close()


@dataclass(frozen=True)
class ImageTextRecord:
    image_path: str
    text: str
    source: str
    license: str = "unknown"
    language: str = "und"
    image_sha256: str = ""
    split: str = "train"


@dataclass(frozen=True)
class ImageTextBuildReport:
    input_pairs: int
    kept_pairs: int
    exact_duplicates: int
    missing_images: int
    split_counts: dict
    output_dir: str


class ImageTextDatasetBuilder:
    """Construit un corpus image-texte JSONL sans déplacer ni modifier les images.

    Manifeste d'entrée : une ligne JSON par paire avec `image` et `text` (ou
    `caption`), plus `source`, `license` et `language` facultatifs. Les chemins
    sont résolus sous image_root et les traversées de répertoire sont refusées.
    Les splits sont groupés par SHA-256 de l'image pour éviter qu'une même image
    apparaisse à la fois dans train et validation/test.
    """

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

    def __init__(self, image_root: str | os.PathLike, *, seed=42,
                 val_fraction=0.05, test_fraction=0.05):
        if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
            raise ValueError("Les fractions val/test doivent être >= 0 et leur somme < 1.")
        self.image_root = Path(image_root).expanduser().resolve()
        if not self.image_root.is_dir():
            raise NotADirectoryError(self.image_root)
        self.seed = int(seed)
        self.val_fraction = float(val_fraction)
        self.test_fraction = float(test_fraction)

    def build(self, manifest_path: str | os.PathLike, output_dir: str | os.PathLike) -> ImageTextBuildReport:
        manifest_path = Path(manifest_path).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        split_files = {name: (output / f"{name}.jsonl").open("w", encoding="utf-8")
                       for name in ("train", "validation", "test")}
        counts = {name: 0 for name in split_files}
        seen = set()
        image_splits = {}
        totals = {"input": 0, "duplicates": 0, "missing": 0, "kept": 0}
        try:
            with manifest_path.open("r", encoding="utf-8-sig") as source:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    totals["input"] += 1
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Manifeste image-texte invalide ligne {line_number}: {exc}") from exc
                    if not isinstance(item, dict):
                        raise ValueError(f"Ligne {line_number}: chaque entrée doit être un objet JSON.")
                    image_value = item.get("image", item.get("image_path"))
                    caption = normalize_text(str(item.get("text", item.get("caption", ""))))
                    if not image_value or not caption:
                        raise ValueError(f"Ligne {line_number}: `image` et `text`/`caption` sont obligatoires.")
                    candidate = (self.image_root / str(image_value)).resolve()
                    try:
                        candidate.relative_to(self.image_root)
                    except ValueError as exc:
                        raise ValueError(f"Ligne {line_number}: chemin image hors du répertoire autorisé.") from exc
                    if not candidate.is_file() or candidate.suffix.lower() not in self.IMAGE_EXTENSIONS:
                        totals["missing"] += 1
                        continue
                    image_hash = hashlib.sha256()
                    with candidate.open("rb") as image_stream:
                        for chunk in iter(lambda: image_stream.read(1024 * 1024), b""):
                            image_hash.update(chunk)
                    digest = image_hash.hexdigest()
                    key = hashlib.sha256((digest + "\0" + caption).encode("utf-8")).hexdigest()
                    if key in seen:
                        totals["duplicates"] += 1
                        continue
                    seen.add(key)
                    # Toutes les légendes d'une même image restent dans le même split.
                    if digest not in image_splits:
                        image_splits[digest] = DatasetBuilder._split_for(
                            digest, self.seed, self.val_fraction, self.test_fraction)
                    split = image_splits[digest]
                    record = ImageTextRecord(
                        image_path=str(candidate), text=caption,
                        source=str(item.get("source", manifest_path.name + f":{line_number}")),
                        license=str(item.get("license") or "unknown"),
                        language=str(item.get("language") or detect_language(caption)),
                        image_sha256=digest, split=split,
                    )
                    payload = asdict(record)
                    line_out = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                    split_files[split].write(line_out + "\n")
                    counts[split] += 1
                    totals["kept"] += 1
        finally:
            for stream in split_files.values():
                stream.close()

        report = ImageTextBuildReport(
            input_pairs=totals["input"], kept_pairs=totals["kept"],
            exact_duplicates=totals["duplicates"], missing_images=totals["missing"],
            split_counts=counts, output_dir=str(output.resolve()),
        )
        meta = {"format": "ghost-image-text-v1", "seed": self.seed,
                "val_fraction": self.val_fraction, "test_fraction": self.test_fraction,
                "report": asdict(report)}
        temporary = output / "dataset.json.tmp"
        temporary.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, output / "dataset.json")
        return report


def iter_image_split(dataset_dir: str | os.PathLike, split: str = "train") -> Iterator[ImageTextRecord]:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split doit être train, validation ou test.")
    path = Path(dataset_dir) / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            record = ImageTextRecord(**item)
            if not Path(record.image_path).is_file():
                raise FileNotFoundError(f"Image absente à la ligne {line_number}: {record.image_path}")
            yield record


@dataclass(frozen=True)
class ImageInstructionExample:
    image_path: str
    prompt: str
    completion: str
    source: str
    license: str = "unknown"
    language: str = "und"
    image_sha256: str = ""
    split: str = "train"


@dataclass(frozen=True)
class ImageInstructionBuildReport:
    input_examples: int
    kept_examples: int
    duplicates: int
    missing_images: int
    split_counts: dict
    output_dir: str


class ImageInstructionDatasetBuilder:
    """Prépare les exemples multimodaux image+prompt→completion.

    Le manifeste JSONL contient `image`, `prompt`, `completion` (ou `response`),
    `source` et `license`. Les chemins restent sous image_root ; les exemples
    associés à une même image sont regroupés dans le même split.
    """
    def __init__(self, image_root: str | os.PathLike, *, seed=42,
                 val_fraction=0.05, test_fraction=0.05):
        if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
            raise ValueError("Les fractions val/test doivent être >= 0 et leur somme < 1.")
        self.image_root = Path(image_root).expanduser().resolve()
        if not self.image_root.is_dir():
            raise NotADirectoryError(self.image_root)
        self.seed = int(seed)
        self.val_fraction = float(val_fraction)
        self.test_fraction = float(test_fraction)

    def build(self, manifest_path: str | os.PathLike, output_dir: str | os.PathLike) -> ImageInstructionBuildReport:
        source_path = Path(manifest_path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        streams = {name: (output / f"{name}.jsonl").open("w", encoding="utf-8")
                   for name in ("train", "validation", "test")}
        counts = {name: 0 for name in streams}
        seen = set()
        image_splits = {}
        totals = {"input": 0, "duplicates": 0, "missing": 0, "kept": 0}
        try:
            with source_path.open("r", encoding="utf-8-sig") as source:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    totals["input"] += 1
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Manifeste image-instruction invalide ligne {line_number}: {exc}") from exc
                    if not isinstance(item, dict):
                        raise ValueError(f"Ligne {line_number}: objet JSON attendu.")
                    image_value = item.get("image", item.get("image_path"))
                    prompt = normalize_text(str(item.get("prompt", "")))
                    completion = normalize_text(str(item.get("completion", item.get("response", ""))))
                    if not image_value or not prompt or not completion:
                        raise ValueError(f"Ligne {line_number}: image, prompt et completion sont obligatoires.")
                    candidate = (self.image_root / str(image_value)).resolve()
                    try:
                        candidate.relative_to(self.image_root)
                    except ValueError as exc:
                        raise ValueError(f"Ligne {line_number}: chemin image hors du répertoire autorisé.") from exc
                    if not candidate.is_file() or candidate.suffix.lower() not in ImageTextDatasetBuilder.IMAGE_EXTENSIONS:
                        totals["missing"] += 1
                        continue
                    image_hash = hashlib.sha256()
                    with candidate.open("rb") as image_stream:
                        for chunk in iter(lambda: image_stream.read(1024 * 1024), b""):
                            image_hash.update(chunk)
                    image_digest = image_hash.hexdigest()
                    pair_digest = hashlib.sha256(
                        (image_digest + "\0" + prompt + "\0" + completion).encode("utf-8")
                    ).hexdigest()
                    if pair_digest in seen:
                        totals["duplicates"] += 1
                        continue
                    seen.add(pair_digest)
                    if image_digest not in image_splits:
                        image_splits[image_digest] = DatasetBuilder._split_for(
                            image_digest, self.seed, self.val_fraction, self.test_fraction)
                    example = ImageInstructionExample(
                        image_path=str(candidate), prompt=prompt, completion=completion,
                        source=str(item.get("source", source_path.name + f":{line_number}")),
                        license=str(item.get("license") or "unknown"),
                        language=str(item.get("language") or detect_language(prompt + "\n" + completion)),
                        image_sha256=image_digest, split=image_splits[image_digest],
                    )
                    streams[example.split].write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
                    counts[example.split] += 1
                    totals["kept"] += 1
        finally:
            for stream in streams.values():
                stream.close()
        report = ImageInstructionBuildReport(
            input_examples=totals["input"], kept_examples=totals["kept"],
            duplicates=totals["duplicates"], missing_images=totals["missing"],
            split_counts=counts, output_dir=str(output.resolve()),
        )
        temporary = output / "dataset.json.tmp"
        temporary.write_text(json.dumps({
            "format": "ghost-image-instruction-v1", "seed": self.seed,
            "val_fraction": self.val_fraction, "test_fraction": self.test_fraction,
            "report": asdict(report),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, output / "dataset.json")
        return report


def iter_image_instruction_split(dataset_dir: str | os.PathLike, split: str = "train") -> Iterator[ImageInstructionExample]:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split doit être train, validation ou test.")
    path = Path(dataset_dir) / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            example = ImageInstructionExample(**json.loads(line))
            if not Path(example.image_path).is_file():
                raise FileNotFoundError(f"Image absente à la ligne {line_number}: {example.image_path}")
            yield example


@dataclass(frozen=True)
class InstructionExample:
    prompt: str
    completion: str
    source: str
    license: str = "unknown"
    split: str = "train"


@dataclass(frozen=True)
class InstructionBuildReport:
    input_examples: int
    kept_examples: int
    duplicates: int
    split_counts: dict
    output_dir: str


class InstructionDatasetBuilder:
    """Prépare un jeu prompt→completion pour le fine-tuning supervisé (SFT)."""

    def __init__(self, *, seed=42, val_fraction=0.05, test_fraction=0.05):
        if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
            raise ValueError("Les fractions val/test doivent être >= 0 et leur somme < 1.")
        self.seed = int(seed)
        self.val_fraction = float(val_fraction)
        self.test_fraction = float(test_fraction)

    def build(self, manifest_path: str | os.PathLike, output_dir: str | os.PathLike) -> InstructionBuildReport:
        source_path = Path(manifest_path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        streams = {name: (output / f"{name}.jsonl").open("w", encoding="utf-8")
                   for name in ("train", "validation", "test")}
        counts = {name: 0 for name in streams}
        seen = set()
        input_count = duplicates = kept = 0
        try:
            with source_path.open("r", encoding="utf-8-sig") as source:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    input_count += 1
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        raise ValueError(f"Ligne {line_number}: objet JSON attendu.")
                    prompt = normalize_text(str(item.get("prompt", "")))
                    completion = normalize_text(str(item.get("completion", item.get("response", ""))))
                    if not prompt or not completion:
                        raise ValueError(f"Ligne {line_number}: prompt et completion obligatoires.")
                    digest = hashlib.sha256((prompt + "\0" + completion).encode("utf-8")).hexdigest()
                    if digest in seen:
                        duplicates += 1
                        continue
                    seen.add(digest)
                    split = DatasetBuilder._split_for(digest, self.seed, self.val_fraction, self.test_fraction)
                    record = InstructionExample(
                        prompt=prompt, completion=completion,
                        source=str(item.get("source", source_path.name + f":{line_number}")),
                        license=str(item.get("license") or "unknown"), split=split,
                    )
                    streams[split].write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                    counts[split] += 1
                    kept += 1
        finally:
            for stream in streams.values():
                stream.close()
        report = InstructionBuildReport(input_count, kept, duplicates, counts, str(output.resolve()))
        temporary = output / "dataset.json.tmp"
        temporary.write_text(json.dumps({"format": "ghost-instruction-v1", "seed": self.seed,
                                         "report": asdict(report)}, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, output / "dataset.json")
        return report


def iter_instruction_split(dataset_dir: str | os.PathLike, split: str = "train") -> Iterator[InstructionExample]:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split doit être train, validation ou test.")
    path = Path(dataset_dir) / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield InstructionExample(**json.loads(line))
