"""ghost.vision — encodeur d'image ONNX optionnel et pont entraînable vers le LLM.

Ghost ne contient pas de poids visuels pré-entraînés. On fournit un encodeur ONNX
externe (par ex. un encodeur ViT/CLIP exporté) et Ghost entraîne le projecteur qui
aligne ses caractéristiques sur l'espace d'embeddings du Transformer.
"""
from __future__ import annotations

from dataclasses import dataclass
import io
import math
import os
from pathlib import Path
from typing import Sequence

from .tensor import Tensor
from . import nn


@dataclass(frozen=True)
class ImagePreprocess:
    size: int = 224
    mean: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    std: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

    def __post_init__(self):
        if self.size < 16 or len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("Configuration de prétraitement d'image invalide.")
        if any(float(v) <= 0 for v in self.std):
            raise ValueError("Les écarts-types de normalisation doivent être positifs.")


class OnnxVisionEncoder:
    """Encodeur image CPU ONNX. Dépendances optionnelles : Pillow, NumPy, ONNX Runtime."""

    def __init__(self, model_path, *, input_name=None, output_name=None,
                 preprocess: ImagePreprocess | None = None, threads=2,
                 spatial_layout="nchw"):
        try:
            import numpy as np
            import onnxruntime as ort
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Vision ONNX nécessite les extras : pip install 'ghost-gsociety[vision]'"
            ) from exc
        self._np = np
        self._Image = Image
        self.preprocess = preprocess or ImagePreprocess()
        if spatial_layout not in {"nchw", "nhwc"}:
            raise ValueError("spatial_layout doit être 'nchw' ou 'nhwc'.")
        self.spatial_layout = spatial_layout
        self.model_path = os.fspath(model_path)
        if not Path(self.model_path).is_file():
            raise FileNotFoundError(self.model_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            self.model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if not inputs or not outputs:
            raise ValueError("Le modèle ONNX doit exposer au moins une entrée et une sortie.")
        self.input_name = input_name or inputs[0].name
        self.output_name = output_name or outputs[0].name
        if self.input_name not in {item.name for item in inputs}:
            raise ValueError("input_name absent du graphe ONNX.")
        if self.output_name not in {item.name for item in outputs}:
            raise ValueError("output_name absent du graphe ONNX.")
        self.feature_dim = None

    def preprocess_image(self, image):
        np = self._np
        if isinstance(image, (bytes, bytearray, memoryview)):
            source = self._Image.open(io.BytesIO(bytes(image)))
        else:
            source = self._Image.open(os.fspath(image))
        with source as opened:
            img = opened.convert("RGB")
            width, height = img.size
            scale = self.preprocess.size / min(width, height)
            resized = img.resize((max(self.preprocess.size, round(width * scale)),
                                  max(self.preprocess.size, round(height * scale))),
                                 resample=self._Image.Resampling.BICUBIC)
            left = (resized.width - self.preprocess.size) // 2
            top = (resized.height - self.preprocess.size) // 2
            crop = resized.crop((left, top, left + self.preprocess.size, top + self.preprocess.size))
            pixels = np.asarray(crop, dtype=np.float32) / 255.0
        pixels = (pixels - np.asarray(self.preprocess.mean, dtype=np.float32)) / np.asarray(self.preprocess.std, dtype=np.float32)
        return np.transpose(pixels, (2, 0, 1))[None, ...].astype(np.float32, copy=False)

    def encode_tokens(self, image) -> list[list[float]]:
        """Encode une image en tokens visuels, en préservant les patches si disponibles.

        Sorties ONNX acceptées : [1,D], [1,N,D], [1,C,H,W] ou [1,H,W,C].
        Les tenseurs spatiaux sont convertis en une séquence N x D. Pour une
        sortie ambiguë, choisir/exporter un graphe dont la convention est connue.
        """
        tensor = self.preprocess_image(image)
        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        value = self._np.asarray(outputs[0], dtype=self._np.float32)
        if value.ndim == 0 or value.shape[0] != 1:
            raise ValueError("La sortie ONNX doit conserver une dimension batch de 1.")
        value = value[0]
        if value.ndim == 1:
            rows = value.reshape(1, -1)
        elif value.ndim == 2:
            # [N,D] : séquence de patches/features (CLS inclus si l'encodeur l'émet).
            rows = value
        elif value.ndim == 3:
            if self.spatial_layout == "nchw":
                rows = value.transpose(1, 2, 0).reshape(-1, value.shape[0])
            else:
                rows = value.reshape(-1, value.shape[-1])
        else:
            rows = value.reshape(-1, value.shape[-1])
        if rows.shape[0] == 0 or rows.shape[1] == 0:
            raise ValueError("L'encodeur ONNX a produit une séquence visuelle vide.")
        if not self._np.isfinite(rows).all():
            raise ValueError("L'encodeur ONNX a produit des caractéristiques non finies.")
        self.feature_dim = int(rows.shape[1])
        return [[float(x) for x in row] for row in rows]

    def encode(self, image) -> list[float]:
        """Retourne un embedding global (compatibilité) par moyenne des tokens visuels."""
        rows = self.encode_tokens(image)
        return [sum(row[j] for row in rows) / len(rows) for j in range(len(rows[0]))]


class VisionProjector:
    """Projection entraînable des caractéristiques visuelles vers d_model."""
    def __init__(self, feature_dim: int, d_model: int, seed=917):
        self.feature_dim = int(feature_dim)
        self.d_model = int(d_model)
        if self.feature_dim <= 0 or self.d_model <= 0:
            raise ValueError("feature_dim et d_model doivent être positifs.")
        self.layer = nn.Dense(self.feature_dim, self.d_model, activation="none", seed=seed)

    def __call__(self, features: Sequence[float] | Sequence[Sequence[float]]) -> Tensor:
        values = list(features)
        if not values:
            raise ValueError("Au moins un vecteur visuel est nécessaire.")
        if isinstance(values[0], (list, tuple)):
            rows = [[float(x) for x in row] for row in values]
            if any(len(row) != self.feature_dim for row in rows):
                raise ValueError("Dimension des patches visuels incompatible.")
        else:
            if len(values) != self.feature_dim:
                raise ValueError("Dimension du vecteur visuel incompatible.")
            rows = [[float(x) for x in values]]
        if any(not math.isfinite(x) for row in rows for x in row):
            raise ValueError("Caractéristiques visuelles non finies.")
        return self.layer(Tensor(rows))

    def parameters(self):
        return self.layer.parameters()

    def state_dict(self):
        return {
            "format": "ghost-vision-projector-v1",
            "feature_dim": self.feature_dim,
            "d_model": self.d_model,
            "weights": self.layer.weights.flat_data(),
            "bias": self.layer.bias.flat_data(),
        }

    def load_state_dict(self, state):
        if state.get("format") != "ghost-vision-projector-v1":
            raise ValueError("Format de projecteur visuel inconnu.")
        if state.get("feature_dim") != self.feature_dim or state.get("d_model") != self.d_model:
            raise ValueError("Dimensions du projecteur visuel incompatibles.")
        self.layer.weights.set_flat_data(state["weights"])
        self.layer.bias.set_flat_data(state["bias"])

    def free(self):
        self.layer.free()


class VisionLanguageModel:
    """Pont vision→LLM avec séquence de patches et rétropropagation vers le projecteur.

    L'encodeur ONNX reste gelé. Le projecteur visuel et le Transformer sont
    entraînables. Les sorties [1,N,D] conservent N tokens ; un encodeur [1,D]
    est accepté comme un seul token global.
    """
    def __init__(self, language_model, encoder, feature_dim=None, seed=917):
        self.language_model = language_model
        self.encoder = encoder
        inferred = feature_dim or getattr(encoder, "feature_dim", None)
        if not inferred:
            raise ValueError("Indique feature_dim selon la dimension de sortie de l'encodeur ONNX.")
        self.projector = VisionProjector(inferred, language_model.d_model, seed=seed)

    def parameters(self):
        return self.language_model.parameters() + self.projector.parameters()

    def _encode_visual_rows(self, image):
        if hasattr(self.encoder, "encode_tokens"):
            rows = self.encoder.encode_tokens(image)
        else:
            rows = [self.encoder.encode(image)]
        if not rows:
            raise ValueError("L'image n'a produit aucun token visuel.")
        checked = [[float(x) for x in row] for row in rows]
        if any(len(row) != self.projector.feature_dim for row in checked):
            raise ValueError("La dimension des caractéristiques ne correspond pas au projecteur.")
        if any(not math.isfinite(x) for row in checked for x in row):
            raise ValueError("Caractéristique visuelle non finie.")
        return checked

    def _visual_embedding(self, image, *, detached=False):
        feature_rows = self._encode_visual_rows(image)
        features = Tensor(feature_rows)
        projected = self.projector.layer(features)
        if detached:
            # Détache l'observation visuelle pour réutiliser ses embeddings à
            # chaque token généré sans conserver le graphe d'inférence.
            persistent = projected.detach()
            projected.free_graph()
            return persistent
        return projected

    def forward_features(self, feature_rows, text_ids):
        ids = list(text_ids)
        if not ids:
            raise ValueError("Le texte cible doit contenir au moins un token.")
        rows = [[float(x) for x in row] for row in feature_rows]
        if not rows or any(len(row) != self.projector.feature_dim for row in rows):
            raise ValueError("Forme des caractéristiques visuelles incompatible.")
        if any(not math.isfinite(x) for row in rows for x in row):
            raise ValueError("Caractéristique visuelle non finie.")
        visual = self.projector.layer(Tensor(rows))
        visual_rows = visual.shape[0]
        if visual_rows + len(ids) - 1 > self.language_model.max_seq_len:
            visual.free_graph()
            raise ValueError("Patches visuels + texte dépassent le contexte du LLM.")
        if len(ids) > 1:
            previous_text = self.language_model.embed_tokens(ids[:-1])
            embeddings = visual.concat_rows(previous_text)
        else:
            embeddings = visual
        full_logits = self.language_model.forward_embeddings(embeddings)
        # Le dernier patch visuel prédit le premier token de la légende ; les
        # tokens suivants sont prédits par les embeddings textuels précédents.
        return full_logits.transpose().slice_cols(visual_rows - 1, len(ids)).transpose()

    def forward(self, image, text_ids):
        return self.forward_features(self._encode_visual_rows(image), text_ids)

    def train_step_features(self, feature_rows, text_ids, optimizer, max_grad_norm=1.0,
                            *, target_start=0):
        """Une mise à jour VLM ; target_start masque le prompt textuel.

        target_start=0 convient au captioning (l'image prédit le premier token).
        Pour du SFT multimodal, target_start=len(prompt_ids) entraîne uniquement
        la completion après la consigne.
        """
        ids = list(text_ids)
        if not ids:
            raise ValueError("Le texte cible doit contenir au moins un token.")
        if isinstance(target_start, bool) or not isinstance(target_start, int) or not 0 <= target_start < len(ids):
            raise ValueError("target_start doit être un index de token valide.")
        logits_all = self.forward_features(feature_rows, ids)
        if target_start:
            logits = logits_all.transpose().slice_cols(target_start, len(ids) - target_start).transpose()
        else:
            logits = logits_all
        loss = nn.cross_entropy_loss_ids(logits, ids[target_start:])
        optimizer.zero_grad()
        loss.backward()
        if hasattr(optimizer, "clip_grad_norm") and max_grad_norm is not None:
            optimizer.clip_grad_norm(max_grad_norm)
        optimizer.step()
        value = float(loss.data)
        loss.free_graph()
        return value

    def train_step(self, image, text_ids, optimizer, max_grad_norm=1.0, *, target_start=0):
        return self.train_step_features(self._encode_visual_rows(image), text_ids,
                                        optimizer, max_grad_norm=max_grad_norm,
                                        target_start=target_start)

    def generate(self, image, prompt_ids, *, max_new_tokens=40, temperature=0.0,
                 top_k=40, seed=42):
        import random
        tokens = list(prompt_ids)
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 or x >= self.language_model.vocab_size for x in tokens):
            raise ValueError("Prompt contenant un token hors vocabulaire.")
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 0:
            raise ValueError("max_new_tokens doit être positif ou nul.")
        if temperature < 0 or not math.isfinite(float(temperature)):
            raise ValueError("temperature doit être finie et >= 0.")
        if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0):
            raise ValueError("top_k doit être un entier positif ou None.")
        visual = self._visual_embedding(image, detached=True)
        if visual.shape[0] >= self.language_model.max_seq_len and max_new_tokens:
            visual.free()
            raise ValueError("La séquence de patches visuels occupe tout le contexte du LLM.")
        rng = random.Random(seed)
        try:
            for _ in range(max_new_tokens):
                text_context = tokens[-max(0, self.language_model.max_seq_len - visual.shape[0]):]
                if text_context:
                    text_emb = self.language_model.embed_tokens(text_context)
                    embeddings = visual.concat_rows(text_emb)
                else:
                    embeddings = visual
                logits = self.language_model.forward_embeddings(embeddings)
                scores = list(logits.data[-1])
                logits.free_intermediates()
                if temperature == 0:
                    next_id = max(range(len(scores)), key=scores.__getitem__)
                else:
                    scaled = [v / temperature for v in scores]
                    if top_k and top_k < len(scaled):
                        allowed = set(sorted(range(len(scaled)), key=scaled.__getitem__, reverse=True)[:top_k])
                    else:
                        allowed = None
                    max_score = max(scaled[i] for i in (allowed if allowed is not None else range(len(scaled))))
                    probs = [math.exp(scaled[i] - max_score) if allowed is None or i in allowed else 0.0
                             for i in range(len(scaled))]
                    total = sum(probs)
                    if not math.isfinite(total) or total <= 0:
                        raise RuntimeError("Distribution de génération visuelle invalide.")
                    threshold = rng.random() * total
                    running = 0.0
                    next_id = len(probs) - 1
                    for index, probability in enumerate(probs):
                        running += probability
                        if running >= threshold:
                            next_id = index
                            break
                tokens.append(next_id)
        finally:
            visual.free()
        return tokens

    def state_dict(self):
        return {
            "format": "ghost-vlm-v1",
            "language_model": self.language_model.state_dict(),
            "projector": self.projector.state_dict(),
        }

    def load_state_dict(self, state):
        if state.get("format") != "ghost-vlm-v1":
            raise ValueError("Format Vision-Language ghost inconnu.")
        # Valider le projecteur d'abord : aucun paramètre n'est modifié avant validation.
        projector = state.get("projector", {})
        if projector.get("feature_dim") != self.projector.feature_dim or projector.get("d_model") != self.projector.d_model:
            raise ValueError("Configuration du projecteur incompatible.")
        self.language_model.load_state_dict(state["language_model"])
        self.projector.load_state_dict(projector)

    def save(self, path):
        import json
        import os
        temporary = os.fspath(path) + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(self.state_dict(), stream, separators=(",", ":"))
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)

    def load(self, path):
        import json
        with open(path, encoding="utf-8") as stream:
            self.load_state_dict(json.load(stream))

    def free(self, *, free_language_model=False):
        self.projector.free()
        if free_language_model:
            self.language_model.free()

