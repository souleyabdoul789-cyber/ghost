import os
import tempfile
import unittest

from ghost import nn, seq
from ghost.tensor import Tensor
from ghost.vision import VisionLanguageModel, VisionProjector


class FakeVisionEncoder:
    feature_dim = 4
    def encode(self, image):
        return [0.25, 0.5, 0.75, 1.0] if image == "sample" else [1.0, 0.5, 0.25, 0.0]

    def encode_tokens(self, image):
        base = self.encode(image)
        return [base, [value * 0.5 for value in base], [value * 0.25 for value in base]]


class VisionIntegrationTests(unittest.TestCase):
    def test_projector_shape_and_validation(self):
        projector = VisionProjector(4, 8, seed=2)
        output = projector([0.1, 0.2, 0.3, 0.4])
        self.assertEqual(output.shape, (1, 8))
        output.free_graph()
        with self.assertRaises(ValueError):
            projector([1.0])
        projector.free()

    def test_vision_language_sft_masks_prompt(self):
        language = seq.TransformerLanguageModel(
            vocab_size=12, d_model=8, n_heads=2, n_layers=1, max_seq_len=10, ff_mult=2, seed=22
        )
        vlm = VisionLanguageModel(language, FakeVisionEncoder(), feature_dim=4, seed=23)
        optimizer = nn.AdamW(vlm.parameters(), lr=0.001, weight_decay=0.0)
        loss = vlm.train_step_features(
            FakeVisionEncoder().encode_tokens("sample"), [2, 3, 4, 5, 6],
            optimizer, target_start=3,
        )
        self.assertTrue(loss > 0.0)
        with self.assertRaises(ValueError):
            vlm.train_step_features(FakeVisionEncoder().encode_tokens("sample"), [2, 3],
                                    optimizer, target_start=2)
        optimizer.free()
        vlm.free(free_language_model=True)

    def test_vision_language_forward_backward_generation(self):
        language = seq.TransformerLanguageModel(
            vocab_size=12, d_model=8, n_heads=2, n_layers=1, max_seq_len=8, ff_mult=2, seed=12
        )
        vlm = VisionLanguageModel(language, FakeVisionEncoder(), feature_dim=4, seed=13)
        optimizer = nn.AdamW(vlm.parameters(), lr=0.003, weight_decay=0.0)
        losses = [vlm.train_step("sample", [2, 3, 4, 5], optimizer) for _ in range(6)]
        self.assertTrue(all(value > 0 for value in losses))
        generated = vlm.generate("sample", [2, 3], max_new_tokens=2, temperature=0)
        self.assertEqual(len(generated), 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "vision.json")
            vlm.save(path)
            state = vlm.state_dict()
            clone_language = seq.TransformerLanguageModel(
                vocab_size=12, d_model=8, n_heads=2, n_layers=1,
                max_seq_len=8, ff_mult=2, seed=99,
            )
            clone = VisionLanguageModel(clone_language, FakeVisionEncoder(), feature_dim=4)
            clone.load(path)
            self.assertEqual(clone.state_dict(), state)
            clone.free(free_language_model=True)
        optimizer.free()
        vlm.free(free_language_model=True)


if __name__ == "__main__":
    unittest.main()
