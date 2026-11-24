import os
import tempfile
import unittest

from ghost import llm, nn, seq, text


class TransformerIntegrationTests(unittest.TestCase):
    def test_public_llm_facade(self):
        self.assertIs(llm.GPTModel, seq.TransformerLanguageModel)
        self.assertIs(llm.Tokenizer, text.Tokenizer)

    def test_native_embedding(self):
        emb = seq.Embedding(6, 4, seed=11)
        out = emb([3, 1, 3])
        self.assertEqual(out.shape, (3, 4))
        loss = out.sum()
        loss.backward()
        grad = emb.weight.grad
        self.assertTrue(all(abs(v) < 1e-6 for v in grad[0]))
        self.assertTrue(all(abs(v - 2.0) < 1e-5 for v in grad[3]))
        loss.free_graph()
        emb.free()

    def test_transformer_forward_backward_generation_checkpoint(self):
        model = seq.TransformerLanguageModel(
            vocab_size=24, d_model=16, n_heads=4, n_layers=1,
            max_seq_len=12, ff_mult=2, seed=5,
        )
        optimizer = nn.AdamW(model.parameters(), lr=0.002)
        corpus = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        before = model.train_step(corpus, optimizer)
        self.assertGreater(before, 0.0)
        generated = model.generate([1, 2, 3], max_new_tokens=5, temperature=0, seed=1)
        self.assertEqual(len(generated), 8)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.json")
            model.save(path)
            clone = seq.TransformerLanguageModel(
                vocab_size=24, d_model=16, n_heads=4, n_layers=1,
                max_seq_len=12, ff_mult=2, seed=99,
            )
            clone.load(path)
            sample_a = model.generate([1, 2], max_new_tokens=3, temperature=0)
            sample_b = clone.generate([1, 2], max_new_tokens=3, temperature=0)
            self.assertEqual(sample_a, sample_b)
            clone.free()
        optimizer.free()
        model.free()

    def test_tokenizer_roundtrip(self):
        tokenizer = text.Tokenizer()
        tokenizer.train(["Un petit corpus pour le modèle ghost.", "Le modèle apprend le texte."], vocab_size=280)
        tokenizer.add_special_tokens(["<|bos|>", "<|eos|>"])
        content = "<|bos|>G-SOCIETY : intelligence artificielle 🤖<|eos|>"
        ids = tokenizer.encode(content)
        self.assertEqual(tokenizer.decode(ids), content)
        self.assertEqual(tokenizer.decode(ids, skip_special_tokens=True), "G-SOCIETY : intelligence artificielle 🤖")
        self.assertEqual(tokenizer.training_stats["documents"], 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tokenizer.json")
            tokenizer.save(path)
            restored = text.Tokenizer().load(path)
            self.assertEqual(restored.encode(content), ids)
            self.assertEqual(restored.decode(ids), content)

    def test_tokenizer_resets_and_trains_from_file(self):
        tokenizer = text.Tokenizer()
        tokenizer.train(["ancien corpus", "ancien corpus"], vocab_size=270)
        self.assertGreater(len(tokenizer.merges), 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "corpus.txt")
            with open(path, "w", encoding="utf-8", newline="") as stream:
                stream.write("Bonjour 🤖\nBonjour au monde.\n")
            tokenizer.train_file(path, vocab_size=280)
            text_value = "Bonjour 🤖\nBonjour au monde.\n"
            self.assertEqual(tokenizer.decode(tokenizer.encode(text_value)), text_value)
            self.assertEqual(tokenizer.training_stats["documents"], 2)

    def test_tokenizer_unicode_roundtrip_without_training(self):
        tokenizer = text.Tokenizer()
        samples = [
            "Français : élève, cœur, déjà vu.",
            "Hausa: lafiya lau. Zarma: fofo.",
            "العربية 中文 हिन्दी Ελληνικά",
            "Tabs\tspaces  multiple\nlines\r\n🤖🧠🌍",
            "",
        ]
        for sample in samples:
            self.assertEqual(tokenizer.decode(tokenizer.encode(sample)), sample)

    def test_incremental_bpe_matches_reference_pair_counts(self):
        from collections import Counter, defaultdict
        samples = ["abababab abcabc", "abababcabc", "abcabc xyzxyz", "ghost ghost ghost"]
        tokenizer = text.Tokenizer().train(samples, vocab_size=275, min_frequency=1)

        chunks = Counter()
        for sample in samples:
            chunks.update(text.Tokenizer._pretokenize(sample))
        words = {tuple(chunk.encode("utf-8")): frequency for chunk, frequency in chunks.items() if chunk}
        expected = []
        for _ in range(19):
            pairs = Counter()
            for word, frequency in words.items():
                pairs.update({pair: count * frequency for pair, count in Counter(zip(word, word[1:])).items()})
            if not pairs:
                break
            best = min(pairs.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
            pair, count = best
            if count < 1:
                break
            expected.append(pair)
            merged = defaultdict(int)
            for word, frequency in words.items():
                merged[text.Tokenizer._apply_merge(word, pair, 256 + len(expected) - 1)] += frequency
            words = dict(merged)
        self.assertEqual(tokenizer.merges, expected)

    def test_tokenizer_rejects_corrupt_merges(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.json")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write('{"format_version":2,"algorithm":"byte_bpe","merges":[[999,1]],"special_tokens":{}}')
            with self.assertRaises(ValueError):
                text.Tokenizer().load(path)

    def test_sft_masks_prompt_and_adamw_resume_is_exact(self):
        config = dict(vocab_size=16, d_model=8, n_heads=2, n_layers=1,
                      max_seq_len=8, ff_mult=2)
        model_a = seq.TransformerLanguageModel(**config, seed=77)
        opt_a = nn.AdamW(model_a.parameters(), lr=0.003, weight_decay=0.001)
        full_sequence = [1, 2, 3, 4, 5, 6]
        # Seuls les tokens 4,5 sont des labels ; 1,2,3 forment le prompt.
        losses = [model_a.train_step(full_sequence, opt_a, target_start=3) for _ in range(8)]
        self.assertLess(losses[-1], losses[0])
        model_state = model_a.state_dict()
        optimizer_state = opt_a.state_dict()

        model_b = seq.TransformerLanguageModel(**config, seed=999)
        opt_b = nn.AdamW(model_b.parameters(), lr=0.003, weight_decay=0.001)
        model_b.load_state_dict(model_state)
        opt_b.load_state_dict(optimizer_state)
        self.assertEqual(opt_b.state_dict(), optimizer_state)

        model_a.train_step(full_sequence, opt_a, target_start=3)
        model_b.train_step(full_sequence, opt_b, target_start=3)
        for left, right in zip(model_a.parameters(), model_b.parameters()):
            for a, b in zip(left.flat_data(), right.flat_data()):
                self.assertAlmostEqual(a, b, places=6)
        with self.assertRaises(ValueError):
            model_a.train_step(full_sequence, opt_a, target_start=0)

        opt_a.free(); model_a.free()
        opt_b.free(); model_b.free()

    def test_model_learns_next_token_pattern(self):
        model = seq.TransformerLanguageModel(
            vocab_size=8, d_model=16, n_heads=4, n_layers=1,
            max_seq_len=4, ff_mult=2, seed=31,
        )
        optimizer = nn.AdamW(model.parameters(), lr=0.01, weight_decay=0.0001)
        sequence = [1, 2, 3, 4, 1]
        losses = [model.train_step(sequence, optimizer) for _ in range(50)]
        self.assertLess(losses[-1], losses[0] * 0.1)
        generated = model.generate([1, 2, 3], max_new_tokens=1, temperature=0)
        self.assertEqual(generated[-1], 4)
        with self.assertRaises(ValueError):
            model.generate([99], max_new_tokens=1)
        optimizer.free()
        model.free()


if __name__ == "__main__":
    unittest.main()
