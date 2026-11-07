"""Unit tests for the utility panel: scoring math, the gate, the validity statistics and the neuron edits.

A tiny random Llama and a character-level tokenizer stand in for SmolLM2, so nothing is downloaded.
Run: python -m unittest discover -s tests
"""
import os
import sys
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    import transformers  # noqa: F401
    HAVE_HF = True
except ImportError:  # the panel's grader-side dependency; the G1 tests don't need it
    HAVE_HF = False


def tiny_lm(seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM
    from panel.score import LM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256)
    return LM(LlamaForCausalLM(cfg).eval(), lambda s: [ord(c) % 128 for c in s], max_length=256, batch_size=3)


@unittest.skipUnless(HAVE_HF, "transformers not installed")
class ScoreTest(unittest.TestCase):
    def test_loglik_matches_a_direct_forward_pass(self):
        lm = tiny_lm()
        ctx, cont = "Question: why?\nAnswer:", " because"
        (ll, greedy), = lm.loglik([(ctx, cont)])
        ids = lm.encode(ctx + cont)
        n = len(cont)
        with torch.no_grad():
            logp = torch.log_softmax(lm.model(input_ids=torch.tensor([ids[:-1]])).logits[0], -1)
        want = sum(float(logp[len(ids) - 1 - n + k, ids[len(ids) - n + k]]) for k in range(n))
        self.assertAlmostEqual(ll, want, places=4)
        self.assertEqual(greedy, all(int(logp[len(ids) - 1 - n + k].argmax()) == ids[len(ids) - n + k] for k in range(n)))

    def test_batching_and_padding_do_not_change_scores(self):
        lm = tiny_lm()
        reqs = [("a", " bb"), ("a much longer context here", " c"), ("mid size", " dd ee")]
        batched = lm.loglik(reqs)
        lm.batch_size = 1
        single = lm.loglik(reqs)
        for (a, ga), (b, gb) in zip(batched, single):
            self.assertAlmostEqual(a, b, places=4)
            self.assertEqual(ga, gb)

    def test_trailing_context_space_moves_to_the_continuation(self):
        lm = tiny_lm()
        self.assertEqual(lm.encode_pair("abc ", "d"), (lm.encode("abc"), lm.encode(" d")))

    def test_acc_norm_divides_by_choice_length_in_characters(self):
        from panel.score import score_mc

        class Fixed:  # log-likelihoods chosen so acc and acc_norm disagree
            def loglik(self, reqs):
                return [(-4.0, False), (-3.0, False)]
        r = score_mc(Fixed(), [("q", ["a much longer choice", "ab"], 0)])
        self.assertEqual(r["per_item_acc"], [0])  # -3 > -4, so plain acc picks choice 1
        self.assertEqual(r["per_item"], [1])  # -4/20 > -3/2, so acc_norm picks choice 0, the gold one
        self.assertEqual(r["acc_norm"], 1.0)

    def test_ece(self):
        from panel.score import ece
        self.assertAlmostEqual(ece([0.9] * 10, [1] * 9 + [0]), 0.0, places=6)
        self.assertAlmostEqual(ece([1.0] * 4, [0, 0, 0, 0]), 1.0, places=6)

    def test_kl_canaries_are_zero_against_itself_and_positive_after_an_edit(self):
        from panel import edits
        from panel.score import kl_random, score_text
        parent, lm = tiny_lm(), tiny_lm()
        ids = list(range(100)) * 3
        self.assertAlmostEqual(kl_random(lm, parent, 128, n_seq=6, seq_len=16), 0.0, places=6)
        self.assertAlmostEqual(score_text(lm, ids, 64, parent)["kl_to_parent"], 0.0, places=6)
        edits.delete(lm.model, edits.random_neurons(lm.model, 0.5, 0))
        self.assertGreater(kl_random(lm, parent, 128, n_seq=6, seq_len=16), 0.0)
        self.assertGreater(score_text(lm, ids, 64, parent)["kl_to_parent"], 0.0)


@unittest.skipUnless(HAVE_HF, "transformers not installed")
class EditTest(unittest.TestCase):
    def test_delete_zeroes_exactly_the_chosen_neurons(self):
        from panel import edits
        lm = tiny_lm()
        before = {k: v.clone() for k, v in lm.model.state_dict().items()}
        edits.delete(lm.model, [(0, 3), (1, 7)])
        ms = edits.mlps(lm.model)
        self.assertTrue(torch.all(ms[0].down_proj.weight[:, 3] == 0) and torch.all(ms[1].gate_proj.weight[7] == 0))
        changed = [k for k, v in lm.model.state_dict().items() if not torch.equal(v, before[k])]
        self.assertEqual(sorted(changed), sorted(f"model.layers.{l}.mlp.{p}.weight" for l in (0, 1)
                                                 for p in ("gate_proj", "up_proj", "down_proj")))
        self.assertTrue(torch.equal(ms[0].down_proj.weight[:, 4], before["model.layers.0.mlp.down_proj.weight"][:, 4]))

    def test_least_contributing_ranks_by_activation_times_output_norm(self):
        from panel import edits
        lm = tiny_lm()
        mag = edits.activation_magnitude(lm.model, list(range(64)), window=32)
        self.assertEqual(len(mag), 2 * 48)
        score = edits.contribution(lm.model, mag)
        picked = edits.least_contributing_neurons(lm.model, 0.25, mag)
        self.assertEqual(len(picked), 24)
        for layer in (0, 1):  # ranked within each layer, never across layers
            mine = [n for n in picked if n[0] == layer]
            self.assertEqual(len(mine), 12)
            self.assertLessEqual(max(score[n] for n in mine),
                                 min(score[n] for n in score if n[0] == layer and n not in picked))

    def _logits(self, model, x):
        with torch.no_grad():
            return model(input_ids=x).logits

    def _reload(self, model):
        import tempfile
        from transformers import LlamaForCausalLM
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d)
            return LlamaForCausalLM.from_pretrained(d).eval()

    def test_function_preserving_mlp_growth_changes_nothing_and_reloads(self):
        from panel import edits
        lm = tiny_lm()
        x = torch.randint(0, 128, (2, 20))
        before = self._logits(lm.model, x)
        edits.grow_mlp(lm.model, 8, out_scale=0.0)
        self.assertEqual(lm.model.config.intermediate_size, 56)
        self.assertEqual(edits.mlps(lm.model)[1].down_proj.weight.shape, (32, 56))
        self.assertTrue(torch.allclose(self._logits(lm.model, x), before, atol=1e-6))
        self.assertTrue(torch.allclose(self._logits(self._reload(lm.model), x), before, atol=1e-6))

    def test_growth_with_live_outputs_changes_the_model(self):
        from panel import edits
        lm = tiny_lm()
        x = torch.randint(0, 128, (2, 20))
        before = self._logits(lm.model, x)
        edits.grow_mlp(lm.model, 8, out_scale=1.0)
        self.assertFalse(torch.allclose(self._logits(lm.model, x), before, atol=1e-4))

    def test_identity_layer_insertion_changes_nothing_reloads_and_caches(self):
        from panel import edits
        lm = tiny_lm()
        x = torch.randint(0, 128, (2, 20))
        before = self._logits(lm.model, x)
        edits.insert_identity_layers(lm.model, [0, 1])
        self.assertEqual(lm.model.config.num_hidden_layers, 4)
        self.assertEqual([l.self_attn.layer_idx for l in lm.model.model.layers], [0, 1, 2, 3])
        self.assertTrue(torch.allclose(self._logits(lm.model, x), before, atol=1e-6))
        self.assertTrue(torch.allclose(self._logits(self._reload(lm.model), x), before, atol=1e-6))
        with torch.no_grad():  # the KV-cache path still works with the renumbered layers
            out = lm.model.generate(x[:1, :5], max_new_tokens=3, do_sample=False)
        self.assertEqual(out.shape, (1, 8))

    def test_random_deletion_matches_the_careful_count_in_every_layer(self):
        from panel import edits
        lm = tiny_lm()
        picked = edits.random_neurons(lm.model, 0.25, seed=3)
        self.assertEqual([sum(1 for n in picked if n[0] == l) for l in (0, 1)], [12, 12])
        self.assertEqual(len(set(picked)), 24)
        self.assertEqual(picked, edits.random_neurons(lm.model, 0.25, seed=3))


class GateTest(unittest.TestCase):
    def test_change_is_signed_so_positive_is_better(self):
        from panel.gate import KL_CANARIES, METRICS, change
        parent = {m: 0.5 for m in METRICS}
        edit = dict(parent, **{"hellaswag.acc_norm": 0.6, "wikitext.nll": 0.7, "kl_random": 0.2})
        c = change(edit, parent)
        self.assertAlmostEqual(c["hellaswag.acc_norm"], 0.1)
        self.assertAlmostEqual(c["wikitext.nll"], -0.2)
        self.assertAlmostEqual(c["kl_random"], -0.2)
        self.assertEqual(set(c), set(METRICS))
        self.assertTrue(all(m in METRICS for m in KL_CANARIES))

    def test_gate_tolerance_is_twice_baseline_sigma_with_a_floor(self):
        from panel.gate import FLOORS, METRICS, gate, tolerances
        quiet = {"a": [{m: 0.0 for m in METRICS}] * 3}
        self.assertEqual(tolerances(quiet), FLOORS)
        noisy = {"a": [{m: v for m in METRICS} for v in (-0.05, 0.0, 0.05)]}
        tol = tolerances(noisy)
        self.assertAlmostEqual(tol["piqa.acc_norm"], 0.1, places=6)
        ok, failed = gate(dict({m: 0.0 for m in METRICS}, **{"piqa.acc_norm": -0.09}), tol)
        self.assertTrue(ok)
        ok, failed = gate(dict({m: 0.0 for m in METRICS}, **{"piqa.acc_norm": -0.11}), tol)
        self.assertFalse(ok)
        self.assertEqual(list(failed), ["piqa.acc_norm"])

    def test_spearman(self):
        from panel.gate import spearman, spearman_ci
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 25, 100]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertAlmostEqual(spearman([1, 2, 2, 3], [1, 2, 2, 3]), 1.0)  # ties
        lo, hi = spearman_ci(np.arange(20), np.arange(20) + np.random.default_rng(0).normal(0, 1, 20))
        self.assertLessEqual(lo, hi)


class ManifestTest(unittest.TestCase):
    def test_manifest_pins_every_task(self):
        from panel.tasks import N_ITEMS, TASKS, load_manifest
        m = load_manifest()
        self.assertEqual(set(m["tasks"]), set(TASKS))
        for name, e in m["tasks"].items():
            self.assertEqual(len(e["revision"]), 40)  # a full commit sha, not a branch
            if e["kind"] == "ppl":
                self.assertIn("text_sha256", e)
            else:
                self.assertEqual(len(e["indices"]), N_ITEMS)
                self.assertEqual(len(set(e["indices"])), N_ITEMS)
                self.assertTrue(all(0 <= i < e["size"] for i in e["indices"]))

    def test_prompts_follow_lm_eval(self):
        from panel.tasks import to_request
        ctx, chs, gold = to_request("arc_easy", {"question": "Q?", "choices": {"text": ["x", "y"], "label": ["A", "B"]},
                                                 "answerKey": "B"})
        self.assertEqual((ctx, chs, gold), ("Question: Q?\nAnswer:", ["x", "y"], 1))
        self.assertEqual(to_request("lambada", {"text": "the cat sat"}), ("the cat", " sat"))
        ctx, chs, _ = to_request("hellaswag", {"activity_label": "Roof", "ctx_a": "A man.", "ctx_b": "he",
                                               "endings": ["x [title] y", "z"], "label": "0"})
        self.assertEqual(ctx, "Roof: A man. He")
        self.assertEqual(chs[0], "x. y")


if __name__ == "__main__":
    unittest.main()
