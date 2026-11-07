#!/usr/bin/env python3
"""Checks for ioi_eval.py. Run on a GPU node after `ioi_eval.py smoke` (uses its data and models)."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import ioi_eval as ie  # noqa: E402

SMOKE_ROOT = ROOT / "results" / "ioi_smoke"
DEVICE = torch.device("cuda")


class IOITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_float32_matmul_precision(ie.SMOKE["matmul_precision"])
        cls.exp = ie.Experiment(SMOKE_ROOT, ie.SMOKE, DEVICE, offline=True)
        cls.model = cls.exp.get_model("trained", 1)
        cls.test = cls.exp.pairs("default", "test")

    def batch(self, n=32):
        return self.test.subset(torch.arange(n))

    def test_parse_sentences(self):
        tok = self.exp.tokenizer()
        prompts, meta = ie.parse_sentences([
            "Then, Mary and John went to the store. John gave a apple to Mary",   # ABBA: IO first
            "Then, John and Mary went to the store. John gave a apple to Mary",   # BABA: IO second
            "Then, Zoe and Adam went to the store. Zoe gave a apple to Adam",     # IO precedes S alphabetically
        ], tok)
        self.assertEqual(prompts[0], "Then, Mary and John went to the store. John gave a apple to")
        self.assertEqual([m[2] for m in meta], [False, True, True])
        self.assertEqual([m[3] for m in meta], [False, False, True])
        self.assertEqual(meta[0][0], tok(" Mary", add_special_tokens=False)["input_ids"][0])
        self.assertEqual(meta[0][1], tok(" John", add_special_tokens=False)["input_ids"][0])

    def test_labels(self):
        p = self.batch(256)
        same = ie.Pairs(p.pool, p.base, p.base)
        for cand in ie.CANDIDATES:
            true, false, changed = same.labels(cand)
            self.assertTrue(torch.equal(true, p.pool["tok_io"][p.base]))
            self.assertFalse(changed.any())
            true, false, changed = p.labels(cand)
            flip = p.pool[cand][p.src] != p.pool[cand][p.base]
            self.assertTrue(torch.equal(changed, flip))
            self.assertTrue(torch.equal(true[flip], p.pool["tok_s"][p.base][flip]))
        agree = (p.labels("pos")[0] == p.labels("alpha")[0]).float().mean().item()
        print(f"pos/alpha counterfactual label agreement on 256 test pairs: {agree:.3f}")

    def test_last_logits_match_full_forward(self):
        ids, mask, last = ie.batch_tokens(self.test.pool, self.test.base[:16], DEVICE)
        with torch.no_grad():
            full = self.model(input_ids=ids, attention_mask=mask).logits[torch.arange(16, device=DEVICE), last]
            ours = ie.last_logits(self.model, ids, mask, last)
        self.assertLess((full - ours).abs().max().item(), 1e-2)

    def test_noop_intervention(self):
        p = self.batch(32)
        same = ie.Pairs(p.pool, p.base, p.base)
        ids, mask, last = ie.batch_tokens(p.pool, p.base, DEVICE)
        cfg = ie.SMOKE
        with torch.no_grad():
            plain = ie.last_logits(self.model, ids, mask, last)
            ident = ie.intervened_logits(ie.IdentityMap(), (self.model, ie.IdentityAlignment()), same, cfg, DEVICE)
            phi = self.exp.new_phi(7).eval()
            rot = ie.FixedOrthogonal(ie.random_orthogonal(cfg["hidden"], 3).to(DEVICE))
            nonlin = ie.intervened_logits(phi, (self.model, rot), same, cfg, DEVICE)
        print(f"no-op intervention max |dlogit|: identity {(plain - ident).abs().max().item():.2e}, "
              f"RevNet+rotation {(plain - nonlin).abs().max().item():.2e}")
        self.assertLess((plain - ident).abs().max().item(), 1e-3)
        self.assertTrue(torch.equal(plain.argmax(1), nonlin.argmax(1)))
        self.assertLess((plain - nonlin).abs().max().item(), 0.05)

    def test_trained_model_solves_ioi(self):
        idx = self.test.base.unique()[:256]
        acc_t = ie.task_accuracy(self.model, self.test.pool, idx, ie.SMOKE, DEVICE)
        acc_r = ie.task_accuracy(self.exp.get_model("random", 1), self.test.pool, idx, ie.SMOKE, DEVICE)
        print(f"task accuracy seed1: trained {acc_t:.3f}, random {acc_r:.3f}")
        self.assertGreater(acc_t, 0.9)

    def test_cayley_alignment(self):
        a = ie.TrainableOrthogonalAlignment(ie.random_orthogonal(1024, 1)).to(DEVICE)
        with torch.no_grad():
            a.skew.normal_(0, 0.01)
        M = a.matrix()
        self.assertLess((M.T @ M - torch.eye(1024, device=DEVICE)).abs().max().item(), 1e-3)
        h = torch.randn(8, 1024, device=DEVICE)
        self.assertLess((a.inverse(a(h)) - h).abs().max().item(), 1e-3)

    def test_smoke_outputs(self):
        self.assertTrue((SMOKE_ROOT / "summary.md").exists())
        self.assertTrue(any((SMOKE_ROOT / "adapt").glob("*.jsonl")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
