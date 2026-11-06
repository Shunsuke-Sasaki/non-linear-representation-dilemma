#!/usr/bin/env python3
"""Checks for main_eval.py (run on a GPU node; uses the pilot run's checkpoints and data).

The key check re-evaluates the pilot's maps (trained with das.DAS_MLP, hook-based) with
das.Shared_Mapping.evaluate (split-model) and requires the same source test IIA, which confirms
the layer index, the direction of phi, and the source/variable assignment.
"""
import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import main_eval as me  # noqa: E402
from das.Classification_Model import MLPForClassification  # noqa: E402
from das.Dataset_Generation import make_intervention_dataset_AndOr, make_intervention_dataset_AndOrAnd  # noqa: E402
from das.Helper_Functions import set_seed  # noqa: E402
from das.RevNet import RevNet  # noqa: E402
from das.Shared_Mapping import Alignment, InterventionSet, evaluate, head_from, hidden_at  # noqa: E402

PILOT = ROOT / me.FULL["pilot_run"]
DEVICE = torch.device("cuda")
SOURCE_DATA_SEED = 104289  # pilot: source seed 4287 + source_mapping_offset 100000 + test offset 2


def load_model(structure):
    model = MLPForClassification(input_size=24)
    ckpt = torch.load(PILOT / "checkpoints" / "base_models" / f"{structure}_source_seed4287.pt",
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(DEVICE).eval()


def load_pilot_test(candidate):
    d = torch.load(PILOT / "data" / "mapping" / "source" / f"{candidate}_test_seed{SOURCE_DATA_SEED}.pt",
                   map_location=DEVICE, weights_only=False)
    return InterventionSet(d["base"], d["sources"].transpose(0, 1).contiguous(), d["label"].long(),
                           d["intervention"], me.distributive_label(d["base"]))


class MainEvalTests(unittest.TestCase):
    def test_reproduces_pilot_source_iia(self):
        results = json.loads((PILOT / "results.json").read_text())
        for s in "EF":
            model = load_model(s)
            for c in "EF":
                phi = RevNet(number_blocks=10, in_features=24, hidden_size=24).to(DEVICE)
                ckpt = torch.load(PILOT / "checkpoints" / "shared_mapping" / f"{s}{c}.pt",
                                  map_location=DEVICE, weights_only=False)
                phi.load_state_dict(ckpt["state_dict"])
                phi.eval()
                res, _ = evaluate(phi, (model, Alignment(24).to(DEVICE)), load_pilot_test(c),
                                  me.FULL["variable_dims"], me.FULL["layer"])
                expected = results[f"{s}{c}E"]["source_mapping_test_iia"]
                print(f"pilot map {s}{c}: split-model IIA={res['iia']:.4f}, pilot={expected:.4f}")
                self.assertAlmostEqual(res["iia"], expected, delta=0.003)

    def test_label_function_matches_trained_models(self):
        data = load_pilot_test("E")
        for s in "EF":
            with torch.no_grad():
                pred = load_model(s)(data.base).argmax(1)
            acc = (pred == data.base_label).float().mean().item()
            print(f"model {s}: accuracy against distributive_label = {acc:.4f}")
            self.assertGreater(acc, 0.99)

    def test_split_model_equals_full_forward(self):
        model = load_model("E")
        x = load_pilot_test("E").base[:256]
        with torch.no_grad():
            self.assertTrue(torch.allclose(head_from(model, hidden_at(model, x, 1), 1), model(x), atol=1e-5))

    def test_alignments_are_orthogonal_and_invertible(self):
        a0 = me.random_orthogonal_alignment(24, 123, DEVICE)
        self.assertLess(me.orthogonality_error(a0.Q), 1e-5)
        a = me.TrainableOrthogonalAlignment(a0).to(DEVICE)
        with torch.no_grad():
            a.skew.normal_(0, 0.1)
        h = torch.randn(32, 24, device=DEVICE)
        self.assertLess(me.orthogonality_error(a.matrix()), 1e-4)
        self.assertTrue(torch.allclose(a.inverse(a(h)), h, atol=1e-4))

    def test_alignment_stats(self):
        dims = me.FULL["variable_dims"]
        s = me.alignment_stats(torch.eye(24, device=DEVICE), dims)
        self.assertAlmostEqual(s["identity_distance"], 0.0, places=5)
        self.assertAlmostEqual(s["in_block_fraction"], 1.0, places=5)
        swap = torch.eye(24, device=DEVICE)[torch.cat([torch.arange(12, 24), torch.arange(12)])]
        self.assertAlmostEqual(me.alignment_stats(swap, dims)["in_block_fraction"], 0.0, places=5)
        within = torch.block_diag(me.random_orthogonal_alignment(12, 1, DEVICE).Q,
                                  me.random_orthogonal_alignment(12, 2, DEVICE).Q)
        self.assertAlmostEqual(me.alignment_stats(within, dims)["in_block_fraction"], 1.0, places=4)
        rand = [me.alignment_stats(me.random_orthogonal_alignment(24, k, DEVICE).Q, dims)["in_block_fraction"]
                for k in range(50)]
        print(f"random orthogonal in_block_fraction: mean {sum(rand) / len(rand):.3f}")
        self.assertLess(abs(sum(rand) / len(rand) - 0.5), 0.05)

    def test_intervention_data_is_label_balanced(self):
        for gen in (make_intervention_dataset_AndOrAnd, make_intervention_dataset_AndOr):
            set_seed(1)
            data = me.intervention_set(gen(4000, 4))
            changed = (data.label != data.base_label).float().mean().item()
            print(f"{gen.__name__}: label_changed rate = {changed:.3f}")
            self.assertGreater(changed, 0.45)
            self.assertLess(changed, 0.55)


if __name__ == "__main__":
    unittest.main(verbosity=2)
