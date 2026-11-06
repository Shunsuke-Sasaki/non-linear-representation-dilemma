#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import pilot
from das.Classification_Model import MLPForClassification
from das.DAS import phi_class
from das.RevNet import RevNet


class PilotTests(unittest.TestCase):
    def test_procrustes_recovers_orthogonal_alignment(self):
        torch.manual_seed(7)
        target = torch.randn(128, 24)
        rotation, _ = torch.linalg.qr(torch.randn(24, 24))
        if torch.linalg.det(rotation) < 0:
            rotation[:, -1] *= -1
        source = target @ rotation

        recovered = pilot.procrustes_rotation(target, source)

        self.assertTrue(torch.allclose(recovered, rotation, atol=1e-5, rtol=1e-5))

    def test_orthogonal_adapter_round_trip(self):
        torch.manual_seed(9)
        initial_rotation, _ = torch.linalg.qr(torch.randn(24, 24))
        if torch.linalg.det(initial_rotation) < 0:
            initial_rotation[:, -1] *= -1
        adapter = pilot.OrthogonalAdapter(initial_rotation)
        inputs = torch.randn(8, 24)

        output = adapter(inputs)

        self.assertTrue(torch.allclose(adapter.inverse(output), inputs, atol=1e-5, rtol=1e-5))
        matrix = adapter.matrix()
        identity = torch.eye(24)
        self.assertTrue(torch.allclose(matrix @ matrix.T, identity, atol=1e-5, rtol=1e-5))

    def test_target_transform_round_trip(self):
        torch.manual_seed(11)
        mapping = RevNet(number_blocks=2, in_features=24, hidden_size=24)
        initial_rotation, _ = torch.linalg.qr(torch.randn(24, 24))
        if torch.linalg.det(initial_rotation) < 0:
            initial_rotation[:, -1] *= -1
        transform = pilot.TargetTransform(mapping, pilot.OrthogonalAdapter(initial_rotation))
        inputs = torch.randn(8, 24)

        output = transform(inputs)

        self.assertTrue(torch.allclose(transform.inverse(output), inputs, atol=1e-5, rtol=1e-5))

    def test_tensor_dataset_expands_interventions_to_hidden_dimensions(self):
        model = MLPForClassification(input_size=24, num_blocks=3)
        samples = 4
        dataset = {
            "base": torch.randn(samples, 24),
            "label": torch.tensor([0, 1, 0, 1]),
            "sources": torch.randn(samples, 2, 24),
            "intervention": torch.tensor(
                [[True, False], [False, True], [True, True], [False, False]],
                dtype=torch.bool,
            ),
        }
        identity = nn.Identity()
        transform = phi_class(identity, lambda values: values, nn.CrossEntropyLoss(), None, None)
        experiment = pilot.TensorDatasetDAS(
            Model=model,
            Model_Layer=model.mlp.h[1],
            Train_Data_Raw=dataset,
            Test_Data_Raw=dataset,
            Eval_Data_Raw=dataset,
            Hidden_Layer_Size=24,
            Variable_Dimensions=[list(range(0, 12)), list(range(12, 24))],
            Transformation_Class=transform,
            Device="cpu",
        )
        try:
            expanded = experiment.Train_Dataset["intervention"]
            self.assertEqual(tuple(expanded.shape), (samples, 24))
            self.assertTrue(expanded[0, :12].all())
            self.assertFalse(expanded[0, 12:].any())
            self.assertTrue(expanded[2].all())
            self.assertFalse(expanded[3].any())
        finally:
            experiment.Cleanup()


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
