import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import onnxruntime as ort
import torch

import model.pytorch.main as pytorch_model


MODEL_PATH = Path(__file__).parent / "weights" / "model_int8.onnx"


class ModelParityTest(unittest.TestCase):
    def test_original_model_entrypoint_runs_onnx_inference(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "main.py")],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("last_hidden_state: (1, 11, 384)", result.stdout)

    def test_pytorch_entrypoint_runs_inference(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "pytorch" / "main.py")],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("last_hidden_state: (1, 11, 384)", result.stdout)

    def test_pytorch_output_matches_onnx_baseline(self):
        load_model = getattr(pytorch_model, "load_model_from_onnx", None)
        self.assertIsNotNone(load_model, "load_model_from_onnx is not implemented")
        if load_model is None:
            return

        session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
        model = load_model(MODEL_PATH)
        input_ids_list = [
            [0, 87, 262, 9384, 2, 1],
            [0, 14343, 4724, 1839, 15091, 7742, 6, 51006, 14849, 5, 2],
            [0, 62, 54063, 25614, 14305, 2558, 13407, 9828, 216348, 1641, 3061, 3941, 32, 2],
        ]

        for input_ids in input_ids_list:
            inputs = {
                "input_ids": np.array([input_ids], dtype=np.int64),
                "attention_mask": np.ones((1, len(input_ids)), dtype=np.int64),
                "token_type_ids": np.zeros((1, len(input_ids)), dtype=np.int64),
            }
            expected = session.run(None, inputs)[0]

            with torch.inference_mode():
                actual = model(
                    **{name: torch.from_numpy(value) for name, value in inputs.items()}
                ).numpy()

            mean_absolute_error = np.abs(actual - expected).mean()
            cosine_similarity = np.dot(actual.ravel(), expected.ravel()) / (
                np.linalg.norm(actual) * np.linalg.norm(expected)
            )

            with self.subTest(input_ids=input_ids):
                self.assertLess(mean_absolute_error, 0.01)
                self.assertGreater(cosine_similarity, 0.999)


if __name__ == "__main__":
    unittest.main()
