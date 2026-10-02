import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import onnxruntime as ort

from .config import MODEL_PATH
from .model import MiniLM


VIETNAMESE_IDS = [0, 14343, 4724, 1839, 15091, 7742, 6, 51006, 14849, 5, 2]
AMBA_IDS = [
    0,
    62,
    54063,
    25614,
    14305,
    2558,
    13407,
    9828,
    216348,
    1641,
    3061,
    3941,
    32,
    2,
]


class NumpyModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MiniLM.from_onnx(MODEL_PATH)
        cls.session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )

    def assert_matches_onnx(self, inputs):
        expected = self.session.run(None, inputs)[0]
        actual = self.model(**inputs)
        difference = np.abs(actual - expected)
        cosine_similarity = np.dot(actual.ravel(), expected.ravel()) / (
            np.linalg.norm(actual) * np.linalg.norm(expected)
        )

        self.assertLess(float(difference.mean()), 0.01)
        self.assertGreater(float(cosine_similarity), 0.999)

    def test_cli_runs_inference(self):
        result = subprocess.run(
            [sys.executable, "-m", "model.numpy.main"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("input_ids: (1, 11)", result.stdout)
        self.assertIn("last_hidden_state: (1, 11, 384)", result.stdout)
        self.assertIn("first_token[:8]:", result.stdout)

    def test_cli_runs_as_a_direct_script_from_model_directory(self):
        result = subprocess.run(
            [sys.executable, "numpy/main.py"],
            cwd=Path(__file__).parents[1],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("input_ids: (1, 11)", result.stdout)
        self.assertIn("last_hidden_state: (1, 11, 384)", result.stdout)

    def test_cli_writes_summary_and_full_markdown_traces(self):
        with tempfile.TemporaryDirectory() as trace_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "model.numpy.main",
                    "--trace-dir",
                    trace_dir,
                    "AMBA",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            summary_path = Path(trace_dir) / "trace_summary.md"
            full_path = Path(trace_dir) / "trace_full.md"
            self.assertTrue(summary_path.is_file())
            self.assertTrue(full_path.is_file())

            summary = summary_path.read_text(encoding="utf-8")
            full = full_path.read_text(encoding="utf-8")
            for heading in (
                "# MiniLM Calculation Trace",
                "## Input",
                "## Embeddings",
                "## Encoder Layer 0",
                "### Self-Attention",
                "### Attention Output",
                "### Feed Forward",
                "## Encoder Layer 11",
            ):
                self.assertIn(heading, summary)
                self.assertIn(heading, full)
            self.assertIn("shape=", summary)
            self.assertIn("sample=", summary)
            self.assertIn("Values (row-major, complete)", full)

    def test_short_input_returns_hidden_state(self):
        inputs = {
            "input_ids": np.array([[0, 87, 2]], dtype=np.int64),
            "attention_mask": np.ones((1, 3), dtype=np.int64),
            "token_type_ids": np.zeros((1, 3), dtype=np.int64),
        }

        output = self.model(**inputs)

        self.assertEqual(output.shape, (1, 3, 384))
        self.assertEqual(output.dtype, np.float32)

    def test_vietnamese_and_amba_inputs_match_onnx(self):
        for input_ids in (VIETNAMESE_IDS, AMBA_IDS):
            inputs = {
                "input_ids": np.array([input_ids], dtype=np.int64),
                "attention_mask": np.ones((1, len(input_ids)), dtype=np.int64),
                "token_type_ids": np.zeros((1, len(input_ids)), dtype=np.int64),
            }
            with self.subTest(input_ids=input_ids):
                self.assert_matches_onnx(inputs)

    def test_padded_batch_matches_onnx(self):
        first = [0, 87, 262, 9384, 2] + [1] * 9
        inputs = {
            "input_ids": np.array([first, AMBA_IDS], dtype=np.int64),
            "attention_mask": np.array(
                [[1] * 5 + [0] * 9, [1] * len(AMBA_IDS)],
                dtype=np.int64,
            ),
            "token_type_ids": np.zeros((2, len(AMBA_IDS)), dtype=np.int64),
        }

        self.assert_matches_onnx(inputs)
        self.assertEqual(self.model(**inputs).shape, (2, 14, 384))

    def test_rejects_sequence_longer_than_position_table(self):
        with self.assertRaisesRegex(ValueError, r"512"):
            self.model(np.zeros((1, 513), dtype=np.int64))

    def test_all_zero_attention_mask_stays_finite(self):
        output = self.model(
            np.array([[0, 87, 2]], dtype=np.int64),
            attention_mask=np.zeros((1, 3), dtype=np.int64),
        )

        self.assertTrue(np.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
