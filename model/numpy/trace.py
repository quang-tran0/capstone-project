from pathlib import Path

import numpy as np


class MarkdownTrace:
    def __init__(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.summary_path = directory / "trace_summary.md"
        self.full_path = directory / "trace_full.md"
        self._summary = self.summary_path.open("w", encoding="utf-8")
        self._full = self.full_path.open("w", encoding="utf-8")

        self._write_both(
            "# MiniLM Calculation Trace\n\n"
            "Operations are listed in execution order. Shapes use NumPy order. "
            "Learned parameters are summarized and remain available in the ONNX file.\n\n"
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if not self._summary.closed:
            self._summary.close()
        if not self._full.closed:
            self._full.close()

    def _write_both(self, text):
        self._summary.write(text)
        self._full.write(text)

    def section(self, title, level=2):
        self._write_both(f"{'#' * level} {title}\n\n")

    def operation(self, title, formula):
        self._write_both(f"#### {title}\n\n**Formula:** `{formula}`\n\n")

    @staticmethod
    def _metadata(value):
        array = np.asarray(value)
        flat = array.reshape(-1)
        if flat.size == 0:
            statistics = "empty"
        else:
            statistics = (
                f"min={float(np.min(flat)):.9g}, "
                f"max={float(np.max(flat)):.9g}, "
                f"mean={float(np.mean(flat, dtype=np.float64)):.9g}"
            )
        return array, flat, statistics

    def scalar(self, name, value):
        if isinstance(value, np.generic):
            value = value.item()
        self._write_both(f"- **{name}:** `{value}`\n")

    def tensor(self, name, value, *, parameter=False):
        array, flat, statistics = self._metadata(value)
        sample = np.array2string(
            flat[:8], separator=", ", threshold=8, max_line_width=1000
        )
        metadata = (
            f"shape={array.shape}, dtype={array.dtype}, {statistics}, sample={sample}"
        )
        self._summary.write(f"- **{name}:** {metadata}\n")
        self._full.write(f"- **{name}:** {metadata}\n")

        if parameter:
            self._full.write("  - Values: immutable parameter; see the ONNX weights file.\n")
            return

        self._full.write("\n  Values (row-major, complete):\n\n  ```text\n")
        if flat.size == 0:
            self._full.write("  []\n")
        else:
            for start in range(0, flat.size, 128):
                values = np.array2string(
                    flat[start : start + 128],
                    separator=", ",
                    threshold=np.inf,
                    max_line_width=1_000_000,
                )
                self._full.write(f"  {start}: {values}\n")
        self._full.write("  ```\n\n")
