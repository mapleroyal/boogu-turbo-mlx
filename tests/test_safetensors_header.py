from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from boogu_turbo_mlx.errors import BooguTurboMlxError
from boogu_turbo_mlx.safetensors_header import (
    MAX_HEADER_LENGTH,
    read_safetensors_header,
)
from safetensors_fixtures import write_safetensors_fixture


class SafetensorsHeaderTests(unittest.TestCase):
    def test_reads_dtype_shape_and_byte_count_without_optional_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.safetensors"
            write_safetensors_fixture(
                path,
                {
                    "decoder.conv.weight": ("F32", [2, 3]),
                    "model.language_model.embed.weight": ("BF16", [4, 5]),
                },
            )

            header = read_safetensors_header(path)

        self.assertEqual(header.metadata["format"], "pt")
        self.assertEqual(header.tensors["decoder.conv.weight"].dtype, "F32")
        self.assertEqual(header.tensors["decoder.conv.weight"].shape, (2, 3))
        self.assertEqual(header.tensors["decoder.conv.weight"].byte_count, 24)
        self.assertEqual(header.tensors["model.language_model.embed.weight"].byte_count, 40)

    def test_rejects_oversized_header_unknown_dtype_and_payload_overrun(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            oversized = root / "oversized.safetensors"
            oversized.write_bytes(struct.pack("<Q", MAX_HEADER_LENGTH + 1))

            with self.assertRaisesRegex(BooguTurboMlxError, "header larger"):
                read_safetensors_header(oversized)

            unknown_dtype = root / "unknown.safetensors"
            _write_raw_safetensors(
                unknown_dtype,
                {"bad.weight": {"dtype": "NOPE", "shape": [1], "data_offsets": [0, 1]}},
                b"\0",
            )

            with self.assertRaisesRegex(BooguTurboMlxError, "unsupported dtype"):
                read_safetensors_header(unknown_dtype)

            overrun = root / "overrun.safetensors"
            _write_raw_safetensors(
                overrun,
                {"bad.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 8]}},
                b"\0" * 4,
            )

            with self.assertRaisesRegex(BooguTurboMlxError, "beyond"):
                read_safetensors_header(overrun)


def _write_raw_safetensors(path: Path, header: dict[str, object], payload: bytes) -> None:
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)


if __name__ == "__main__":
    unittest.main()
