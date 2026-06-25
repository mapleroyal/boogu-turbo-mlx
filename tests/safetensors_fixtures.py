from __future__ import annotations

import json
import math
import struct
from pathlib import Path


DTYPE_BYTE_SIZES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "I64": 8,
}


def write_safetensors_fixture(
    path: Path,
    tensors: dict[str, tuple[str, list[int]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    data = bytearray()
    offset = 0
    for key, (dtype, shape) in tensors.items():
        byte_count = math.prod(shape) * DTYPE_BYTE_SIZES[dtype]
        header[key] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + byte_count],
        }
        data.extend(b"\0" * byte_count)
        offset += byte_count

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + data)
