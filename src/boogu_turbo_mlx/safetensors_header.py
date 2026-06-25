from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import BooguTurboMlxError


DTYPE_BYTE_SIZES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}
MAX_HEADER_LENGTH = 100 * 1024 * 1024


@dataclass(frozen=True)
class TensorHeader:
    key: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]

    @property
    def byte_count(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]

    @property
    def expected_byte_count(self) -> int | None:
        dtype_size = DTYPE_BYTE_SIZES.get(self.dtype)
        if dtype_size is None:
            return None
        return math.prod(self.shape) * dtype_size


@dataclass(frozen=True)
class SafetensorsHeader:
    metadata: dict[str, str]
    tensors: dict[str, TensorHeader]


def read_safetensors_header(path: Path) -> SafetensorsHeader:
    """Read only the safetensors header, not the tensor payload."""
    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            header_length_bytes = handle.read(8)
            if len(header_length_bytes) != 8:
                raise BooguTurboMlxError(f"{path} is too small to contain a safetensors header")

            header_length = struct.unpack("<Q", header_length_bytes)[0]
            if header_length > MAX_HEADER_LENGTH:
                raise BooguTurboMlxError(
                    f"{path} declares a safetensors header larger than {MAX_HEADER_LENGTH} bytes"
                )
            if header_length > file_size - 8:
                raise BooguTurboMlxError(
                    f"{path} declares a safetensors header longer than the file"
                )
            header_bytes = handle.read(header_length)
            if len(header_bytes) != header_length:
                raise BooguTurboMlxError(
                    f"{path} ended before the declared safetensors header was complete"
                )
    except OSError as exc:
        raise BooguTurboMlxError(f"Unable to read safetensors header from {path}: {exc}") from exc

    try:
        raw_header = json.loads(header_bytes)
    except json.JSONDecodeError as exc:
        raise BooguTurboMlxError(f"Invalid safetensors header JSON in {path}: {exc}") from exc

    if not isinstance(raw_header, dict):
        raise BooguTurboMlxError(f"Safetensors header in {path} is not a JSON object")

    metadata = raw_header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        metadata = {}

    tensors: dict[str, TensorHeader] = {}
    for key, value in raw_header.items():
        if key == "__metadata__":
            continue
        tensors[str(key)] = _parse_tensor_header(
            path,
            str(key),
            value,
            payload_size=file_size - 8 - header_length,
        )

    return SafetensorsHeader(
        metadata={str(key): str(value) for key, value in metadata.items()},
        tensors=tensors,
    )


def _parse_tensor_header(
    path: Path,
    key: str,
    value: Any,
    *,
    payload_size: int,
) -> TensorHeader:
    if not isinstance(value, dict):
        raise BooguTurboMlxError(f"Tensor {key!r} in {path} has a non-object header")

    dtype = value.get("dtype")
    shape = value.get("shape")
    data_offsets = value.get("data_offsets")
    if not isinstance(dtype, str):
        raise BooguTurboMlxError(f"Tensor {key!r} in {path} is missing a string dtype")
    if dtype not in DTYPE_BYTE_SIZES:
        raise BooguTurboMlxError(
            f"Tensor {key!r} in {path} has unsupported dtype {dtype!r}"
        )
    if not (
        isinstance(shape, list)
        and all(isinstance(dim, int) and dim >= 0 for dim in shape)
    ):
        raise BooguTurboMlxError(f"Tensor {key!r} in {path} has an invalid shape")
    if not (
        isinstance(data_offsets, list)
        and len(data_offsets) == 2
        and all(isinstance(offset, int) and offset >= 0 for offset in data_offsets)
        and data_offsets[0] <= data_offsets[1]
    ):
        raise BooguTurboMlxError(f"Tensor {key!r} in {path} has invalid data_offsets")
    if data_offsets[1] > payload_size:
        raise BooguTurboMlxError(
            f"Tensor {key!r} in {path} points beyond the safetensors payload"
        )

    tensor = TensorHeader(
        key=key,
        dtype=dtype,
        shape=tuple(shape),
        data_offsets=(data_offsets[0], data_offsets[1]),
    )
    expected = tensor.expected_byte_count
    if tensor.byte_count != expected:
        raise BooguTurboMlxError(
            f"Tensor {key!r} in {path} declares {tensor.byte_count} bytes but "
            f"shape {list(tensor.shape)} and dtype {tensor.dtype} require {expected}"
        )
    return tensor
