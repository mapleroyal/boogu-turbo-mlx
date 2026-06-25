from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .errors import BooguTurboMlxError


MIN_DISK_MARGIN_BYTES = 64 * 1024 * 1024
MAX_DISK_MARGIN_BYTES = 2 * 1024 * 1024 * 1024


@contextmanager
def atomic_output_dir(output: Path, *, label: str) -> Iterator[Path]:
    ensure_empty_or_missing_output_dir(output, label=label)
    _ensure_parent_dir(output)
    temp = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.tmp-",
            dir=output.parent,
        )
    )
    committed = False
    try:
        yield temp
        _replace_output_dir(temp, output, label=label)
        committed = True
    finally:
        if not committed and temp.exists():
            shutil.rmtree(temp, ignore_errors=True)


def ensure_empty_or_missing_output_dir(output: Path, *, label: str) -> None:
    if not output.exists():
        return
    if not output.is_dir():
        raise BooguTurboMlxError(f"{label} output must be a directory: {output}")
    if any(output.iterdir()):
        raise BooguTurboMlxError(
            f"Output directory must be empty before {label}: {output}"
        )


def preflight_free_space(
    output: Path,
    *,
    required_bytes: int,
    label: str,
) -> None:
    if required_bytes <= 0:
        return
    _ensure_parent_dir(output)
    required = _with_margin(required_bytes)
    free = shutil.disk_usage(output.parent).free
    if free < required:
        raise BooguTurboMlxError(
            f"Not enough free disk space for {label}: need about "
            f"{_format_bytes(required)}, found {_format_bytes(free)} at {output.parent}."
        )


def _replace_output_dir(temp: Path, output: Path, *, label: str) -> None:
    ensure_empty_or_missing_output_dir(output, label=label)
    if output.exists():
        output.rmdir()
    os.replace(temp, output)


def _ensure_parent_dir(output: Path) -> None:
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BooguTurboMlxError(
            f"Unable to create output parent directory {output.parent}: {exc}"
        ) from exc


def _with_margin(required_bytes: int) -> int:
    margin = max(MIN_DISK_MARGIN_BYTES, required_bytes // 20)
    margin = min(margin, MAX_DISK_MARGIN_BYTES)
    return int(required_bytes) + margin


def _format_bytes(value: int) -> str:
    units = ("bytes", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            if unit == "bytes":
                return f"{int(amount)} {unit}"
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
