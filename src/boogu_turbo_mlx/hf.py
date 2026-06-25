from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .errors import BooguTurboMlxError
from .manifest import _resolve_remote_revision


def download_source(
    source: str | Path,
    *,
    revision: str | None = None,
    dest_root: str | Path = Path("models"),
    local_dir: str | Path | None = None,
) -> Path:
    """Download a Hugging Face model source, or return an existing local path."""

    source_path = Path(str(source)).expanduser()
    if source_path.exists():
        return source_path

    repo_id = str(source)
    effective_revision = _resolve_remote_revision(repo_id, revision)
    snapshot_download = _load_snapshot_download()
    destination = (
        Path(local_dir).expanduser()
        if local_dir is not None
        else Path(dest_root).expanduser() / repo_id.rstrip("/").rsplit("/", 1)[-1]
    )
    try:
        downloaded = snapshot_download(
            repo_id=repo_id,
            revision=effective_revision,
            local_dir=destination,
        )
    except Exception as exc:  # pragma: no cover - exercised through the real HF client.
        raise BooguTurboMlxError(
            f"Failed to download Hugging Face source {repo_id!r}: {exc}"
        ) from exc
    return Path(downloaded).expanduser()


def _load_snapshot_download() -> Callable[..., Any]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise BooguTurboMlxError(
            "Downloading Hugging Face sources requires huggingface-hub. "
            "Install `boogu-turbo-mlx[conversion]` or add `huggingface-hub`."
        ) from exc
    return snapshot_download
