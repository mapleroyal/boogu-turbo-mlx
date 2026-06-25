#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

find_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    printf '%s\n' "$PYTHON"
    return 0
  fi

  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
      then
        printf '%s\n' "$candidate"
        return 0
      fi
    fi
  done

  return 1
}

PYTHON_BIN="$(find_python)" || {
  printf 'Python 3.10 or newer is required. Install Python, then run ./setup.sh again.\n' >&2
  exit 1
}

clear_macos_hidden_flags() {
  if command -v chflags >/dev/null 2>&1; then
    chflags -R nohidden .venv 2>/dev/null || true
  fi
}

pip_install() {
  if ! python -m pip install "$@"; then
    printf '\nInstall failed. This is often transient; re-run ./setup.sh to retry.\n' >&2
    return 1
  fi
}

"$PYTHON_BIN" - <<'PY'
import platform
import sys

if platform.system() != "Darwin":
    raise SystemExit("macOS is required for MLX setup.")
if platform.machine().lower() not in {"arm64", "aarch64"}:
    raise SystemExit("Apple Silicon is required for MLX setup.")
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10 or newer is required.")
PY

"$PYTHON_BIN" -m venv .venv
clear_macos_hidden_flags
. .venv/bin/activate

pip_install --upgrade pip setuptools wheel
clear_macos_hidden_flags
pip_install '.[runtime,conversion]'
clear_macos_hidden_flags

python -m boogu_turbo_mlx setup "$@"
