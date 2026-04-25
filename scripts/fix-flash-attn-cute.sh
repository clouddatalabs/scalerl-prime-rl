#!/bin/bash
# Both `flash-attn` (FA2) and `flash-attn-cute` (FA4) ship a `flash_attn/cute/`
# sub-package.  The one from `flash-attn` is a tiny stub, while the one from
# `flash-attn-cute` contains the real FA4 kernels (>1000 lines in interface.py).
# When both extras are installed, `uv sync` may install `flash-attn` *after*
# `flash-attn-cute`, causing the stub to overwrite the real module.
#
# This script reinstalls `flash-attn-cute` so the real module wins.
# Run it after `uv sync` if you have both flash-attn and flash-attn-cute extras enabled.

set -e

# Activate the repo's venv so `python` and `uv pip` operate on it. Without this,
# running the script before activation resolves to system Python (which doesn't
# have flash_attn at all) and fails with a misleading "0 lines" error.
# Also re-activate if a DIFFERENT venv is currently active — the install would
# otherwise land in someone else's venv silently.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ ! -f "$REPO_ROOT/.venv/bin/activate" ]; then
    echo "Error: $REPO_ROOT/.venv missing; run 'uv sync --extra all' on the login node first." >&2
    exit 1
fi
if [ "${VIRTUAL_ENV:-}" != "$REPO_ROOT/.venv" ]; then
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        echo "Note: deactivating $VIRTUAL_ENV to operate on $REPO_ROOT/.venv" >&2
        deactivate 2>/dev/null || true
    fi
    # shellcheck source=/dev/null
    source "$REPO_ROOT/.venv/bin/activate"
fi

# Serialize concurrent invocations against the same venv so two parallel slurm
# submits don't fight over the wheel install. 5-minute timeout so a stale lock
# from a crashed sibling doesn't hang the job indefinitely.
LOCK_FD=9
LOCK_FILE="$REPO_ROOT/.venv/.fix-flash-attn-cute.lock"
exec {LOCK_FD}> "$LOCK_FILE"
if ! flock -x -w 300 "$LOCK_FD"; then
    echo "Error: timed out waiting for flock on $LOCK_FILE." >&2
    echo "       Another submit may have crashed mid-install. Inspect & remove if safe." >&2
    exit 1
fi

echo "Reinstalling flash-attn-cute to fix namespace conflict with flash-attn..."
# Read the FA4 git rev from `pyproject.toml` rather than hardcoding it here —
# previously this script's pin and pyproject's pin were two independent
# constants. A future bump of pyproject (with matching `uv lock` update) would
# silently leave this script reinstalling the OLD rev, downgrading FA4
# whenever the script ran (which the sbatch + Dockerfile both invoke after
# `uv sync` as defense-in-depth). The line-count check below would still
# pass on the wrong rev (any real FA4 has >1000 lines), so the downgrade
# would surface only as wrong attention numerics deep into a training run.
FA4_REV=$(
    python -c "
import sys, tomllib
with open('$REPO_ROOT/pyproject.toml', 'rb') as f:
    cfg = tomllib.load(f)
src = cfg.get('tool', {}).get('uv', {}).get('sources', {}).get('flash-attn-4')
if not isinstance(src, dict) or 'rev' not in src:
    print('ERROR: pyproject.toml [tool.uv.sources.flash-attn-4].rev not found', file=sys.stderr)
    sys.exit(1)
print(src['rev'])
"
) || exit 1

echo "Pinning to flash-attn-4 rev $FA4_REV (read from pyproject.toml)"
uv pip install --reinstall --no-deps \
    "flash-attn-4 @ git+https://github.com/Dao-AILab/flash-attention.git@${FA4_REV}#subdirectory=flash_attn/cute"

# Verify installation
LINES=$(wc -l < "$(python -c 'import flash_attn.cute.interface as m; print(m.__file__)')")
if [ "$LINES" -gt 1000 ]; then
    echo "Success: flash-attn-cute interface.py has $LINES lines (correct version)"
else
    echo "Error: flash-attn-cute interface.py has only $LINES lines (wrong version)"
    exit 1
fi
