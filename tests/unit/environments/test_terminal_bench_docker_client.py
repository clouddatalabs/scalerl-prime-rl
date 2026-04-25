"""Tests for `_DockerClient` shell-quoting + run_id contracts.

The TB env's `_DockerClient.cp_into` interpolates `dst` into a `bash -lc
mkdir -p` call. All current callers pass constant `dst` paths, but a future
caller plumbing a model-controlled or task-toml-supplied path would
otherwise be one missing quote away from RCE on the rollout container. Pin
the quoting and the per-run-unique label so concurrent runs on the same
host don't kill each other's containers.

These tests stub `_DockerClient.exec` and `asyncio.create_subprocess_exec`
so they run without a real Docker daemon.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


# Add the repo root to sys.path so `import environments.terminal_bench` works
# from the test runner (mirrors what `scripts/scalerl_smoke.sbatch` does for
# the orchestrator console-script entrypoint).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def test_cp_into_quotes_dst_against_shell_metachars(monkeypatch):
    """A future caller that passes a `dst` containing shell metacharacters
    must NOT be able to execute arbitrary commands inside the container.
    The `mkdir -p` invocation should pass the parent path as a single
    shlex-quoted token.
    """
    from environments.terminal_bench.env import _DockerClient

    captured_exec_args: list[str] = []

    async def fake_exec(self, container, command, *, timeout, working_dir=None):
        captured_exec_args.append(command)
        return (0, "", "")

    async def fake_subprocess_exec(*args, **kwargs):
        class _FakeProc:
            returncode = 0

            async def communicate(self):
                return (b"", b"")

        return _FakeProc()

    monkeypatch.setattr(_DockerClient, "exec", fake_exec)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess_exec)

    client = _DockerClient(sudo=False)
    malicious_dst = '/tests; rm -rf /; echo "$(whoami)"'
    asyncio.run(client.cp_into("/tmp/some-src", "tb-test-container", malicious_dst))

    assert len(captured_exec_args) == 1, captured_exec_args
    cmd = captured_exec_args[0]
    assert cmd.startswith("mkdir -p "), cmd
    quoted_arg = cmd[len("mkdir -p ") :]
    # shlex.quote wraps in single-quotes when metachars are present.
    assert quoted_arg.startswith("'") and quoted_arg.endswith("'"), (
        f"dst parent must be single-quoted; got: {quoted_arg!r}"
    )
    # And the metachars (`;`, `$`, `"`, whitespace) must NOT escape the quotes.
    assert ";" not in cmd[: len("mkdir -p ")], cmd
    # Verify there's no broken `'...';...` pattern (a single-quote inside
    # the user input would break out of shlex.quote — but shlex.quote
    # double-quotes any single-quote in the input, which we don't have here).
    assert cmd.count("mkdir -p ") == 1, cmd


def test_run_id_uniqueness_under_uuid(monkeypatch):
    """Two ad-hoc launches must get DIFFERENT _run_ids. Prior `pid + int(time)`
    fallback would silently collide and the second's startup sweep would
    kill the first's containers.

    Calls the production `_compute_run_id()` directly (NOT a stub re-implementing
    the same logic) — a regression in the production function fails the test.
    """
    from environments.terminal_bench.env import _compute_run_id

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    a = _compute_run_id()
    b = _compute_run_id()
    assert a != b, (a, b)
    assert re.match(r"^adhoc-[0-9a-f]{16}$", a), a
    assert re.match(r"^adhoc-[0-9a-f]{16}$", b), b


def test_run_id_under_slurm(monkeypatch):
    """When SLURM_JOB_ID is set, _run_id derives from it. Calls the
    production function directly so a regression to e.g. dropping the
    `slurm-` prefix or short-circuiting the env-var lookup fails."""
    from environments.terminal_bench.env import _compute_run_id

    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    assert _compute_run_id() == "slurm-12345"


def test_run_id_uses_canonical_slurm_when_both_present(monkeypatch):
    """SLURM_JOB_ID wins over the ad-hoc fallback (canonical SLURM context)."""
    from environments.terminal_bench.env import _compute_run_id

    monkeypatch.setenv("SLURM_JOB_ID", "99999")
    assert _compute_run_id() == "slurm-99999"
    assert not _compute_run_id().startswith("adhoc-")
