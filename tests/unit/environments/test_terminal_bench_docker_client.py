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


# ---- registry-pull mode -------------------------------------------------


def test_local_image_tag_registry_qualified_when_set():
    """`_local_image_tag` returns `<registry>/tb-local/<task>:latest` when
    registry_host is set, and the bare short form otherwise. This is the
    single helper every downstream caller (image_exists, run_detached,
    _resolved_tags lookup, _build_lock_path) goes through; touching it is
    sufficient to thread the registry namespace everywhere.
    """
    from environments.terminal_bench.env import (
        TerminalBenchLocalEnv,
        _TaskSpec,
    )

    spec = _TaskSpec(
        name="aimo-airline-departures",
        dir=Path("/tmp/fake/aimo"),
        instruction="x",
        docker_image="ghcr.io/.../aimo:latest",
    )

    # Registry mode
    env_reg = TerminalBenchLocalEnv.__new__(TerminalBenchLocalEnv)
    env_reg._registry_host = "rlgpu0:5000"
    assert env_reg._local_image_tag(spec) == "rlgpu0:5000/tb-local/aimo-airline-departures:latest"

    # Live-build mode (None)
    env_live = TerminalBenchLocalEnv.__new__(TerminalBenchLocalEnv)
    env_live._registry_host = None
    assert env_live._local_image_tag(spec) == "tb-local/aimo-airline-departures:latest"


def test_dockerclient_pull_raises_on_nonzero_exit(monkeypatch):
    """`_DockerClient.pull` raises RuntimeError on non-zero exit, surfacing
    stderr in the message so a registry-mode caller can act on the error.
    """
    from environments.terminal_bench.env import _DockerClient

    async def fake_subprocess_exec(*args, **kwargs):
        class _FakeProc:
            returncode = 1

            async def communicate(self):
                return (b"", b"unauthorized: registry auth required")

            def kill(self):
                pass

            async def wait(self):
                pass

        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess_exec)

    client = _DockerClient(sudo=False)
    with pytest.raises(RuntimeError, match="docker pull .* failed.*unauthorized"):
        asyncio.run(client.pull("rlgpu0:5000/tb-local/missing:latest"))


def test_dockerclient_pull_raises_on_timeout(monkeypatch):
    """A pull that exceeds the timeout is killed and surfaces a timeout error."""
    from environments.terminal_bench.env import _DockerClient

    async def fake_subprocess_exec(*args, **kwargs):
        class _FakeProc:
            returncode = None

            async def communicate(self):
                # Never returns — simulate a hung pull.
                await asyncio.sleep(60)
                return (b"", b"")

            def kill(self):
                self.returncode = -9

            async def wait(self):
                pass

        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess_exec)

    client = _DockerClient(sudo=False)
    with pytest.raises(RuntimeError, match="timed out after 1s"):
        asyncio.run(client.pull("rlgpu0:5000/tb-local/slow:latest", timeout=1))


def test_resolve_image_registry_mode_raises_on_pull_failure(monkeypatch):
    """In registry mode, a missing/unreachable image surfaces as a hard
    RuntimeError pointing the operator at the prebuild script — NOT a
    silent fallback to live build.
    """
    from environments.terminal_bench.env import (
        TerminalBenchLocalEnv,
        _DockerClient,
        _TaskSpec,
    )

    spec = _TaskSpec(
        name="aimo-airline-departures",
        dir=Path("/tmp/fake/aimo"),
        instruction="x",
        docker_image="ghcr.io/.../aimo:latest",
    )

    env = TerminalBenchLocalEnv.__new__(TerminalBenchLocalEnv)
    env._registry_host = "rlgpu0:5000"
    env._build_local_images = True
    env._docker = _DockerClient(sudo=False)
    env._build_locks = {}
    env._resolved_tags = {}

    async def fake_image_exists(self, image):
        return False

    async def fake_pull(self, image, *, timeout=600):
        raise RuntimeError("connection refused: rlgpu0:5000")

    def fake_dockerfile_path(self, _spec):
        return Path("/tmp/fake/aimo/environment/Dockerfile")

    monkeypatch.setattr(_DockerClient, "image_exists", fake_image_exists)
    monkeypatch.setattr(_DockerClient, "pull", fake_pull)
    monkeypatch.setattr(TerminalBenchLocalEnv, "_dockerfile_path", fake_dockerfile_path)

    with pytest.raises(RuntimeError, match="docker pull .* failed.*"
                       "Run.*scripts/prebuild_tb_images.sh"):
        asyncio.run(env._resolve_image(spec))


def test_resolve_image_registry_mode_caches_after_pull(monkeypatch):
    """A successful pull caches the registry-qualified tag and avoids
    re-pulling on subsequent calls — `image_exists` short-circuits.
    """
    from environments.terminal_bench.env import (
        TerminalBenchLocalEnv,
        _DockerClient,
        _TaskSpec,
    )

    spec = _TaskSpec(
        name="aimo-airline-departures",
        dir=Path("/tmp/fake/aimo"),
        instruction="x",
        docker_image="ghcr.io/.../aimo:latest",
    )

    env = TerminalBenchLocalEnv.__new__(TerminalBenchLocalEnv)
    env._registry_host = "rlgpu0:5000"
    env._build_local_images = True
    env._docker = _DockerClient(sudo=False)
    env._build_locks = {}
    env._resolved_tags = {}

    pull_count = 0

    async def fake_image_exists(self, image):
        # First call: not yet pulled. Subsequent calls (after pull): exists.
        return pull_count > 0

    async def fake_pull(self, image, *, timeout=600):
        nonlocal pull_count
        pull_count += 1

    def fake_dockerfile_path(self, _spec):
        return Path("/tmp/fake/aimo/environment/Dockerfile")

    monkeypatch.setattr(_DockerClient, "image_exists", fake_image_exists)
    monkeypatch.setattr(_DockerClient, "pull", fake_pull)
    monkeypatch.setattr(TerminalBenchLocalEnv, "_dockerfile_path", fake_dockerfile_path)

    expected_tag = "rlgpu0:5000/tb-local/aimo-airline-departures:latest"
    tag1 = asyncio.run(env._resolve_image(spec))
    assert tag1 == expected_tag
    assert pull_count == 1

    tag2 = asyncio.run(env._resolve_image(spec))
    assert tag2 == expected_tag
    assert pull_count == 1, "second resolve must hit fast-cache, not pull again"


def test_exec_runs_plain_bash_lc(monkeypatch):
    """`_DockerClient.exec` runs `docker exec ... bash -lc <command>` —
    no setsid wrapper, no pidfile. The earlier setsid wrapping (commits
    cddd0d58e + 6c9b11bf2) was reverted because empirically (job 738)
    it correlated with test.sh failing to write reward.txt across all
    rollouts; the simpler bash -lc form was the one that worked in
    jobs 715/717.
    """
    from environments.terminal_bench.env import _DockerClient

    captured: dict = {}

    async def fake_subprocess_exec(*args, **kwargs):
        captured["args"] = args

        class _FakeProc:
            returncode = 0

            async def communicate(self):
                return (b"hi", b"")

        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess_exec)

    client = _DockerClient(sudo=False)
    asyncio.run(client.exec("tb-test", "echo hello", timeout=10))

    args = captured["args"]
    # docker exec ... <container> bash -lc <command>
    assert "exec" in args
    assert "bash" in args
    assert "-lc" in args
    assert args[-1] == "echo hello", f"command should be passed verbatim; got: {args[-1]!r}"
    assert "setsid" not in args, f"setsid should NOT wrap exec — was reverted: {args}"


def test_exec_returns_124_on_timeout(monkeypatch):
    """On asyncio.TimeoutError, exec returns exit_code=124 to match
    timeout(1) semantics. The local docker CLI process is killed; the
    in-container shell may keep running until the container is reaped
    at rollout end (`@vf.cleanup` -> `remove_force`).
    """
    from environments.terminal_bench.env import _DockerClient

    async def fake_subprocess_exec(*args, **kwargs):
        class _FakeProc:
            returncode = None

            async def communicate(self):
                # Hang to force timeout.
                await asyncio.sleep(60)
                return (b"", b"")

            def kill(self):
                self.returncode = -9

            async def wait(self):
                pass

        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess_exec)

    client = _DockerClient(sudo=False)
    rc, stdout, stderr = asyncio.run(
        client.exec("tb-test", "sleep 60", timeout=1)
    )

    assert rc == 124, "timeout returns 124 to match timeout(1) semantics"
    assert "timed out after 1s" in stderr
