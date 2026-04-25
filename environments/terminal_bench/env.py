"""Harbor-format terminal-bench tasks driven by a local Docker daemon.

Design notes (see also ``environments.terminal_bench/__init__.py``):

- Task format is the upstream Harbor layout: one directory per task
  containing ``task.toml``, ``instruction.md``, ``environment/Dockerfile``
  (base image pinned via ``task.toml::[environment].docker_image``),
  and ``tests/test.sh`` which writes a ``/logs/verifier/reward.txt``
  (float -- typically 0 or 1). The 28 task fixtures imported from
  upstream Terminal-Bench / opencode_harbor live under
  ``environments/terminal_bench/tasks/``; see ``THIRD_PARTY_NOTICES.md``
  for attribution.
- The upstream ``HarborEnv`` spins up a Prime Intellect cloud sandbox
  per rollout and tunnels the agent's OpenAI traffic back to the
  training host via frpc. That's great for cloud agents but costs
  money, requires ``PRIME_API_KEY``, and wants outbound network -- none
  of which we have set up on this cluster. We bypass the sandbox +
  tunnel layer entirely:

    orchestrator -> vLLM -> assistant tool_call
                                 |
                                 v
                    docker exec <rollout_container> ...
                                 |
                                 v
                            stdout/stderr -> next turn

  Reward is still computed by copying the task's ``tests/`` directory
  into the container and running ``bash test.sh``, matching the Harbor
  reward semantics byte-for-byte.
- Per-rollout state is carried through ``vf.StatefulToolEnv``'s
  ``update_tool_args`` hook, which injects the docker container id
  into the ``shell`` tool call without exposing it to the model. This
  is the same pattern ``verifiers.envs.sandbox_env.SandboxEnv`` uses
  for its ``bash`` tool; we just swap the ``prime_sandboxes`` backend
  for async docker subprocess calls.
- We expect a host docker daemon reachable from the process -- either
  directly (bare host) or via ``/var/run/docker.sock`` bind-mounted
  into the prime-rl container. The ``_DockerClient`` helper auto-detects
  whether ``sudo -n`` is required (dev host vs prime-rl container as
  root).

Usage sketch for the orchestrator config::

    [[orchestrator.env]]
    id = "environments.terminal_bench"
    args = { tasks = ["analyze-access-logs"], task_root = "environments/terminal_bench/tasks" }
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import tomllib
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import verifiers as vf
from datasets import Dataset
from verifiers.types import State

_logger = logging.getLogger(__name__)


# Hard cap on shell / test output fed back to the model. Qwen3-8B tokenizes
# ~3-4 chars per token; 8KiB keeps a single tool response under ~2.5k tokens
# which fits inside the shipped seq_len=8192 with headroom for prompt + reasoning.
_MAX_OUTPUT_CHARS = 8192

# Default timeout for each shell tool call. hello-world's commands are
# trivial but e.g. ``apt-get update`` inside test.sh is slow, so test
# execution uses its own longer timeout (see ``_TEST_TIMEOUT_SECONDS``).
_DEFAULT_COMMAND_TIMEOUT_SECONDS = 30

# Timeout for ``bash test.sh`` at post-rollout reward time. Covers the
# Harbor hello-world test (apt-get + uv install + pytest ≈ 20-30s) plus
# headroom for the heavier opencode_harbor tasks.
_TEST_TIMEOUT_SECONDS = 300


def _compute_run_id() -> str:
    """Per-run-unique ID for the TB env's container labels.

    `SLURM_JOB_ID` is the canonical SLURM context (slurm guarantees
    uniqueness across jobs). For ad-hoc launches we use a 16-hex-char
    uuid (64 bits, ~1e-9 collision at 100k containers) — NOT
    `pid + integer-second time`, which would silently collide on two
    ad-hoc launches in the same wall-clock second on the same host
    and cause the second's startup sweep to kill the first's
    containers. Module-level so tests pin the production path
    rather than re-implementing it in a stub subclass.
    """
    slurm_id = os.environ.get("SLURM_JOB_ID")
    if slurm_id:
        return f"slurm-{slurm_id}"
    return f"adhoc-{uuid.uuid4().hex[:16]}"


@dataclass(frozen=True)
class _TaskSpec:
    name: str
    dir: Path
    instruction: str
    docker_image: str


def _load_task_spec(task_dir: Path) -> _TaskSpec:
    """Parse one Harbor-format task directory into a typed spec.

    Raises if required files are missing so the orchestrator preflight
    fails loudly rather than silently skipping tasks at rollout time.
    """
    task_toml = task_dir / "task.toml"
    instruction_md = task_dir / "instruction.md"
    if not task_toml.exists() or not instruction_md.exists():
        raise FileNotFoundError(
            f"Harbor task at {task_dir} is missing task.toml or instruction.md"
        )

    with task_toml.open("rb") as fh:
        config = tomllib.load(fh)
    image = (config.get("environment") or {}).get("docker_image")
    if not image:
        raise ValueError(
            f"Harbor task {task_dir.name} is missing [environment].docker_image"
        )

    instruction = instruction_md.read_text(encoding="utf-8").strip()
    return _TaskSpec(
        name=task_dir.name,
        dir=task_dir,
        instruction=instruction,
        docker_image=image,
    )


_BUILD_LOCK_DIR = Path(os.environ.get("TB_BUILD_LOCK_DIR", "/tmp/tb-build-locks"))


def _build_lock_path(tag: str) -> Path:
    """Return a filesystem path for a cross-process advisory lock keyed to
    a Docker image tag. We hash the tag — `tb-local/<task>:latest` contains
    `/` and `:` which are filename-illegal on some filesystems — and prefix
    with a short readable slug so a stuck lock is greppable in `lsof`.
    """
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", tag)[:48]
    digest = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:12]
    return _BUILD_LOCK_DIR / f"{safe}-{digest}.lock"


@asynccontextmanager
async def _flock_exclusive(path: Path):
    """Acquire an OS-level exclusive advisory lock at ``path`` for the
    duration of the context. fcntl.flock is process-scoped (NOT
    thread-scoped) so this serializes peer Python PROCESSES, which the
    in-class ``asyncio.Lock`` cannot — verifiers' ZMQEnvServer spawns
    ``num_workers`` separate worker processes via ``mp.spawn`` and each
    worker has its own ``TerminalBenchLocalEnv`` and its own
    ``_build_locks`` dict, so without an OS-level lock, concurrent peer
    workers race ``docker build`` on the same tag and corrupt BuildKit's
    snapshotter cache (visible as cascading "parent snapshot does not
    exist" / "failed to extract layer / content digest not found"
    errors that fail-cascade across unrelated tasks).

    Run the lock-acquire/release on a thread because ``fcntl.flock`` is
    blocking; the event loop must continue serving other rollouts while
    we wait for a peer worker's build to finish (qemu-tasks can be ~15
    min cold). Override the lock dir with ``TB_BUILD_LOCK_DIR`` if /tmp
    is not shared across the relevant peer processes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = await asyncio.to_thread(os.open, str(path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        await asyncio.to_thread(fcntl.flock, fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            await asyncio.to_thread(fcntl.flock, fd, fcntl.LOCK_UN)
        finally:
            await asyncio.to_thread(os.close, fd)


class _DockerClient:
    """Tiny async wrapper around the ``docker`` CLI.

    We shell out instead of using ``docker-py`` for three reasons:
      1. The prime-rl image doesn't ship ``docker-py``; the CLI binary
         is on PATH everywhere we run.
      2. ``docker exec`` / ``docker cp`` semantics are stable across
         daemon versions; async subprocess is enough.
      3. Keeps the failure modes obvious -- exit code + stderr is what
         operators already know how to read.
    """

    def __init__(self, *, sudo: bool | None = None) -> None:
        self._binary = shutil.which("docker") or "/usr/bin/docker"
        # Prefer direct socket access when the current process can
        # open ``/var/run/docker.sock`` (either because it's root, or
        # because the launcher passed ``--group-add <docker-gid>``
        # so the container user inherits host docker-group
        # membership). Fall back to ``sudo -n`` only on dev hosts
        # where the user is unprivileged AND ``sudo`` is actually on
        # PATH and passwordless. The prime-rl image doesn't ship
        # ``sudo``, so falling through to a ``["sudo", "-n", "docker", ...]``
        # invocation there would crash every rollout with an opaque
        # ``FileNotFoundError: 'sudo'`` instead of a clear "mount
        # /var/run/docker.sock with --group-add docker" message.
        # We check the socket directly rather than probing with
        # ``docker info`` to avoid ``asyncio.run`` nesting issues
        # (this __init__ is called from inside the verifiers loop).
        if sudo is None:
            sock = "/var/run/docker.sock"
            if os.access(sock, os.R_OK | os.W_OK):
                sudo = False
            elif shutil.which("sudo") is not None:
                sudo = True
            else:
                # Socket unreachable AND no sudo binary — surface the
                # actionable error here, not on every rollout.
                raise RuntimeError(
                    "_DockerClient cannot reach /var/run/docker.sock and 'sudo' "
                    "is not on PATH (this is the prime-rl container's default). "
                    "Either bind-mount /var/run/docker.sock with '--group-add "
                    "<host-docker-gid>' on the launcher, or run from a host shell "
                    "where the user has docker-group membership."
                )
        self._use_sudo = sudo

    def _prefix(self) -> list[str]:
        return ["sudo", "-n", self._binary] if self._use_sudo else [self._binary]

    async def run_detached(
        self,
        image: str,
        *,
        name: str,
        start_command: list[str] | None = None,
        labels: dict[str, str] | None = None,
    ) -> None:
        """``docker run -d --name <name> <image> <start_command>``.

        Uses ``--rm`` so the container auto-cleans if we crash before
        ``remove_force``. ``--init`` ensures zombie reaping for
        ``tail -f``-style start commands. ``labels`` forwards to
        ``--label`` so sweep_stale_containers can find siblings.

        Deliberately does NOT accept caller-controlled `extra_args`: a
        future config that plumbed `["--privileged", "-v", "/:/host"]`
        through here would silently break the rollout sandbox. If a
        caller legitimately needs extra docker flags, add them here as
        an explicit, allow-listed parameter (e.g. `network_mode`,
        `cpu_limit`) so the surface area stays auditable.
        """
        label_flags: list[str] = []
        for k, v in (labels or {}).items():
            label_flags += ["--label", f"{k}={v}"]
        cmd = [
            *self._prefix(),
            "run", "-d", "--rm", "--init",
            "--name", name,
            *label_flags,
            image,
            *(start_command or ["tail", "-f", "/dev/null"]),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"docker run failed (exit={proc.returncode}) for image={image!r} "
                f"name={name!r}: {stderr.decode(errors='replace').strip()[:500]}"
            )

    async def exec(
        self,
        container: str,
        command: str,
        *,
        timeout: int,
        working_dir: str | None = None,
    ) -> tuple[int, str, str]:
        """Run ``bash -lc <command>`` inside ``container``.

        Returns ``(exit_code, stdout, stderr)``. Always captures; never
        raises for non-zero exit (we want to show errors to the model).
        Timeouts return exit_code=124 to match ``timeout(1)`` semantics.

        ``working_dir`` forwards to ``docker exec -w``; omit to let the
        container's default WORKDIR apply. We use this to root the
        agent's shell at ``/app`` so relative paths in the task
        instruction resolve the way Harbor's agents expect.
        """
        cmd = [*self._prefix(), "exec"]
        if working_dir:
            cmd += ["-w", working_dir]
        cmd += [container, "bash", "-lc", command]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            # Also kill whatever's still running inside the container
            # for this exec chain so subsequent calls aren't serialized
            # behind a stuck process.
            return 124, "", f"Command timed out after {timeout}s"
        return (
            proc.returncode or 0,
            stdout_b.decode(errors="replace"),
            stderr_b.decode(errors="replace"),
        )

    async def cp_into(self, src: str | os.PathLike[str], container: str, dst: str) -> None:
        """``docker cp <src> <container>:<dst>`` (creates parent dirs if needed).

        ``src`` is forwarded as a string so callers can preserve the
        ``dir/.`` trailing form that tells docker to copy directory
        *contents* rather than the directory itself (``pathlib.Path``
        normalizes ``/.`` away, which is subtly wrong here). ``dst`` is
        interpreted by the docker daemon; pass an absolute path.
        """
        src_str = os.fspath(src)

        # docker cp refuses a missing parent dir; create it first. Quote
        # the path because it's interpolated into a `bash -lc` command —
        # all current callers pass constant `dst` values, but a future
        # caller plumbing a model-controlled or task-toml-supplied path
        # would otherwise be one shell-injection away from RCE on the
        # rollout container.
        parent = os.path.dirname(dst.rstrip("/")) or "/"
        ensure = f"mkdir -p {shlex.quote(parent)}"
        exit_code, _, stderr = await self.exec(container, ensure, timeout=10)
        if exit_code != 0:
            raise RuntimeError(
                f"mkdir parent for {dst!r} in {container!r} failed: {stderr!r}"
            )

        cmd = [*self._prefix(), "cp", src_str, f"{container}:{dst}"]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr_b = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"docker cp {src} -> {container}:{dst} failed "
                f"(exit={proc.returncode}): {stderr_b.decode(errors='replace').strip()[:500]}"
            )

    async def remove_force(self, container: str) -> None:
        """Best-effort ``docker rm -f <container>``. Never raises."""
        cmd = [*self._prefix(), "rm", "-f", container]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr_b = await proc.communicate()
        if proc.returncode != 0:
            # Already gone is fine; log at debug to avoid noise.
            _logger.debug(
                "docker rm -f %s failed (exit=%d): %s",
                container,
                proc.returncode,
                stderr_b.decode(errors="replace").strip()[:200],
            )

    async def ps_by_label(self, key: str, value: str) -> list[str]:
        """Return container IDs with ``--label <key>=<value>``.

        Used at env startup to sweep stale rollout containers left
        behind by a previous crash / scancel (we observed 3 qemu +
        sqlite containers surviving the orchestrator's ``Cancelling
        active tasks`` shutdown because the ``@vf.cleanup`` task was
        cancelled mid-``test.sh``). Returns an empty list if docker
        isn't reachable or there are no matches; never raises.
        """
        cmd = [
            *self._prefix(),
            "ps", "-aq",  # also pick up exited containers for the sweep
            "--filter", f"label={key}={value}",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout_b, _ = await proc.communicate()
        if proc.returncode != 0:
            return []
        return [cid for cid in stdout_b.decode().split() if cid]

    async def image_exists(self, image: str) -> bool:
        """``docker image inspect`` exit=0 iff the tag is present locally."""
        cmd = [*self._prefix(), "image", "inspect", image]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        return proc.returncode == 0

    async def rmi(self, image: str) -> None:
        """``docker rmi -f <image>``. Used by the BuildKit-corruption recovery
        path to invalidate a tagged-but-broken image so the next build goes
        from scratch. ``-f`` removes even if a stopped container references
        it; running containers are NOT killed.
        """
        cmd = [*self._prefix(), "rmi", "-f", image]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"docker rmi {image!r} failed: "
                f"{stderr.decode(errors='replace').strip()[:300]}"
            )

    # BuildKit corruption signatures we recover from with a --no-cache retry.
    # When a previous racing build left dangling parent snapshots in the
    # snapshotter's metadata, every subsequent cached build of the same
    # Dockerfile step fails with "parent snapshot ... does not exist". The
    # cache is poisoned at the daemon layer; we can't `flock` our way out.
    # Detect the symptom and retry with --no-cache to rebuild from scratch
    # against the FROM layer, which restores the snapshotter to a healthy
    # state for that lineage. Once a clean rebuild lands, subsequent cached
    # builds succeed normally.
    _BUILDKIT_CORRUPTION_MARKERS = (
        "parent snapshot",
        "failed to extract layer",
        "failed to get reader from content store",
    )

    async def build(
        self,
        *,
        tag: str,
        context_dir: str | os.PathLike[str],
        dockerfile: str | os.PathLike[str] | None = None,
        timeout: int = 900,
        log_path: str | os.PathLike[str] | None = None,
    ) -> None:
        """``docker build -t <tag> [-f <dockerfile>] <context_dir>``.

        Build output is large (apt-get install chatter, package downloads,
        multi-hundred-MB layers) so we stream it to ``log_path`` rather
        than buffering it in the orchestrator's memory. The verifier log
        for this env typically lives under
        ``<run>/logs/envs/train/environments.terminal_bench/`` which
        the operator can tail if a build stalls.

        Raises on non-zero exit; stderr tail is surfaced in the exception
        so rollout failures point at the actual missing package rather
        than "docker run: image not found". On detected BuildKit cache
        corruption (see ``_BUILDKIT_CORRUPTION_MARKERS``), retry once
        with ``--no-cache`` to bypass the poisoned cache layer.
        """
        await self._build_one(
            tag=tag,
            context_dir=context_dir,
            dockerfile=dockerfile,
            timeout=timeout,
            log_path=log_path,
            no_cache=False,
        )

    async def _build_one(
        self,
        *,
        tag: str,
        context_dir: str | os.PathLike[str],
        dockerfile: str | os.PathLike[str] | None,
        timeout: int,
        log_path: str | os.PathLike[str] | None,
        no_cache: bool,
    ) -> None:
        cmd = [*self._prefix(), "build", "-t", tag]
        if no_cache:
            cmd += ["--no-cache"]
        if dockerfile is not None:
            cmd += ["-f", os.fspath(dockerfile)]
        cmd += [os.fspath(context_dir)]

        if log_path is not None:
            log_file = open(log_path, "ab")
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=log_file,
                    stderr=asyncio.subprocess.STDOUT,
                )
                try:
                    rc = await asyncio.wait_for(proc.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    raise RuntimeError(
                        f"docker build for {tag!r} timed out after {timeout}s "
                        f"(context={context_dir}); see {log_path}"
                    )
                if rc != 0:
                    log_tail = ""
                    try:
                        with open(log_path, "rb") as f:
                            f.seek(0, os.SEEK_END)
                            size = f.tell()
                            f.seek(max(0, size - 8192))
                            log_tail = f.read().decode(errors="replace")
                    except OSError:
                        pass
                    if not no_cache and self._is_buildkit_corruption(log_tail):
                        _logger.warning(
                            "terminal_bench_local: BuildKit cache corruption "
                            "detected for %s; retrying with --no-cache",
                            tag,
                        )
                        await self._build_one(
                            tag=tag,
                            context_dir=context_dir,
                            dockerfile=dockerfile,
                            timeout=timeout,
                            log_path=log_path,
                            no_cache=True,
                        )
                        return
                    raise RuntimeError(
                        f"docker build for {tag!r} failed (exit={rc}); "
                        f"see {log_path} for the full log\n{log_tail[-2000:]}"
                    )
            finally:
                log_file.close()
        else:
            # No log path -- capture to an in-memory buffer and surface
            # the tail on failure. Fine for small/cached builds.
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                stdout_b, _ = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise RuntimeError(
                    f"docker build for {tag!r} timed out after {timeout}s "
                    f"(context={context_dir})"
                )
            if proc.returncode != 0:
                output = stdout_b.decode(errors="replace")
                if not no_cache and self._is_buildkit_corruption(output):
                    _logger.warning(
                        "terminal_bench_local: BuildKit cache corruption "
                        "detected for %s; retrying with --no-cache",
                        tag,
                    )
                    await self._build_one(
                        tag=tag,
                        context_dir=context_dir,
                        dockerfile=dockerfile,
                        timeout=timeout,
                        log_path=log_path,
                        no_cache=True,
                    )
                    return
                tail = output.strip()[-2000:]
                raise RuntimeError(
                    f"docker build for {tag!r} failed (exit={proc.returncode}):\n{tail}"
                )

    @classmethod
    def _is_buildkit_corruption(cls, text: str) -> bool:
        return any(marker in text for marker in cls._BUILDKIT_CORRUPTION_MARKERS)


class TerminalBenchLocalEnv(vf.StatefulToolEnv):
    """StatefulToolEnv that runs Harbor tasks in per-rollout Docker containers."""

    # How long to allow ``docker build`` per task before giving up. qemu
    # tasks download an ~800 MB Alpine ISO + install qemu/grub/telnet, so
    # ~15 min is the realistic worst case on a cold build cache. Override
    # via ``build_timeout_seconds`` if you're on a slow-network host.
    _DEFAULT_BUILD_TIMEOUT_SECONDS = 15 * 60

    def __init__(
        self,
        task_specs: list[_TaskSpec],
        *,
        max_turns: int = 10,
        command_timeout_seconds: int = _DEFAULT_COMMAND_TIMEOUT_SECONDS,
        test_timeout_seconds: int = _TEST_TIMEOUT_SECONDS,
        docker_client: _DockerClient | None = None,
        build_local_images: bool = True,
        build_timeout_seconds: int | None = None,
        build_log_dir: str | os.PathLike[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(max_turns=max_turns, **kwargs)
        self._task_specs = {spec.name: spec for spec in task_specs}
        self._command_timeout_seconds = command_timeout_seconds
        self._test_timeout_seconds = test_timeout_seconds
        self._docker = docker_client or _DockerClient()

        # ---- local image build plumbing ------------------------------
        # Harbor task.toml files point at ``us-central1-docker.pkg.dev/
        # prime-intellect-platform/prod-sandbox/alexgshaw/<task>:<date>``
        # which requires GCR auth we don't have in this cluster. Every
        # task ships its source ``environment/Dockerfile`` though, so
        # when ``build_local_images`` is True we ignore the registry tag
        # and produce ``tb-local/<task>:latest`` from the local context.
        # A per-tag asyncio.Lock prevents parallel rollouts from racing
        # on the same build; a ``_resolved_tags`` cache skips the
        # ``image inspect`` probe after the first successful resolve.
        self._build_local_images = build_local_images
        self._build_timeout_seconds = (
            build_timeout_seconds or self._DEFAULT_BUILD_TIMEOUT_SECONDS
        )
        self._build_log_dir: Path | None = (
            Path(build_log_dir) if build_log_dir is not None else None
        )
        if self._build_log_dir is not None:
            self._build_log_dir.mkdir(parents=True, exist_ok=True)
        self._build_locks: dict[str, asyncio.Lock] = {}
        self._resolved_tags: dict[str, str] = {}

        # ``setup_state`` triggers a one-shot sweep of stale containers from
        # THIS run on the first rollout. We do it lazily because ``__init__``
        # isn't async; doing it eagerly would need ``asyncio.run`` which the
        # verifiers harness forbids (we're already inside its event loop). A
        # lock guards the flag so concurrent first rollouts don't race the
        # sweep.
        #
        # The sweep filters by `prime_rl_tb_run_id` (per-run-unique) — NOT the
        # generic `prime_rl_tb_env=1` tag — so two concurrent runs on the same
        # host don't `docker rm -f` each other's live rollout containers. The
        # run id is `SLURM_JOB_ID` if set (canonical SLURM context), otherwise
        # PID + start time (good enough for ad-hoc launches).
        self._stale_sweep_lock = asyncio.Lock()
        self._stale_sweep_done = False
        self._run_id = _compute_run_id()

        # ``_container_id`` is injected by ``update_tool_args`` on every
        # call; advertising it in the model-visible signature would just
        # tempt the policy into copying hallucinated ids between turns.
        self.add_tool(self.shell, args_to_skip=["_container_id"])

    # ----- image resolution -----------------------------------------------

    def _local_image_tag(self, spec: _TaskSpec) -> str:
        # Deterministic, namespace-isolated tag -- ``tb-local/...`` keeps
        # our images visually grouped in ``docker images`` and avoids
        # colliding with anything users might have pulled manually.
        return f"tb-local/{spec.name}:latest"

    def _dockerfile_path(self, spec: _TaskSpec) -> Path | None:
        candidate = spec.dir / "environment" / "Dockerfile"
        return candidate if candidate.is_file() else None

    def _tb_labels(self, task_name: str) -> dict[str, str]:
        """Labels stamped on every rollout container.

        `prime_rl_tb_env=1` is the discoverability label (find any tb
        container on this host); `prime_rl_tb_run_id={run}` is the
        per-run-unique label the startup sweep filters by, so concurrent
        runs on the same host don't kill each other. `prime_rl_tb_task`
        is the task name for human-greppable `docker ps` output.
        """
        return {
            "prime_rl_tb_env": "1",
            "prime_rl_tb_run_id": self._run_id,
            "prime_rl_tb_task": task_name,
        }

    async def _invalidate_image(self, spec: _TaskSpec, tag: str) -> None:
        """Drop the cached tag and best-effort `docker rmi` it so the next
        `_resolve_image` call goes through the build path again. Used when
        `docker run` fails with a BuildKit-cache-corruption signature on a
        tagged-but-broken image (`image_exists()` returns True but the run
        fails to extract a layer). Without invalidation, every subsequent
        rollout short-circuits straight back to the broken tag.
        """
        self._resolved_tags.pop(spec.name, None)
        try:
            await self._docker.rmi(tag)
        except Exception as e:
            _logger.warning(
                "terminal_bench_local: docker rmi %s failed: %r (continuing — "
                "the next rebuild will overwrite the tag anyway)",
                tag,
                e,
            )

    async def _resolve_image(self, spec: _TaskSpec) -> str:
        """Return a ``docker run``-able tag for ``spec``.

        Fast path: tag already resolved this process lifetime.
        Build path: local Dockerfile exists, tag is missing locally,
        we're allowed to build -> ``docker build`` under a per-tag
        in-process asyncio.Lock AND an OS-level flock (the asyncio.Lock
        alone does NOT serialize peer worker processes spawned by
        verifiers' ZMQEnvServer — see ``_flock_exclusive`` docstring).
        Fallback: no local Dockerfile -> trust ``task.toml::docker_image``
        (still subject to daemon auth when ``docker run`` pulls it).
        """
        if spec.name in self._resolved_tags:
            return self._resolved_tags[spec.name]

        dockerfile = self._dockerfile_path(spec)
        if not self._build_local_images or dockerfile is None:
            # No local build option; use the upstream tag as-is. This is
            # the right behaviour for e.g. hello-world where task.toml
            # already points at ``python:3.11-slim`` which is public.
            self._resolved_tags[spec.name] = spec.docker_image
            return spec.docker_image

        local_tag = self._local_image_tag(spec)
        lock = self._build_locks.setdefault(spec.name, asyncio.Lock())
        async with lock:
            # Re-check inside the in-process lock in case a peer
            # coroutine in this process already built while we were
            # waiting.
            if spec.name in self._resolved_tags:
                return self._resolved_tags[spec.name]
            if await self._docker.image_exists(local_tag):
                self._resolved_tags[spec.name] = local_tag
                return local_tag

            # Cross-process serialization. Without this, peer worker
            # processes (verifiers spawns ``num_workers`` ≈ 3 for the
            # shipped TB train batch) race on ``docker build`` of the
            # same tag and corrupt BuildKit's snapshotter cache. Hold
            # the flock across both the existence re-check and the
            # build itself so a peer that just finished can short-circuit.
            async with _flock_exclusive(_build_lock_path(local_tag)):
                if await self._docker.image_exists(local_tag):
                    self._resolved_tags[spec.name] = local_tag
                    return local_tag

                log_path: Path | None = None
                if self._build_log_dir is not None:
                    log_path = self._build_log_dir / f"{spec.name}.build.log"

                _logger.info(
                    "terminal_bench_local: building image tag=%s context=%s "
                    "(log=%s timeout=%ds)",
                    local_tag,
                    dockerfile.parent,
                    log_path,
                    self._build_timeout_seconds,
                )
                await self._docker.build(
                    tag=local_tag,
                    context_dir=dockerfile.parent,
                    dockerfile=dockerfile,
                    timeout=self._build_timeout_seconds,
                    log_path=log_path,
                )
                self._resolved_tags[spec.name] = local_tag
                return local_tag

    # ----- per-rollout setup ------------------------------------------------

    async def setup_state(self, state: State) -> State:
        # Per-sample Harbor task name lives in ``info["task_name"]`` --
        # see ``load_environment`` for why we don't reuse ``state["task"]``
        # (that's the env id, used by PrimeRL's buffer for routing).
        # `Environment.run_rollout` populates `info` to a dict before this
        # hook runs; trust the contract and read keys directly. Any wrong-
        # type / missing-key surfaces as a clear KeyError/TypeError instead
        # of the prior `or {}` -> `or ""` -> `Unknown task: ''` chain.
        info = state["info"]
        if not isinstance(info, dict):
            raise TypeError(
                f"setup_state: state['info'] must be dict, got {type(info).__name__}"
            )
        task_name = info["task_name"]
        spec = self._task_specs.get(task_name)
        if spec is None:
            raise ValueError(
                f"Unknown terminal_bench_local task: {task_name!r} "
                f"(info={info!r}, state.task={state.get('task')!r}). "
                f"Known: {sorted(self._task_specs)}"
            )

        # One-shot sweep on the first rollout: bulk-remove containers from
        # THIS run only (label `prime_rl_tb_run_id={self._run_id}`). Doing
        # this here rather than in the worker boot path keeps env construction
        # synchronous and avoids reaching for the docker socket during import.
        # We deliberately do NOT sweep the generic `prime_rl_tb_env=1` label
        # because two concurrent runs on the same host would otherwise
        # `docker rm -f` each other's live rollout containers.
        if not self._stale_sweep_done:
            async with self._stale_sweep_lock:
                if not self._stale_sweep_done:
                    stale = await self._docker.ps_by_label(
                        "prime_rl_tb_run_id", self._run_id
                    )
                    if stale:
                        _logger.warning(
                            "terminal_bench_local: sweeping %d stale containers "
                            "from this run (run_id=%s): %s",
                            len(stale),
                            self._run_id,
                            stale[:10],
                        )
                        for cid in stale:
                            await self._docker.remove_force(cid)
                    self._stale_sweep_done = True

        # Resolve the image *before* we mint a container name -- a build
        # failure shouldn't leak a "tb-*" name into logs that suggests
        # the rollout got further than it did.
        image = await self._resolve_image(spec)

        # Use a uuid suffix so parallel rollouts of the same task don't
        # collide on container names. 16 hex chars = 64 bits — birthday-bound
        # collision probability is ~1e-9 even at 100k containers per training
        # run (8 chars = 32 bits gives 50% collision after ~65k, which the
        # 768-rollout × 500-step × 28-task TB config can hit). ``tb-`` prefix
        # makes these greppable with ``docker ps | grep ^tb-``.
        container = f"tb-{spec.name}-{uuid.uuid4().hex[:16]}"

        _logger.info(
            "terminal_bench_local: launching container name=%s image=%s task=%s",
            container,
            image,
            spec.name,
        )
        try:
            await self._docker.run_detached(
                image,
                name=container,
                labels=self._tb_labels(spec.name),
            )
        except RuntimeError as e:
            # `docker run` failed. If the failure looks like the BuildKit
            # corruption symptom ("failed to extract layer / content digest
            # not found"), the tag points at an image whose layers are
            # missing on disk — `image_exists()` returns True but the run
            # is unrecoverable. Untag, drop our cached resolution, rebuild
            # under flock, and retry the run once. Without this, every
            # subsequent rollout of the corrupted task hits the same error
            # forever (image_exists keeps short-circuiting back to the
            # broken tag).
            if (
                self._build_local_images
                and self._dockerfile_path(spec) is not None
                and _DockerClient._is_buildkit_corruption(str(e))
            ):
                _logger.warning(
                    "terminal_bench_local: docker run for %s hit BuildKit "
                    "corruption (%s); untagging + rebuilding under flock",
                    image,
                    e,
                )
                await self._invalidate_image(spec, image)
                image = await self._resolve_image(spec)
                container = f"tb-{spec.name}-{uuid.uuid4().hex[:16]}"
                try:
                    await self._docker.run_detached(
                        image,
                        name=container,
                        labels=self._tb_labels(spec.name),
                    )
                except Exception as e2:
                    state["tb_container_id"] = None
                    state["tb_task_dir"] = str(spec.dir)
                    state["tb_setup_error"] = str(e2)
                    raise
            else:
                # Leave state in a safe shape so cleanup can no-op. Re-raise
                # so the orchestrator marks the rollout as errored (zero
                # reward) instead of spinning forever.
                state["tb_container_id"] = None
                state["tb_task_dir"] = str(spec.dir)
                state["tb_setup_error"] = str(e)
                raise

        state["tb_container_id"] = container
        state["tb_task_dir"] = str(spec.dir)
        state["tb_reward"] = 0.0
        state["tb_reward_computed"] = False

        # Post-sandbox setup mirrors ``HarborEnv.prepare_harbor_task``:
        # ensure the agent workdir and verifier log dir exist, since
        # Harbor tasks assume both (task instructions say "create
        # hello.txt" expecting cwd=/app, and test.sh writes reward
        # to /logs/verifier). Doing this once at rollout start means
        # the model's first shell call can't race this setup.
        exit_code, _, stderr = await self._docker.exec(
            container,
            f"mkdir -p {_AGENT_WORKDIR} /logs/verifier",
            timeout=10,
        )
        if exit_code != 0:
            raise RuntimeError(
                f"Post-create setup (mkdir agent dirs) failed in {container}: {stderr!r}"
            )

        return await super().setup_state(state)

    # ----- per-turn tool execution ------------------------------------------

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> dict[str, Any]:
        """Inject the rollout's container id into every ``shell`` call."""
        updated = dict(tool_args)
        if tool_name == "shell":
            updated["_container_id"] = state.get("tb_container_id")
        return updated

    async def shell(self, command: str, _container_id: str | None = None) -> str:
        """Run a shell command inside the task sandbox.

        The sandbox is a Linux container pre-populated with the task's
        base image. Use standard POSIX shell syntax. Relative paths
        resolve from the container's WORKDIR (typically ``/app``). The
        container persists across turns so files and installed packages
        are carried forward.

        Args:
            command: The shell command to execute, e.g.
                ``echo "Hello, world!" > /app/hello.txt`` or
                ``ls -la /app``. Commands are executed via
                ``bash -lc``, so ``&&``, pipes, and heredocs work.

        Returns:
            A plain-text block ``exit_code=<n>\\n<stdout>\\nstderr:\\n<stderr>``
            truncated to ~8KiB. An empty result shows as ``(no output)``.
        """
        if not _container_id:
            # Defensive: setup_state should have errored long before we
            # get here, but keep the tool loop crash-free.
            return "Error: sandbox is not available for this rollout."

        exit_code, stdout, stderr = await self._docker.exec(
            _container_id,
            command,
            timeout=self._command_timeout_seconds,
            working_dir=_AGENT_WORKDIR,
        )
        return _format_shell_result(exit_code, stdout, stderr)

    # ----- post-rollout scoring --------------------------------------------

    @vf.cleanup
    async def finalize_rollout(self, state: State) -> None:
        """Run Harbor ``tests/test.sh`` for the reward, then destroy the container.

        ``@vf.cleanup`` fires once per rollout regardless of stop reason
        (including model errors or timeouts). Guarded with
        ``tb_reward_computed`` so idempotent.
        """
        container = state.get("tb_container_id")
        if not container:
            # Setup failed; nothing to score or clean up.
            return
        if not state.get("tb_reward_computed"):
            try:
                state["tb_reward"] = await self._compute_reward(state)
            except (RuntimeError, ValueError, OSError, LookupError) as e:
                # Narrow the catch: docker exec, parse failures, FS errors,
                # and missing reward keys are the documented reachable
                # failures of `_compute_reward`. A broader `except Exception`
                # would silently turn a real bug in the rubric into reward=0,
                # biasing the policy against tasks whose plumbing is flaky
                # rather than tasks the model actually fails. Mark the
                # state so downstream rubric/judge code can attach a
                # setup-error tag (rather than feeding the 0.0 to training
                # as a real all-failed group).
                _logger.warning(
                    "terminal_bench_local: reward computation failed for container=%s: %s",
                    container,
                    e,
                )
                state["tb_reward"] = 0.0
                state["tb_reward_setup_error"] = repr(e)
            finally:
                state["tb_reward_computed"] = True
        await self._docker.remove_force(container)

    async def _compute_reward(self, state: State) -> float:
        container = cast(str, state["tb_container_id"])
        task_dir = Path(cast(str, state["tb_task_dir"]))

        tests_dir = task_dir / "tests"
        if not tests_dir.is_dir():
            _logger.warning(
                "terminal_bench_local: task at %s has no tests/ dir; reward=0",
                task_dir,
            )
            return 0.0

        # Harbor expects tests at /tests and reward output at
        # /logs/verifier/reward.txt; pre-create both paths.
        prep = "mkdir -p /tests /logs/verifier && rm -rf /tests/* 2>/dev/null || true"
        exit_code, stdout, stderr = await self._docker.exec(container, prep, timeout=30)
        if exit_code != 0:
            # Include exit_code AND stdout — when the container has exited
            # before exec runs, docker prints to its own stderr but in some
            # versions the message lands on stdout, leaving the historical
            # `prep /tests failed: ''` empty-string footprint with no
            # diagnostic. exec returns 124 on timeout (with a message in
            # stderr); a 125-126-127 from docker indicates daemon/exec
            # failure rather than the inner shell command. Surface all of
            # it so the operator can tell apart "container died" from
            # "filesystem read-only" from "docker daemon down".
            raise RuntimeError(
                f"prep /tests failed: exit={exit_code} stdout={stdout!r} stderr={stderr!r}"
            )

        # ``docker cp <dir>/. <container>:<dst>`` copies dir *contents*
        # (trailing /.), which is what we want because test.sh expects
        # to live directly under /tests. Built as a raw string so the
        # trailing ``/.`` survives -- ``pathlib.Path`` would normalise
        # it away and we'd end up copying the directory itself into
        # /tests/tests/, making test.sh unreachable.
        await self._docker.cp_into(f"{tests_dir}/.", container, "/tests")

        exit_code, stdout, stderr = await self._docker.exec(
            container,
            "cd /tests && bash test.sh",
            timeout=self._test_timeout_seconds,
        )
        # test.sh can exit non-zero even on a valid 0-reward run if the
        # pytest harness returns non-zero; the reward file is the source
        # of truth. Log the diagnostic output for postmortems.
        if exit_code != 0:
            _logger.info(
                "terminal_bench_local: test.sh exit=%d in %s; stdout=%.200s stderr=%.200s",
                exit_code,
                container,
                stdout,
                stderr,
            )

        # Prefer reward.txt (a plain float) over reward.json (Harbor's
        # alternative). Matches HarborEnv.compute_reward upstream.
        _, reward_txt, _ = await self._docker.exec(
            container,
            "if [ -s /logs/verifier/reward.txt ]; then cat /logs/verifier/reward.txt; "
            "elif [ -s /logs/verifier/reward.json ]; then cat /logs/verifier/reward.json; fi",
            timeout=10,
        )
        raw = (reward_txt or "").strip()
        if not raw:
            _logger.warning(
                "terminal_bench_local: no reward.txt/json produced in %s; reward=0",
                container,
            )
            return 0.0
        try:
            return float(raw)
        except ValueError:
            # Strict access on the JSON path: a malformed reward.json
            # without a `reward` key is a rubric bug, not a "model failed"
            # signal. Returning 0.0 silently was biasing the policy
            # against tasks with flaky rubric plumbing rather than tasks
            # the model actually fails. Caller (`finalize_rollout`) sees
            # the LookupError and tags `tb_reward_setup_error`.
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as e:
                _logger.warning(
                    "terminal_bench_local: reward payload is neither float nor JSON: %r in %s: %s",
                    raw[:200],
                    container,
                    e,
                )
                raise ValueError(f"reward payload not parseable: {raw[:200]!r}") from e
            if "reward" not in payload:
                raise KeyError(f"reward.json missing 'reward' key in {container}: {raw[:200]!r}")
            return float(payload["reward"])


def _format_shell_result(exit_code: int, stdout: str, stderr: str) -> str:
    """Canonical tool-response shape: exit code + truncated stdout/stderr."""
    parts = [f"exit_code={exit_code}"]
    out = stdout.rstrip()
    err = stderr.rstrip()
    if out:
        parts.append(_truncate(out))
    if err:
        parts.append("stderr:\n" + _truncate(err))
    if len(parts) == 1:
        parts.append("(no output)")
    return "\n".join(parts)


def _truncate(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit - 64]
    return f"{head}\n...[truncated {len(text) - len(head)} chars]"


# ---------------------------------------------------------------------------
# Rubric: reward = the float produced by test.sh (already stored in state by
# the cleanup hook). We keep it as a plain Rubric rather than a JudgeRubric
# since the harness is authoritative -- no LLM judge needed.
# ---------------------------------------------------------------------------


async def _harbor_reward(state: vf.State, **kwargs) -> float:
    """Read the float written by `finalize_rollout`.

    If the reward path OR the rollout-setup path failed, raise instead of
    returning 0.0. A 0.0 reward feeds NPR's pass-rate Welford update and the
    advantage's group baseline as a real "model failed" signal — biasing the
    policy against tasks whose plumbing is flaky rather than tasks the model
    actually fails. Raising here lets verifiers' run_rollout mark the
    rollout's `error` field, which the scheduler already reschedules on.

    Honor BOTH error tags:
    - `tb_reward_setup_error`: set in `finalize_rollout` when reward
      computation itself fails (Docker exec error, unparseable
      reward.txt, missing tests/ dir).
    - `tb_setup_error`: set in `setup_state` when the rollout container
      could not even start (image build timeout, registry-whitelist
      block, EACCES on docker.sock). Previously only `tb_reward_setup_error`
      fired here — image-build failures slipped through as
      `state.get("tb_reward") or 0.0 → 0.0` and got attributed to the
      model. Same class of bug as the in-container egress failure mode
      (see comment block in `finalize_rollout`).
    """
    for key in ("tb_reward_setup_error", "tb_setup_error"):
        setup_error = state.get(key)
        if setup_error:
            raise RuntimeError(
                f"terminal_bench_local: {key} (rollout error, NOT a real zero "
                f"reward): {setup_error}"
            )
    return float(state.get("tb_reward") or 0.0)


# ---------------------------------------------------------------------------
# Entry point wired into [[orchestrator.env]].id = "environments.terminal_bench"
# ---------------------------------------------------------------------------


_DEFAULT_TASK_ROOT = Path("environments/terminal_bench/tasks")

# All shell tool calls run in /app by default, matching Harbor's
# ``agent_workdir = "/app"`` convention. Task instructions that say
# "create hello.txt" assume this cwd; without it we'd land at /.
_AGENT_WORKDIR = "/app"

_SYSTEM_PROMPT = (
    "You are a careful shell agent operating inside a Linux container.\n"
    f"Your working directory is {_AGENT_WORKDIR}; relative paths resolve from there.\n"
    "Use the `shell` tool to run commands; you can run multiple commands across\n"
    "turns. Each command is executed via `bash -lc`, so pipes, redirects, and\n"
    "heredocs work.\n"
    "\n"
    "Budget your reasoning: plan briefly, then ACT by calling the `shell` tool\n"
    "within the first few hundred thinking tokens. You can refine across turns\n"
    "based on observed output; over-planning before any tool call wastes your\n"
    "token budget. When you believe the task is complete, reply with a brief\n"
    "natural-language summary and stop calling tools; that signals that\n"
    "scoring should run."
)


def load_environment(
    tasks: list[str] | None = None,
    task_root: str | os.PathLike[str] = _DEFAULT_TASK_ROOT,
    max_turns: int = 10,
    command_timeout_seconds: int = _DEFAULT_COMMAND_TIMEOUT_SECONDS,
    test_timeout_seconds: int = _TEST_TIMEOUT_SECONDS,
    system_prompt: str = _SYSTEM_PROMPT,
    rollouts_per_task_in_dataset: int = 1,
    build_local_images: bool = True,
    build_timeout_seconds: int | None = None,
    build_log_dir: str | os.PathLike[str] | None = None,
) -> vf.Environment:
    """Build a ``TerminalBenchLocalEnv`` for one or more Harbor tasks.

    Args:
        tasks: Task directory names to include (relative to
            ``task_root``). If None, every subdirectory that has both
            ``task.toml`` and ``instruction.md`` is included.
        task_root: Path (absolute or relative to the prime-rl repo root)
            pointing at a Harbor-format ``tasks/`` dir. Defaults to
            ``environments/terminal_bench/tasks`` (28 in-tree task
            fixtures imported from upstream Terminal-Bench).
        max_turns: Hard cap on assistant turns per rollout.
        command_timeout_seconds: Timeout for individual ``shell`` calls.
        test_timeout_seconds: Timeout for ``bash test.sh`` at scoring
            time. Defaults to 300s which covers the hello-world
            ``apt-get update`` + ``uv install`` + ``pytest`` chain.
        system_prompt: Chat system message pushed ahead of every task.
        rollouts_per_task_in_dataset: Number of duplicate dataset rows
            per task. The orchestrator's own ``rollouts_per_example``
            already controls variance; this is only useful if you want
            the dataset itself to contain duplicates (e.g. ``batch_size``
            larger than ``len(tasks)`` with ``shuffle=False``). Default
            1 is the right choice for the phase-1 hello-world smoke.
        build_local_images: When True (default) and ``environment/
            Dockerfile`` exists under the task dir, ignore
            ``task.toml::docker_image`` (which typically points at a
            private Prime Intellect GCR) and build a local image
            ``tb-local/<task>:latest`` from the in-tree Dockerfile. Set
            False only if you've pre-pulled or built the upstream tags.
        build_timeout_seconds: Per-task ``docker build`` timeout.
            Defaults to ``TerminalBenchLocalEnv._DEFAULT_BUILD_TIMEOUT_SECONDS``
            (15 min) which covers the qemu tasks on a cold cache.
        build_log_dir: When set, per-task build output is streamed to
            ``<dir>/<task>.build.log`` instead of being captured in
            memory. Recommended for production; pass the orchestrator
            run's envs log dir so builds are greppable from the run.
    """
    root = Path(task_root)
    if not root.is_absolute():
        # Anchor relative paths to this env module's repo root, NOT the
        # caller's cwd. A caller from a directory other than the repo root
        # (ad-hoc unit invocation, resume from a different cwd) would
        # otherwise silently pick up a sibling repo's tasks/ dir or
        # FileNotFoundError on cwd. `Path(__file__).parent` is
        # `environments/terminal_bench/`; two parents up is the repo root.
        repo_root = Path(__file__).resolve().parent.parent.parent
        root = (repo_root / root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Harbor task root not found: {root}")

    if tasks is None:
        task_dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    else:
        task_dirs = [root / name for name in tasks]

    specs: list[_TaskSpec] = []
    for task_dir in task_dirs:
        specs.append(_load_task_spec(task_dir))
    if not specs:
        raise ValueError(f"No Harbor tasks loaded from {root}")

    # Each dataset row maps to one prompt. ``info["task_name"]`` carries
    # the Harbor task identifier used by ``setup_state`` to look up the
    # spec. We intentionally do NOT set the top-level ``"task"`` column:
    # PrimeRL's orchestrator/buffer uses that as the *env id* for
    # routing to the correct env server (see ``prime_rl.orchestrator.
    # buffer.Buffer`` and ``Environment._ensure_task`` which fills it
    # with ``self.env_id`` when absent). Overloading it with per-sample
    # names made every rollout land with
    # ``state["task"] == "environments.terminal_bench"`` inside
    # ``setup_state`` instead of the real task name.
    rows: list[dict[str, Any]] = []
    for idx, spec in enumerate(specs):
        for _ in range(max(1, rollouts_per_task_in_dataset)):
            rows.append(
                {
                    "prompt": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": spec.instruction},
                    ],
                    "answer": "",  # No ground-truth string; reward is from test.sh.
                    "info": {
                        "sample_id": f"tb-{spec.name}-{idx:02d}",
                        "task_name": spec.name,
                    },
                }
            )
    dataset = Dataset.from_list(rows)

    rubric = vf.Rubric(funcs=[_harbor_reward], weights=[1.0])

    env = TerminalBenchLocalEnv(
        task_specs=specs,
        dataset=dataset,
        eval_dataset=dataset,
        rubric=rubric,
        max_turns=max_turns,
        command_timeout_seconds=command_timeout_seconds,
        test_timeout_seconds=test_timeout_seconds,
        build_local_images=build_local_images,
        build_timeout_seconds=build_timeout_seconds,
        build_log_dir=build_log_dir,
    )
    return env
