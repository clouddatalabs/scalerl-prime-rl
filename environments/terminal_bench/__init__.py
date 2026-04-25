"""Harbor-format Terminal-Bench tasks executed against a local Docker daemon.

This env mirrors the upstream Harbor task format (``instruction.md``,
``task.toml``, ``environment/Dockerfile``, ``tests/test.sh``) but replaces
the cloud-sandbox + frpc tunnel pipeline with plain ``docker run`` /
``docker exec`` against the host Docker daemon. That keeps training
self-contained on a single node.

Imported task fixtures under ``tasks/`` retain their upstream Apache-2.0
license; see ``THIRD_PARTY_NOTICES.md`` in this directory for sources
and per-task attribution.

Preconditions:
- A Docker daemon reachable from the rollout process. Typically that means
  ``/var/run/docker.sock`` is bind-mounted into the prime-rl container, or
  the rollout process runs on a host with direct access to the daemon.
- Each task's base image (``task.toml::[environment].docker_image``) must
  either exist locally or be pullable; the first rollout of a task will
  trigger a pull which dominates time-to-first-step.

Harbor-format tasks are checked in under ``environments/terminal_bench/tasks/``
— no external dataset download required. Some entries in that directory are
symlinks pointing outside the repo (e.g. to a sibling ``opencode_harbor``
tree); ``configs/scalerl_terminal_bench/rl.toml`` enumerates only the
runnable ones, so the runtime list is the source of truth, not ``ls tasks/``.

Usage in an orchestrator config (``name`` is required when more than one
env entry references this module — `validate_unique_env_names` rejects
duplicates):

    [[orchestrator.train.env]]
    id = "environments.terminal_bench"
    name = "terminal-bench-train"
    args = { tasks = ["analyze-access-logs"], task_root = "environments/terminal_bench/tasks" }
"""


def load_environment(*args, **kwargs):
    """Lazy shim so config preflight stays metadata-only (does not import vf at parse time)."""
    from environments.terminal_bench.env import load_environment as _load_environment

    return _load_environment(*args, **kwargs)


__all__ = ["load_environment"]
