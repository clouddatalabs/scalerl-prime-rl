"""Harbor-format Terminal-Bench tasks executed against a local Docker daemon.

This env mirrors the upstream Harbor task format (``instruction.md``,
``task.toml``, ``environment/Dockerfile``, ``tests/test.sh``) but replaces
the cloud-sandbox + frpc tunnel pipeline with plain ``docker run`` /
``docker exec`` against the host Docker daemon. That keeps training
self-contained on a single node.

Preconditions:
- A Docker daemon reachable from the rollout process. Typically that means
  ``/var/run/docker.sock`` is bind-mounted into the prime-rl container, or
  the rollout process runs on a host with direct access to the daemon.
- Each task's base image (``task.toml::[environment].docker_image``) must
  either exist locally or be pullable; the first rollout of a task will
  trigger a pull which dominates time-to-first-step.

The 30 tasks under ``environments/terminal_bench/tasks/`` are checked-in
Harbor-format directories — no external dataset download required.

Usage in an orchestrator config:

    [[orchestrator.train.env]]
    id = "environments.terminal_bench"
    args = { tasks = ["analyze-access-logs"], task_root = "environments/terminal_bench/tasks" }
"""


def load_environment(*args, **kwargs):
    """Lazy shim so config preflight stays metadata-only (does not import vf at parse time)."""
    from environments.terminal_bench.env import load_environment as _load_environment

    return _load_environment(*args, **kwargs)


__all__ = ["load_environment"]
