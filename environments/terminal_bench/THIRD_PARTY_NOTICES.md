# Third-Party Notices — `environments/terminal_bench/`

The Harbor-format task fixtures committed under
`environments/terminal_bench/tasks/` are imported verbatim from the upstream
**Terminal-Bench** project. The fork applies no logic
changes to those task directories; only file-format / path adjustments
needed to run them against a local Docker daemon (see
`environments/terminal_bench/__init__.py`).

## Upstream sources

- **Terminal-Bench**:
  https://github.com/laude-institute/terminal-bench
  Apache License 2.0.

The `.gitattributes` and `pyproject.toml::extend-exclude` rules in this
fork mark the imported fixtures as third-party so style / lint checks
don't rewrite them, preserving binary fidelity for upstream rebases.

## Per-task licensing

Each individual task directory under `tasks/` may carry its own
`LICENSE`, `task.toml::license`, or attribution metadata sourced from
the upstream task author. The aggregate `LICENSE` file at the
repository root applies to fork-original code (`src/prime_rl/*`,
`baker/*`, configs, scripts); imported task fixtures retain their
upstream license terms.

## Modifications in this fork

- The `_DockerClient` thin wrapper around `docker` CLI is fork-original
  (`environments/terminal_bench/env.py`); the underlying Harbor rubric
  conventions (`tests/test.sh`, `/logs/verifier/reward.{txt,json}`) are
  upstream contracts honored without modification.

## Reporting

If you believe a task fixture in this tree is missing required
attribution or has been modified in a way that would change its
license obligations, open an issue on the fork repository.
