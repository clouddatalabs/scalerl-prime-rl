"""Tests for the partial-credit reward discount on rollouts whose tool
calls were ONLY parsed because the inference-side tolerant-JSON Hermes
patch (`monkey_patch_hermes_tool_parser_tolerant_json`) rescued an
invalid `\\X` escape inside the JSON body.

Contract under test:

- `TerminalBenchLocalEnv.update_tool_args` detects + strips the
  `__prime_rl_repaired_json__` sentinel that the inference patch
  injects when it rescues a tool call. The sentinel must NOT reach
  the `shell()` handler.
- Each detected sentinel bumps a per-rollout counter on `state`.
- `_harbor_reward` halves the returned reward when the counter > 0
  AND the underlying reward is > 0. Wrong answer → still 0; clean
  answer → still full reward.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from environments.terminal_bench.env import (  # noqa: E402
    TerminalBenchLocalEnv,
    _harbor_reward,
)


def _bare_env() -> TerminalBenchLocalEnv:
    """Build an env instance bypassing __init__ — we only exercise
    `update_tool_args`, which doesn't touch any of the heavy state set
    up by `__init__` (docker client, task specs, etc.)."""
    return TerminalBenchLocalEnv.__new__(TerminalBenchLocalEnv)


def test_update_tool_args_strips_sentinel_and_increments_counter():
    env = _bare_env()
    state: dict = {"tb_container_id": "tb-fake-deadbeef"}
    args = {"command": "ls -la", "__prime_rl_repaired_json__": True}

    out = env.update_tool_args("shell", args, messages=[], state=state)

    # Sentinel stripped before reaching shell()
    assert "__prime_rl_repaired_json__" not in out
    assert out["command"] == "ls -la"
    # Container id injected as before
    assert out["_container_id"] == "tb-fake-deadbeef"
    # Per-rollout repair counter incremented
    assert state["tb_repaired_tool_call_count"] == 1


def test_update_tool_args_counter_accumulates_across_calls():
    env = _bare_env()
    state: dict = {"tb_container_id": "tb-fake"}
    for _ in range(3):
        env.update_tool_args(
            "shell",
            {"command": "echo hi", "__prime_rl_repaired_json__": True},
            messages=[],
            state=state,
        )
    assert state["tb_repaired_tool_call_count"] == 3


def test_update_tool_args_no_sentinel_means_no_counter_bump():
    env = _bare_env()
    state: dict = {"tb_container_id": "tb-fake"}
    env.update_tool_args(
        "shell",
        {"command": "ls"},
        messages=[],
        state=state,
    )
    assert "tb_repaired_tool_call_count" not in state


def test_update_tool_args_handles_non_shell_tool():
    """Sentinel detection is tool-agnostic — counter still bumps even
    if the call isn't `shell` (defensive: a future tool added to the
    env's surface should still discount). But `_container_id` is only
    injected for `shell`."""
    env = _bare_env()
    state: dict = {"tb_container_id": "tb-fake"}
    out = env.update_tool_args(
        "some_future_tool",
        {"k": "v", "__prime_rl_repaired_json__": True},
        messages=[],
        state=state,
    )
    assert state["tb_repaired_tool_call_count"] == 1
    assert "__prime_rl_repaired_json__" not in out
    assert "_container_id" not in out


def test_harbor_reward_clean_full_credit():
    state = {"tb_reward": 1.0}
    assert asyncio.run(_harbor_reward(state)) == 1.0


def test_harbor_reward_repaired_half_credit():
    state = {"tb_reward": 1.0, "tb_repaired_tool_call_count": 1}
    assert asyncio.run(_harbor_reward(state)) == 0.5


def test_harbor_reward_repaired_partial_reward_still_halves():
    """Discount is multiplicative — partial reward × 0.5."""
    state = {"tb_reward": 0.4, "tb_repaired_tool_call_count": 2}
    assert asyncio.run(_harbor_reward(state)) == pytest.approx(0.2)


def test_harbor_reward_zero_stays_zero_with_or_without_repair():
    """Wrong answer → 0% either way (per user spec)."""
    state_clean = {"tb_reward": 0.0}
    state_dirty = {"tb_reward": 0.0, "tb_repaired_tool_call_count": 5}
    assert asyncio.run(_harbor_reward(state_clean)) == 0.0
    assert asyncio.run(_harbor_reward(state_dirty)) == 0.0


def test_harbor_reward_setup_error_still_raises():
    """Setup-error path is unchanged: discount logic must not swallow
    the orchestrator's "rollout errored" signal."""
    state = {
        "tb_reward": 1.0,
        "tb_repaired_tool_call_count": 1,
        "tb_setup_error": "image pull failed",
    }
    with pytest.raises(RuntimeError, match="tb_setup_error"):
        asyncio.run(_harbor_reward(state))
