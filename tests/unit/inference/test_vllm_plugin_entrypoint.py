"""Pin the `vllm.general_plugins` entry-point contract.

The single wiring point for fp32-LM-head, deep-gemm patches, DP pause/resume,
and Qwen3.5 LoRA fixes is `pyproject.toml::[project.entry-points."vllm.general_plugins"]`
mapping `prime_rl` to `prime_rl.inference.patches:transformers_v5_compat`. A
rename or removal silently disables every ScaleRL §3.2 patch with no functional
test failure (vLLM doesn't error if a configured plugin is missing — it just
no-ops). These tests catch the regression at the contract level.
"""

from importlib.metadata import entry_points

import pytest


def _get_vllm_general_plugins():
    """Discover `vllm.general_plugins` entry points without importing vllm."""
    try:
        return entry_points(group="vllm.general_plugins")
    except TypeError:
        # importlib.metadata < 3.10 returns a dict-like
        return entry_points().get("vllm.general_plugins", [])


def test_vllm_general_plugins_entrypoint_registered():
    """The `prime_rl` plugin must be discoverable on `vllm.general_plugins`."""
    eps = _get_vllm_general_plugins()
    names = {ep.name for ep in eps}
    assert "prime_rl" in names, (
        f"Expected `prime_rl` plugin under vllm.general_plugins, found: {sorted(names)}. "
        "Check pyproject.toml [project.entry-points.\"vllm.general_plugins\"]."
    )


def test_vllm_general_plugins_entrypoint_target_resolves():
    """The plugin must resolve to a callable."""
    eps = _get_vllm_general_plugins()
    target = next((ep for ep in eps if ep.name == "prime_rl"), None)
    assert target is not None, "prime_rl plugin not registered"
    assert target.value == "prime_rl.inference.patches:transformers_v5_compat", (
        f"Plugin target drifted: {target.value!r}. The umbrella function name is the "
        "single hook for every monkey-patch listed in patches.py — renaming silently "
        "disables fp32_lm_head and the rest."
    )
    fn = target.load()
    assert callable(fn), f"plugin target {target.value} did not load to a callable"


def test_vllm_fp32_lm_head_enabled_env_parsing():
    """`PRIME_RL_VLLM_FP32_LM_HEAD` parsing must accept the documented truthy
    spellings and reject the documented falsy ones — drift here would silently
    disable the FP32 LM-head path on inference workers.
    """
    import os

    from prime_rl.inference.patches import vllm_fp32_lm_head_enabled

    truthy = ["1", "true", "TRUE", "True", "yes", "YES", "on", "ON"]
    falsy = ["", "0", "false", "FALSE", "no", "NO", "off", "OFF"]

    saved = os.environ.pop("PRIME_RL_VLLM_FP32_LM_HEAD", None)
    try:
        for v in truthy:
            os.environ["PRIME_RL_VLLM_FP32_LM_HEAD"] = v
            assert vllm_fp32_lm_head_enabled(), f"value {v!r} should enable fp32 lm-head"
        for v in falsy:
            os.environ["PRIME_RL_VLLM_FP32_LM_HEAD"] = v
            assert not vllm_fp32_lm_head_enabled(), f"value {v!r} should NOT enable fp32 lm-head"
        os.environ.pop("PRIME_RL_VLLM_FP32_LM_HEAD")
        assert not vllm_fp32_lm_head_enabled(), "unset should default to disabled"
    finally:
        if saved is None:
            os.environ.pop("PRIME_RL_VLLM_FP32_LM_HEAD", None)
        else:
            os.environ["PRIME_RL_VLLM_FP32_LM_HEAD"] = saved
