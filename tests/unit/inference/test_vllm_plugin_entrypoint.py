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


def test_transformers_v5_compat_does_not_raise_at_module_top():
    """Regression guard: invoking the umbrella entry point must not blow up
    with NameError / ImportError before any monkey-patch runs.

    vLLM's plugin loader catches plugin-load exceptions and only logs a
    warning; an exception in the FIRST line of `transformers_v5_compat`
    silently disables every patch — including the load-bearing
    `monkey_patch_vllm_fp32_lm_head` for ScaleRL §3.2 train/inference
    logprob parity. The previous coverage at
    `test_vllm_general_plugins_entrypoint_target_resolves` only loaded the
    callable but never called it; this test fills that gap.

    Each patch has its own try/except inside the umbrella, so transient
    vLLM-version mismatches are tolerated. What we're guarding here is
    the umbrella's prologue (logger setup, transformers import) —
    failures there bypass every per-patch try/except.
    """
    from prime_rl.inference.patches import transformers_v5_compat

    # Should return cleanly, even on a vLLM pin where some internal
    # patches log warnings about renamed symbols. The contract is
    # "umbrella body executes without raising"; per-patch failures are
    # logged and isolated.
    transformers_v5_compat()


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


# `promote_parallel_lm_head_to_fp32` raise paths — three §3.2 footguns that
# silently disable the FP32 LM-head ingredient if the patch regresses.


class _FakeFloatLMHead:
    """Stand-in for vLLM's ParallelLMHead with a floating-point weight."""

    def __init__(self, dtype):
        import torch

        self.weight = torch.zeros(4, 8, dtype=dtype)
        self.bias = None

    def modules(self):
        return [self]


class _FakeIntLMHead(_FakeFloatLMHead):
    def __init__(self):
        import torch

        # Quantized vocab heads pack weights as int8/int4. Casting to fp32
        # silently produces garbage — `promote_parallel_lm_head_to_fp32`
        # must refuse loudly.
        self.weight = torch.zeros(4, 8, dtype=torch.int8)
        self.bias = None


class _FakeModel:
    """Top-level model object exposing a config + module iterator."""

    def __init__(self, head, *, tie_word_embeddings=False):
        self.config = type("Cfg", (), {"tie_word_embeddings": tie_word_embeddings})()
        self._head = head

    def modules(self):
        # `promote_parallel_lm_head_to_fp32` filters by isinstance(_, ParallelLMHead);
        # we monkeypatch ParallelLMHead in the test to be the fake's class so the
        # filter passes.
        return [self._head]


def _stub_parallel_lm_head_class(monkeypatch, cls):
    """Replace vllm's ParallelLMHead symbol with `cls` for the duration of the test."""
    import sys
    import types

    fake_module = types.ModuleType("vllm.model_executor.layers.vocab_parallel_embedding")
    fake_module.ParallelLMHead = cls
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.vocab_parallel_embedding", fake_module)


def test_promote_parallel_lm_head_to_fp32_no_op_when_disabled(monkeypatch):
    """env-var off → function returns 0 without touching modules."""
    import os

    from prime_rl.inference.patches import promote_parallel_lm_head_to_fp32

    monkeypatch.delenv("PRIME_RL_VLLM_FP32_LM_HEAD", raising=False)
    _stub_parallel_lm_head_class(monkeypatch, _FakeFloatLMHead)
    head = _FakeFloatLMHead(__import__("torch").float16)
    model = _FakeModel(head)
    assert promote_parallel_lm_head_to_fp32(model) == 0


def test_promote_parallel_lm_head_to_fp32_promotes_floating_point_head(monkeypatch):
    """Happy path: float16 weight → cast to fp32, sentinel set."""
    import os

    import torch

    from prime_rl.inference.patches import promote_parallel_lm_head_to_fp32

    monkeypatch.setenv("PRIME_RL_VLLM_FP32_LM_HEAD", "1")
    _stub_parallel_lm_head_class(monkeypatch, _FakeFloatLMHead)
    head = _FakeFloatLMHead(torch.bfloat16)
    model = _FakeModel(head)
    n = promote_parallel_lm_head_to_fp32(model)
    assert n == 1
    assert head.weight.dtype == torch.float32
    assert getattr(head, "_prime_rl_fp32_lm_head", False) is True


def test_promote_parallel_lm_head_to_fp32_refuses_quantized_head(monkeypatch):
    """Quantized (integer-dtype) head → loud refusal."""
    import pytest

    from prime_rl.inference.patches import promote_parallel_lm_head_to_fp32

    monkeypatch.setenv("PRIME_RL_VLLM_FP32_LM_HEAD", "1")
    _stub_parallel_lm_head_class(monkeypatch, _FakeIntLMHead)
    head = _FakeIntLMHead()
    model = _FakeModel(head)
    with pytest.raises(RuntimeError, match="quantized ParallelLMHead"):
        promote_parallel_lm_head_to_fp32(model)


def test_promote_parallel_lm_head_to_fp32_refuses_tied_embeddings(monkeypatch):
    """Tied-embedding model → loud refusal (input embedding shares LM-head weight)."""
    import pytest
    import torch

    from prime_rl.inference.patches import promote_parallel_lm_head_to_fp32

    monkeypatch.setenv("PRIME_RL_VLLM_FP32_LM_HEAD", "1")
    _stub_parallel_lm_head_class(monkeypatch, _FakeFloatLMHead)
    head = _FakeFloatLMHead(torch.bfloat16)
    model = _FakeModel(head, tie_word_embeddings=True)
    with pytest.raises(RuntimeError, match="tie_word_embeddings"):
        promote_parallel_lm_head_to_fp32(model)


def test_promote_parallel_lm_head_to_fp32_refuses_storage_aliased_embedding(monkeypatch):
    """Belt-and-suspenders: even when `tie_word_embeddings=False` is *declared*,
    some vLLM load paths still alias `ParallelLMHead.weight`'s STORAGE to the
    input embedding's. Promoting in place would push fp32 weights into the
    embedding's bf16 forward and produce dtype-mismatch or silent garbage.

    Critical distinction: this test constructs DISTINCT `nn.Parameter` wrappers
    that share the underlying storage (`data_ptr()` matches), to exercise the
    real-world failure mode. A wrapper-identity check (`id(weight) == id(emb.weight)`)
    would miss this; the storage-identity check (`data_ptr()`) catches it.
    """
    import pytest
    import torch
    from torch import nn

    from prime_rl.inference.patches import promote_parallel_lm_head_to_fp32

    monkeypatch.setenv("PRIME_RL_VLLM_FP32_LM_HEAD", "1")
    _stub_parallel_lm_head_class(monkeypatch, _FakeFloatLMHead)

    head = _FakeFloatLMHead(torch.bfloat16)

    class _ModelWithAliasedStorage:
        def __init__(self, head_):
            self.config = type("Cfg", (), {"tie_word_embeddings": False})()
            self._head = head_
            # Distinct Parameter wrapper, shared storage — the realistic
            # vLLM-load-path footgun the comment describes.
            shared = nn.Parameter(head_.weight.data)
            assert shared is not head_.weight, "expected distinct wrappers"
            assert shared.data_ptr() == head_.weight.data_ptr(), "expected shared storage"
            self._embedding = type("Emb", (), {"weight": shared})()

        def modules(self):
            return [self._head]

        def get_input_embeddings(self):
            return self._embedding

    model = _ModelWithAliasedStorage(head)
    with pytest.raises(RuntimeError, match="aliased to the input embedding"):
        promote_parallel_lm_head_to_fp32(model)


def test_promote_parallel_lm_head_to_fp32_accepts_distinct_embedding(monkeypatch):
    """Sanity: when the embedding weight is a SEPARATE tensor (no aliasing),
    the storage-id check must NOT trip. Same-shape weights with different
    storage are the typical untied-embedding case."""
    import torch

    from prime_rl.inference.patches import promote_parallel_lm_head_to_fp32

    monkeypatch.setenv("PRIME_RL_VLLM_FP32_LM_HEAD", "1")
    _stub_parallel_lm_head_class(monkeypatch, _FakeFloatLMHead)

    head = _FakeFloatLMHead(torch.bfloat16)

    class _ModelWithDistinctEmbedding:
        def __init__(self, head_):
            self.config = type("Cfg", (), {"tie_word_embeddings": False})()
            self._head = head_
            self._embedding = type("Emb", (), {"weight": torch.zeros_like(head_.weight)})()

        def modules(self):
            return [self._head]

        def get_input_embeddings(self):
            return self._embedding

    model = _ModelWithDistinctEmbedding(head)
    promoted = promote_parallel_lm_head_to_fp32(model)
    assert promoted == 1
    assert head.weight.dtype == torch.float32


def test_promote_parallel_lm_head_to_fp32_raises_when_zero_promoted(monkeypatch):
    """No ParallelLMHead found but env var is on → raise instead of silently
    shipping a recipe regression. Catches vLLM module-hierarchy drift.
    """
    import pytest

    from prime_rl.inference.patches import promote_parallel_lm_head_to_fp32

    monkeypatch.setenv("PRIME_RL_VLLM_FP32_LM_HEAD", "1")
    _stub_parallel_lm_head_class(monkeypatch, _FakeFloatLMHead)

    class _NoHeadModel:
        config = type("Cfg", (), {"tie_word_embeddings": False})()

        def modules(self):
            return []

    with pytest.raises(RuntimeError, match="no ParallelLMHead modules found"):
        promote_parallel_lm_head_to_fp32(_NoHeadModel())
