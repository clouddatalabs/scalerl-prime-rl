"""CPU-runnable tests for the FA4 HF AttentionInterface bridge.

The smoke and Snowflake handoff configs both depend on `attn="fa4"` +
`impl="hf"` — these tests pin the registration / signature / config-validation
contract that bridge relies on. The actual FA4 forward path requires CUDA +
the cute kernels and is exercised by the GPU smoke run.
"""

import inspect

import pytest
import torch

from prime_rl.configs.trainer import ModelConfig as TrainerModelConfig
from prime_rl.trainer.model import (
    _fa4_attention_forward,
    _register_fa4_attention_interface,
)


def test_register_fa4_attention_interface_actually_registers_the_bridge():
    """Confirm `_register_fa4_attention_interface` makes "fa4" callable through
    the transformers registry — not the previous no-op stub."""
    from transformers import AttentionInterface

    _register_fa4_attention_interface()
    registered = AttentionInterface._global_mapping.get("fa4") if hasattr(
        AttentionInterface, "_global_mapping"
    ) else AttentionInterface.get("fa4")
    assert registered is _fa4_attention_forward, (
        "fa4 must resolve to our HF bridge after registration; got "
        f"{registered!r}. If this fails, the transformers AttentionInterface API drifted."
    )


def test_fa4_attention_forward_signature_matches_transformers_contract():
    """Pin the transformers FA-adapter signature so silent renames are caught."""
    sig = inspect.signature(_fa4_attention_forward)
    expected = ["module", "query", "key", "value", "attention_mask"]
    actual = list(sig.parameters)[: len(expected)]
    assert actual == expected, f"FA4 bridge signature drift: got {actual}, expected prefix {expected}"
    keyword_only = {
        name
        for name, p in sig.parameters.items()
        if p.kind is inspect.Parameter.KEYWORD_ONLY or p.default is not inspect.Parameter.empty
    }
    for k in ("dropout", "scaling", "sliding_window", "softcap", "is_causal"):
        assert k in keyword_only or k in sig.parameters, (
            f"FA4 bridge must accept transformers FA-adapter kwarg `{k}`."
        )


def test_fa4_attention_forward_imports_transformers_helpers():
    """The bridge imports six private helpers from transformers — surface them
    here so a transformers minor-version rename fails this test instead of
    blowing up inside a `_dummy_run` 200 GiB later."""
    from transformers.modeling_flash_attention_utils import (  # noqa: F401
        _is_packed_sequence,
        _pad_input,
        _prepare_from_posids,
        _unpad_input,
        _upad_input,
        fa_peft_integration_check,
    )


def test_fa4_attention_forward_rejects_dropout():
    """The bridge documents dropout=0 only — guardrail must fail loudly."""

    class _StubModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.is_causal = True

    module = _StubModule()
    q = torch.zeros(1, 1, 1, 8)
    with pytest.raises(ValueError, match="dropout=0.0"):
        _fa4_attention_forward(module, q, q, q, attention_mask=None, dropout=0.5)


def test_fa4_attention_forward_rejects_zero_dim_query():
    """Empty inputs are never legitimate for FA — the bridge should redirect to SDPA."""

    class _StubModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.is_causal = True

    module = _StubModule()
    q = torch.zeros(0, 1, 1, 8)
    with pytest.raises(ValueError, match="zero dimension"):
        _fa4_attention_forward(module, q, q, q, attention_mask=None)


def test_model_config_accepts_fa4_with_hf_impl():
    """The validator that previously gated fa4 to impl="custom" must now accept
    impl="hf" too — this is the path used by both POC configs."""
    cfg = TrainerModelConfig.model_validate({"attn": "fa4", "impl": "hf"})
    assert cfg.attn == "fa4"
    assert cfg.impl == "hf"


def test_model_config_accepts_fa4_with_custom_impl():
    """Custom impl + fa4 still works for the MoE-Qwen3 path."""
    cfg = TrainerModelConfig.model_validate({"attn": "fa4", "impl": "custom"})
    assert cfg.attn == "fa4"
    assert cfg.impl == "custom"


def test_model_config_accepts_fa4_with_auto_impl():
    cfg = TrainerModelConfig.model_validate({"attn": "fa4", "impl": "auto"})
    assert cfg.attn == "fa4"


@pytest.mark.gpu
def test_fa4_attention_forward_runs_on_gpu_with_packed_position_ids():
    """Exercise the cute kernels through `_fa4_attention_forward` on real GPU.

    Constructs a synthetic packed RL batch (two short "sequences" packed into
    a single buffer with restart position_ids) and checks the bridge returns
    a tensor with the right shape and no NaN. Runs only when CUDA is
    available; the math smoke job 694 is the empirical backstop.

    Marked with `@pytest.mark.gpu` so the default CPU `baker test` matrix
    skips it (consistent with how the rest of the GPU tests are gated).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for FA4 forward path")

    _register_fa4_attention_interface()
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Pretend two packed sequences of length 4 each, with 8 attention heads,
    # 8 kv heads, head_dim 64. Positions restart at the boundary.
    batch, total_q, num_heads, head_dim = 1, 8, 8, 64
    query = torch.randn(batch, num_heads, total_q, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch, num_heads, total_q, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch, num_heads, total_q, head_dim, device=device, dtype=dtype)
    position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], device=device)

    class _StubModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.is_causal = True
            self.linear = torch.nn.Linear(head_dim, head_dim, dtype=dtype, device=device)
            self.config = type("C", (), {})()

    module = _StubModule()
    out, _ = _fa4_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask=None,
        scaling=1.0 / (head_dim**0.5),
        position_ids=position_ids,
    )
    assert out.shape[0] == batch
    assert out.shape[-1] == head_dim
    assert not torch.isnan(out).any(), "FA4 produced NaN — kernel/dtype mismatch on this GPU"


def test_model_config_rejects_fa4_with_invalid_impl():
    """impl is a Literal["hf", "custom", "auto"], so anything outside that set
    fails the typed-value check before our FA4 validator sees it. The
    `flash_attention_4_supported_impls` validator is therefore a defensive
    second line — kept so that adding a new impl Literal without updating
    the FA4 list trips immediately."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match=r"hf'.*'custom'.*'auto'|Flash attention 4"):
        TrainerModelConfig.model_validate({"attn": "fa4", "impl": "fsdp"})
