"""Tests for the runtime guard against synthesized inference_logprobs.

Pairs with `tests/unit/orchestrator/test_batch.py::test_packed_micro_batch_or_merges_synth_flag`
and `validate_external_rollout_mode` config tests in `tests/unit/test_configs.py`
to close the IS-ratio + zero-fill silent-corruption mode end to end.
"""

import pytest

from prime_rl.trainer.rl.train import (
    _LOSS_TYPES_SAFE_UNDER_SYNTHESIZED_LOGPROBS,
    _assert_synthesized_logprobs_compatible_with_loss,
    _loss_consumes_importance_ratio,
)


def test_synth_logprobs_rejected_for_cispo():
    with pytest.raises(RuntimeError, match="synthesized"):
        _assert_synthesized_logprobs_compatible_with_loss(loss_type="cispo", synthesized=True)


def test_synth_logprobs_rejected_for_default():
    with pytest.raises(RuntimeError, match="synthesized"):
        _assert_synthesized_logprobs_compatible_with_loss(loss_type="default", synthesized=True)


def test_synth_logprobs_passes_when_clean():
    _assert_synthesized_logprobs_compatible_with_loss(loss_type="cispo", synthesized=False)
    _assert_synthesized_logprobs_compatible_with_loss(loss_type="default", synthesized=False)


def test_synth_logprobs_allowed_for_sft():
    _assert_synthesized_logprobs_compatible_with_loss(loss_type="sft", synthesized=True)
    _assert_synthesized_logprobs_compatible_with_loss(loss_type="sft", synthesized=False)


def test_error_message_names_recovery_options():
    with pytest.raises(RuntimeError) as excinfo:
        _assert_synthesized_logprobs_compatible_with_loss(loss_type="cispo", synthesized=True)
    msg = str(excinfo.value)
    assert "use_token_client=True" in msg
    assert "loss.type='sft'" in msg


def test_synth_guard_safelist_only_contains_sft():
    """Encoded as a safelist so future losses (DAPO, GSPO, …) are denied by
    default. Any new loss that genuinely doesn't consume `inference_logprobs`
    must be added here explicitly."""
    assert _LOSS_TYPES_SAFE_UNDER_SYNTHESIZED_LOGPROBS == frozenset({"sft"})


def test_synth_guard_denies_unknown_loss_type_by_default():
    """A future loss name not in the safelist is treated as IS-ratio-consuming."""
    with pytest.raises(RuntimeError, match="synthesized"):
        _assert_synthesized_logprobs_compatible_with_loss(loss_type="dapo", synthesized=True)
    with pytest.raises(RuntimeError, match="synthesized"):
        _assert_synthesized_logprobs_compatible_with_loss(loss_type="gspo", synthesized=True)


def test_loss_consumes_importance_ratio_resolves_custom_opt_out():
    """CustomLossConfig.consumes_importance_ratio drives the gating for custom losses."""
    from types import SimpleNamespace

    cfg_default = SimpleNamespace(loss=SimpleNamespace(type="custom", consumes_importance_ratio=True))
    cfg_optout = SimpleNamespace(loss=SimpleNamespace(type="custom", consumes_importance_ratio=False))
    cfg_cispo = SimpleNamespace(loss=SimpleNamespace(type="cispo"))
    cfg_sft = SimpleNamespace(loss=SimpleNamespace(type="sft"))

    assert _loss_consumes_importance_ratio(cfg_default) is True
    assert _loss_consumes_importance_ratio(cfg_optout) is False
    assert _loss_consumes_importance_ratio(cfg_cispo) is True
    assert _loss_consumes_importance_ratio(cfg_sft) is False
