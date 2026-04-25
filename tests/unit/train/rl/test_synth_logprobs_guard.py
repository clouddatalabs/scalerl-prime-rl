"""Tests for the runtime guard against synthesized inference_logprobs.

Pairs with `tests/unit/orchestrator/test_batch.py::test_packed_micro_batch_or_merges_synth_flag`
and `validate_external_rollout_mode` config tests in `tests/unit/test_configs.py`
to close the IS-ratio + zero-fill silent-corruption mode end to end.
"""

import pytest

from prime_rl.trainer.rl.train import _assert_synthesized_logprobs_compatible_with_loss


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
