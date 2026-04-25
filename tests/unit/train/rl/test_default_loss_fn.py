"""CPU tests for default_loss_fn (DPPO+KL).

The advantage-sign-conditioned mask is the load-bearing recipe ingredient
(`torch.where(advantages > 0, mask_high, mask_low)` in `default_loss_fn`).
Existing tests in `test_loss.py` are GPU-marked smokes that only pin
`loss.shape == ()`. Mirror the CISPO coverage here.
"""

import math

import torch

from prime_rl.configs.trainer import DefaultLossConfig
from prime_rl.trainer.rl.loss import LossInputs, default_loss_fn


def _inputs(*, trainer_lp, inference_lp, advantages, loss_mask=None, teacher_lp=None):
    if loss_mask is None:
        loss_mask = [True] * len(trainer_lp)
    return LossInputs(
        trainer_logprobs=torch.tensor(trainer_lp),
        inference_logprobs=torch.tensor(inference_lp),
        teacher_logprobs=torch.tensor(teacher_lp) if teacher_lp is not None else None,
        advantages=torch.tensor(advantages),
        loss_mask=torch.tensor(loss_mask),
    )


def test_default_loss_masks_high_when_advantage_positive():
    """A token whose probability INCREASED past dppo_mask_high gets masked
    iff its advantage is positive (trust-region violation in upweight direction)."""
    # probs_diff = exp(0) - exp(log(0.5)) = 1 - 0.5 = 0.5 > 0.2 -> high mask trips
    inputs = _inputs(
        trainer_lp=[0.0],
        inference_lp=[math.log(0.5)],
        advantages=[1.0],
    )
    out = default_loss_fn(inputs, DefaultLossConfig(dppo_mask_high=0.2, dppo_mask_low=0.2, kl_tau=0.0))
    assert torch.isclose(out.metrics["is_masked_high"], torch.tensor(1.0))
    assert torch.isclose(out.metrics["is_masked_low"], torch.tensor(0.0))
    assert torch.isclose(out.metrics["is_masked"], torch.tensor(1.0))


def test_default_loss_does_not_mask_high_when_advantage_negative():
    """Same probs_diff > dppo_mask_high but advantage < 0 -> the high mask must NOT fire.
    For negative advantages only the low mask fires.
    """
    inputs = _inputs(
        trainer_lp=[0.0],
        inference_lp=[math.log(0.5)],
        advantages=[-1.0],
    )
    out = default_loss_fn(inputs, DefaultLossConfig(dppo_mask_high=0.2, dppo_mask_low=0.2, kl_tau=0.0))
    # probs_diff = +0.5; not < -0.2; so low mask doesn't trip either.
    assert torch.isclose(out.metrics["is_masked_high"], torch.tensor(0.0))
    assert torch.isclose(out.metrics["is_masked_low"], torch.tensor(0.0))
    assert torch.isclose(out.metrics["is_masked"], torch.tensor(0.0))


def test_default_loss_masks_low_when_advantage_negative():
    """A token whose probability DECREASED past dppo_mask_low gets masked
    iff its advantage is negative (trust-region violation in downweight direction)."""
    # probs_diff = exp(log(0.3)) - exp(log(0.8)) = 0.3 - 0.8 = -0.5 < -0.2 -> low mask trips
    inputs = _inputs(
        trainer_lp=[math.log(0.3)],
        inference_lp=[math.log(0.8)],
        advantages=[-1.0],
    )
    out = default_loss_fn(inputs, DefaultLossConfig(dppo_mask_high=0.2, dppo_mask_low=0.2, kl_tau=0.0))
    assert torch.isclose(out.metrics["is_masked_low"], torch.tensor(1.0))
    assert torch.isclose(out.metrics["is_masked_high"], torch.tensor(0.0))
    assert torch.isclose(out.metrics["is_masked"], torch.tensor(1.0))


def test_default_loss_does_not_mask_low_when_advantage_positive():
    """Same probs_diff < -dppo_mask_low but advantage > 0 -> the low mask must NOT fire."""
    inputs = _inputs(
        trainer_lp=[math.log(0.3)],
        inference_lp=[math.log(0.8)],
        advantages=[1.0],
    )
    out = default_loss_fn(inputs, DefaultLossConfig(dppo_mask_high=0.2, dppo_mask_low=0.2, kl_tau=0.0))
    assert torch.isclose(out.metrics["is_masked_low"], torch.tensor(0.0))
    assert torch.isclose(out.metrics["is_masked_high"], torch.tensor(0.0))
    assert torch.isclose(out.metrics["is_masked"], torch.tensor(0.0))


def test_default_loss_masked_token_drops_from_pg_loss():
    """Masked tokens contribute zero to the pg term; the kl term still counts them.

    Construct: one positive-adv token that's masked-high, one positive-adv token
    that isn't. Compare pg-only loss (kl_tau=0) to the manual sum.
    """
    inputs = _inputs(
        trainer_lp=[0.0, math.log(0.6)],            # token 0 trips high mask, token 1 doesn't
        inference_lp=[math.log(0.5), math.log(0.5)],
        advantages=[1.0, 1.0],
    )
    cfg = DefaultLossConfig(dppo_mask_high=0.2, dppo_mask_low=0.2, kl_tau=0.0, adv_tau=1.0)
    out = default_loss_fn(inputs, cfg)
    # Token 0 masked, contributes 0 to pg. Token 1 unmasked:
    #   importance_ratio = exp(log0.6 - log0.5) = 1.2
    #   pg = -1 * keep_mask(1) * advantages(1) * 1.2 = -1.2
    assert torch.isclose(out.loss, torch.tensor(-1.2), atol=1e-5)


def test_default_loss_kl_term_uses_loss_mask_not_keep_mask():
    """The KL term sums over `loss_mask` (all trainable tokens) regardless of
    whether the DPPO mask flagged them — masked tokens still contribute KL.
    """
    # Both positive advantage; first trips high mask. Set adv_tau=0 so pg_loss=0
    # and only kl_loss survives. log_ratio for token 0 = 0 - log(0.5) = log(2);
    # log_ratio for token 1 = log(0.6) - log(0.5) = log(1.2)
    inputs = _inputs(
        trainer_lp=[0.0, math.log(0.6)],
        inference_lp=[math.log(0.5), math.log(0.5)],
        advantages=[1.0, 1.0],
    )
    cfg = DefaultLossConfig(dppo_mask_high=0.2, dppo_mask_low=0.2, kl_tau=1.0, adv_tau=0.0)
    out = default_loss_fn(inputs, cfg)
    # kl_loss = sum(loss_mask * log_ratio**2) = log(2)**2 + log(1.2)**2
    expected = math.log(2.0) ** 2 + math.log(1.2) ** 2
    assert torch.isclose(out.loss, torch.tensor(expected), atol=1e-5)


def test_default_loss_loss_mask_overrides_keep_mask():
    """A token outside loss_mask is never trained, even if probs_diff is small."""
    inputs = _inputs(
        trainer_lp=[math.log(0.5), math.log(0.5)],
        inference_lp=[math.log(0.5), math.log(0.5)],
        advantages=[1.0, 1.0],
        loss_mask=[True, False],
    )
    cfg = DefaultLossConfig(dppo_mask_high=0.2, dppo_mask_low=0.2, kl_tau=0.0, adv_tau=1.0)
    out = default_loss_fn(inputs, cfg)
    # Only token 0 contributes: importance_ratio=1, advantage=1 -> pg = -1
    assert torch.isclose(out.loss, torch.tensor(-1.0), atol=1e-5)


def test_default_loss_teacher_kl_metric_present_only_when_teacher_logprobs_set():
    no_teacher = _inputs(
        trainer_lp=[0.0], inference_lp=[0.0], advantages=[1.0]
    )
    out_no = default_loss_fn(no_teacher, DefaultLossConfig(kl_tau=0.0))
    assert "teacher_kl" not in out_no.metrics

    with_teacher = _inputs(
        trainer_lp=[0.0], inference_lp=[0.0], advantages=[1.0], teacher_lp=[-1.0]
    )
    out_yes = default_loss_fn(with_teacher, DefaultLossConfig(kl_tau=0.0))
    assert "teacher_kl" in out_yes.metrics
