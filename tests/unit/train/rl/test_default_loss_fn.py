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


def test_default_loss_does_not_propagate_inf_teacher_logprob_at_masked_position():
    """Non-finite teacher_logprobs at non-trainable positions must not poison
    the loss. IEEE 754 has 0.0 * NaN/-Inf = NaN, so the obvious
    `keep_mask * advantages * importance_ratio` formula is not safe by itself
    — teacher_kl must be masked BEFORE folding into advantages.

    Trigger: teacher prefill on an off-policy completion can underflow to -inf
    for tokens the teacher rates impossible. If ANY of those land on a
    loss_mask=False position, NaN propagates through the entire batch's
    gradient.
    """
    inputs = _inputs(
        trainer_lp=[0.0, -1.0, 0.0],
        inference_lp=[0.0, 0.0, 0.0],
        advantages=[1.0, 1.0, 1.0],
        loss_mask=[True, False, True],
        teacher_lp=[0.0, float("-inf"), 0.0],  # -inf at the loss-masked position
    )
    cfg = DefaultLossConfig(dppo_mask_high=10.0, dppo_mask_low=10.0,
                            kl_tau=0.0, adv_tau=1.0, teacher_tau=1.0)
    out = default_loss_fn(inputs, cfg)
    assert torch.isfinite(out.loss), f"loss leaked NaN/Inf via teacher_kl: {out.loss}"


def test_default_loss_does_not_propagate_inf_inference_logprob_at_masked_position():
    """Same hazard on the importance-ratio path. If either trainer_lp or
    inference_lp is non-finite at a loss_mask=False position, the IS ratio
    becomes Inf there; mask must be applied before multiplication.
    """
    inputs = _inputs(
        trainer_lp=[0.0, 0.0, 0.0],
        inference_lp=[0.0, float("-inf"), 0.0],  # -inf at loss_mask=False position
        advantages=[1.0, 1.0, 1.0],
        loss_mask=[True, False, True],
    )
    cfg = DefaultLossConfig(dppo_mask_high=10.0, dppo_mask_low=10.0,
                            kl_tau=1.0, adv_tau=1.0)
    out = default_loss_fn(inputs, cfg)
    assert torch.isfinite(out.loss), f"loss leaked NaN/Inf via importance_ratio: {out.loss}"


def test_default_loss_does_not_propagate_nan_advantage():
    """A single NaN advantage at a trainable position must not NaN the loss.

    `apply_batch_advantage_normalization` raises on non-finite advantages
    but ONLY under `normalization == "batch"`. The `"none"` / `"group"` /
    custom advantage paths can produce NaN (env contract violation,
    divide-by-zero in a length-shaping fn) and the loss-side safe-mask
    must defend belt-and-suspenders.
    """
    inputs = _inputs(
        trainer_lp=[0.0, 0.0, 0.0],
        inference_lp=[0.0, 0.0, 0.0],
        # NaN advantage at trainable position 1.
        advantages=[0.5, float("nan"), 0.5],
        loss_mask=[True, True, True],
    )
    cfg = DefaultLossConfig(dppo_mask_high=10.0, dppo_mask_low=10.0,
                            kl_tau=1e-3, adv_tau=1.0)
    out = default_loss_fn(inputs, cfg)
    assert torch.isfinite(out.loss), (
        f"loss leaked NaN at trainable position with NaN advantage: {out.loss}"
    )


def test_cispo_loss_does_not_propagate_nan_advantage():
    """Same NaN-advantage defense for cispo. Same reachability story."""
    from prime_rl.configs.trainer import CISPOLossConfig
    from prime_rl.trainer.rl.loss import cispo_loss_fn

    inputs = _inputs(
        trainer_lp=[0.0, 0.0, 0.0],
        inference_lp=[0.0, 0.0, 0.0],
        advantages=[0.5, float("nan"), 0.5],
        loss_mask=[True, True, True],
    )
    out = cispo_loss_fn(inputs, CISPOLossConfig(eps_max=4.0, adv_tau=1.0))
    assert torch.isfinite(out.loss), (
        f"cispo loss leaked NaN at trainable position with NaN advantage: {out.loss}"
    )


def test_cispo_loss_does_not_propagate_nan_at_trainable_position():
    """CISPO must mirror default_loss_fn's `torch.isfinite` AND-gate at
    trainable positions. The clamp `truncated_ratio = clamp(rho, max=eps_max)`
    only bounds the upper tail; a NaN flows through clamp unchanged. If both
    `trainer_logprobs[t]` and `inference_logprobs[t]` are simultaneously
    `-inf` (an aliased adapter rollout where inference is several adapters
    off-policy), `(-inf) - (-inf) = NaN` then `exp(NaN) = NaN` and a single
    such position would NaN the entire micro-batch's loss.
    """
    from prime_rl.configs.trainer import CISPOLossConfig
    from prime_rl.trainer.rl.loss import cispo_loss_fn

    inputs = _inputs(
        # Position 1 has both trainer and inference logprobs at -inf, so
        # `exp(trainer - inference) = exp(NaN) = NaN`.
        trainer_lp=[0.0, float("-inf"), 0.0],
        inference_lp=[0.0, float("-inf"), 0.0],
        advantages=[0.5, 0.5, 0.5],
        loss_mask=[True, True, True],
    )
    out = cispo_loss_fn(inputs, CISPOLossConfig(eps_max=4.0, adv_tau=1.0))
    assert torch.isfinite(out.loss), (
        f"cispo loss leaked NaN at trainable position with both logprobs=-inf: {out.loss}"
    )


def test_default_loss_does_not_propagate_inf_inference_logprob_at_TRAINABLE_position():
    """Belt-and-suspenders for the importance-ratio path at TRAINABLE positions.

    The DPPO `keep_mask` (probs_diff > eps) is a trust-region threshold, not a
    finiteness check. A trainable position with inference_logprobs = -inf
    (which an external rollout service can emit for "impossible" tokens, or a
    teacher-rollout path with -inf logprobs) yields:
      probs_diff = trainer_probs - exp(-inf) = trainer_probs ∈ (0, 1]
      |probs_diff| <= dppo_mask_high (low trainer_probs case) → keep_mask=True
      log_importance_ratio = trainer_lp - (-inf) = +inf
      importance_ratio = exp(+inf) = +inf
      kl_loss = +inf**2 = +inf
    With kl_tau > 0, that flows into `loss.sum()` unbounded — `kl_tau * (+inf)²`
    = +inf — and corrupts every parameter's gradient on the next backward.
    The fix mirrors cispo_loss_fn's torch.clamp(..., max=eps_max) belt-and-
    suspenders: AND the keep_mask / loss_mask with `torch.isfinite(...)` so
    non-finite ratios at trainable positions truly contribute zero.
    """
    # trainer_lp = log(0.05) so trainer_probs = 0.05 < dppo_mask_high=0.2
    # and the high-mask does NOT fire (probs_diff = 0.05 - 0 = 0.05 < 0.2).
    # keep_mask is True at this position pre-fix.
    trainer_lp_low = math.log(0.05)
    inputs = _inputs(
        trainer_lp=[0.0, trainer_lp_low, 0.0],
        inference_lp=[0.0, float("-inf"), 0.0],  # -inf at TRAINABLE position
        advantages=[0.5, 0.5, 0.5],
        loss_mask=[True, True, True],
    )
    cfg = DefaultLossConfig(dppo_mask_high=0.2, dppo_mask_low=0.2,
                            kl_tau=1e-3, adv_tau=1.0)
    out = default_loss_fn(inputs, cfg)
    assert torch.isfinite(out.loss), (
        f"loss leaked +inf at trainable position with inference_lp=-inf: {out.loss}"
    )
