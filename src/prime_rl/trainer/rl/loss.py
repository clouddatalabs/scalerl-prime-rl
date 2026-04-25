from dataclasses import dataclass
from typing import Any, Callable

import torch
from beartype import beartype as typechecker
from jaxtyping import Bool, Float, Int, jaxtyped
from torch import Tensor

from prime_rl.configs.trainer import (
    CISPOLossConfig,
    CustomLossConfig,
    DefaultLossConfig,
    LossConfig,
    LossScaleMode,
    SFTLossConfig,
)
from prime_rl.utils.utils import import_object


@dataclass
class LossInputs:
    """Inputs for computing loss on a single sample."""

    trainer_logprobs: Float[Tensor, " seq"]
    inference_logprobs: Float[Tensor, " seq"]
    teacher_logprobs: Float[Tensor, " seq"] | None
    advantages: Float[Tensor, " seq"]
    loss_mask: Bool[Tensor, " seq"]


@dataclass
class LossOutputs:
    """Outputs from computing loss on a single sample."""

    loss: Float[Tensor, ""]
    metrics: dict[str, Tensor]


LossFn = Callable[..., LossOutputs]
"""Type for a per-sample loss function.

Expected signature:
    def my_loss(inputs: LossInputs, **kwargs) -> LossOutputs:
        ...
"""


@jaxtyped(typechecker=typechecker)
@torch.compile(dynamic=True)
def selective_log_softmax(
    logits: Float[Tensor, "batch seq vocab"], index: Int[Tensor, "batch seq"]
) -> Float[Tensor, "batch seq"]:
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


@jaxtyped(typechecker=typechecker)
@torch.compile(dynamic=True)
def compute_entropy(shifted_logits: Float[Tensor, "batch seq vocab"]) -> Float[Tensor, "batch seq"]:
    with torch.no_grad():
        pd = torch.nn.functional.softmax(shifted_logits, dim=-1)
        entropy = torch.logsumexp(shifted_logits, dim=-1) - torch.sum(pd * shifted_logits, dim=-1)
    return entropy


@jaxtyped(typechecker=typechecker)
def shift_logits(
    logits: Float[Tensor, "batch seq vocab"], left_pad_logit: Float[Tensor, "batch 1 vocab"] | None = None
) -> Float[Tensor, "batch seq vocab"]:
    """Removes final token logits and adds a left pad logit for the first token."""
    # We drop the last logit because it corresponds to the next token that will be sampled but is not here yet
    batch, seq, vocab = logits.shape
    logits = logits[:, :-1, :]  # (batch, seq-1, vocab)
    if left_pad_logit is None:
        left_pad_logit = torch.zeros(batch, 1, vocab, device=logits.device, dtype=logits.dtype)  # (batch, 1, vocab)
    logits = torch.cat([left_pad_logit, logits], dim=1)  # (batch, seq, vocab)
    return logits


def shift_tensor_left(t: Float[Tensor, "batch seq"]) -> Float[Tensor, "batch seq"]:
    """Shifts the tensor one token to the left.

    Used to create labels from input_ids: labels[i] = input_ids[i+1].
    The last position is padded with 0 (a valid token index) since this value
    will be shifted off by shift_tensor_right and never used.
    """
    return torch.cat([t[:, 1:], torch.full((t.shape[0], 1), 0, device=t.device, dtype=t.dtype)], dim=1)


def shift_tensor_right(t: Float[Tensor, "batch seq"], pad_value: float | None = None) -> Float[Tensor, "batch seq"]:
    """Shifts the tensor one token to the right, prepending a padding value.

    Used to realign logprobs/entropy after computing with shifted labels.
    After shift: result[i] = t[i-1], result[0] = pad_value.
    This converts from "predict next token" convention to "probability of current token" convention.

    Args:
        t: Tensor to shift right
        pad_value: Value to use for position 0. If None, uses 0.0 for backward compatibility.
                   For logprobs, should be log(1/vocab_size) to represent uniform distribution.
                   For entropy, should be log(vocab_size) to represent maximum entropy.
    """
    if pad_value is None:
        pad_value = 0.0
    return torch.cat([torch.full((t.shape[0], 1), pad_value, device=t.device, dtype=t.dtype), t[:, :-1]], dim=1)


def _safe_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean of values over a boolean mask; returns 0 when mask is empty."""
    denom = torch.clamp_min(mask.sum(), 1)
    return values[mask].sum() / denom


def default_loss_fn(inputs: LossInputs, loss_config: DefaultLossConfig) -> LossOutputs:
    """
    DPPO+KL loss, combining:
    - DPPO-Binary TV Loss (https://arxiv.org/pdf/2602.04879)
    - Kimi-K2.5 KL Loss (https://arxiv.org/pdf/2602.02276)

    The mask is conditioned on the advantage sign: for positive advantages,
    we mask tokens whose probability increased too much (trust region violation
    in the upweight direction); for negative advantages, we mask tokens whose
    probability decreased too much (trust region violation in the downweight
    direction).
    """
    trainer_logprobs = inputs.trainer_logprobs
    inference_logprobs = inputs.inference_logprobs
    teacher_logprobs = inputs.teacher_logprobs
    advantages = inputs.advantages
    loss_mask = inputs.loss_mask

    trainer_probs = torch.exp(trainer_logprobs)
    inference_probs = torch.exp(inference_logprobs)
    probs_diff = trainer_probs - inference_probs
    dppo_invalid_mask_high = probs_diff > loss_config.dppo_mask_high
    dppo_invalid_mask_low = probs_diff < -loss_config.dppo_mask_low
    dppo_invalid_mask = torch.where(advantages > 0, dppo_invalid_mask_high, dppo_invalid_mask_low)

    is_masked = dppo_invalid_mask
    is_masked_high = (advantages > 0) & dppo_invalid_mask_high
    is_masked_low = (advantages < 0) & dppo_invalid_mask_low
    keep_mask = loss_mask & ~is_masked

    log_importance_ratio = trainer_logprobs - inference_logprobs
    importance_ratio = torch.exp(log_importance_ratio)
    mismatch_kl = importance_ratio - log_importance_ratio - 1

    advantages = loss_config.adv_tau * advantages
    if teacher_logprobs is not None:
        teacher_kl = teacher_logprobs - trainer_logprobs
        # Zero out teacher_kl at non-trainable positions BEFORE folding into
        # advantages. The downstream `keep_mask * advantages * importance_ratio`
        # relies on `0.0 * x == 0.0`, but IEEE 754 has `0.0 * NaN == NaN` and
        # `0.0 * ±Inf == NaN`. Teacher inference (especially under the off-policy
        # ScaleRL §3.5 KL-distillation regime with max_async_level=8) can emit
        # `-inf` logprobs at non-trainable positions for tokens it rates as
        # impossible — those would silently poison `advantages` and propagate
        # NaN through the entire batch's gradient. Mask first, fold second.
        teacher_kl = torch.where(loss_mask, teacher_kl, torch.zeros_like(teacher_kl))
        advantages = advantages + loss_config.teacher_tau * teacher_kl.detach()
    else:
        teacher_kl = None

    # Same IEEE 0*NaN=NaN hazard on the importance-ratio path: if either
    # `trainer_logprobs` or `inference_logprobs` is non-finite at a non-trainable
    # position (vLLM occasionally emits ±inf for impossible tokens at packed
    # prefixes; the trainer FP32 LM head normally produces finite values but
    # `selective_log_softmax` can return -inf when `labels` indexes a position
    # the masked-fill set to -inf), the unmasked product propagates NaN. Mask
    # before multiplying so `keep_mask=False` positions truly contribute zero.
    safe_importance_ratio = torch.where(
        keep_mask, importance_ratio, torch.zeros_like(importance_ratio)
    )
    safe_advantages = torch.where(keep_mask, advantages, torch.zeros_like(advantages))
    pg_loss = keep_mask * safe_advantages * safe_importance_ratio
    safe_log_importance_ratio_sq = torch.where(
        loss_mask, log_importance_ratio**2, torch.zeros_like(log_importance_ratio)
    )
    kl_loss = loss_mask * safe_log_importance_ratio_sq
    loss = (-pg_loss + loss_config.kl_tau * kl_loss).sum()

    metrics = {
        "mismatch_kl": _safe_mean(mismatch_kl, loss_mask),  # all trainable tokens
        "masked_mismatch_kl": _safe_mean(mismatch_kl, loss_mask & is_masked),  # all trainable, masked tokens
        "unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),  # all trainable, unmasked tokens
        "is_masked": _safe_mean(is_masked, loss_mask),
        "is_masked_low": _safe_mean(is_masked_low, loss_mask),
        "is_masked_high": _safe_mean(is_masked_high, loss_mask),
    }
    if teacher_kl is not None:
        metrics["teacher_kl"] = _safe_mean(teacher_kl, loss_mask)

    return LossOutputs(loss=loss, metrics=metrics)


def cispo_loss_fn(inputs: LossInputs, loss_config: CISPOLossConfig) -> LossOutputs:
    """Canonical Minimax-CISPO loss — REINFORCE with stop-gradient, upper-truncated IS weight.

    Per token (trainable mask only):
        L_t = -sg(min(rho_t, eps_max)) * adv * log pi_theta(y_t)

    rho_t = pi_train / pi_gen (ratio of fresh trainer logprobs to logprobs the inference
    engine actually emitted at sampling time). The stop-gradient detaches the ratio so
    only log pi_theta carries gradient. No lower clip, no token-level masking — that's
    the whole point versus PPO/DAPO.

    Returns the per-sample sum; the batch-level reduction (token-mean / sequence-weight)
    happens in compute_loss.
    """
    importance_ratio = torch.exp(inputs.trainer_logprobs - inputs.inference_logprobs)
    truncated_ratio = torch.clamp(importance_ratio, max=loss_config.eps_max).detach()

    advantages = loss_config.adv_tau * inputs.advantages
    # Mirror the IEEE 0.0 * NaN/Inf = NaN defense applied in default_loss_fn.
    # Mask EVERY non-mask factor so any single non-finite value at a
    # non-trainable position truly contributes zero — `False * finite * NaN`
    # still evaluates to NaN under IEEE 754. On the shipped FP32 LM-head +
    # Qwen3-8B path inference_logprobs is hard-zero at non-trainable
    # positions (orchestrator/trajectories.py:377) so the cross-product is
    # finite, but the symmetry with default_loss_fn matters as a defense
    # against future schema changes that introduce non-finite values
    # upstream (e.g. external generators emitting -inf at impossible
    # tokens).
    safe_trainer_logprobs = torch.where(
        inputs.loss_mask, inputs.trainer_logprobs, torch.zeros_like(inputs.trainer_logprobs)
    )
    safe_truncated_ratio = torch.where(
        inputs.loss_mask, truncated_ratio, torch.zeros_like(truncated_ratio)
    )
    safe_advantages = torch.where(
        inputs.loss_mask, advantages, torch.zeros_like(advantages)
    )
    pg_per_token = inputs.loss_mask * safe_truncated_ratio * safe_advantages * safe_trainer_logprobs
    loss = -pg_per_token.sum()

    metrics = {
        # Mean of the POST-CLAMP ratio. Bounded above by `eps_max`, NOT the true
        # generator/trainer probability ratio — a heavily-off-policy step that
        # pushes true rho above eps_max saturates this metric at the cap. The
        # `ratio_truncated` companion metric is what reports magnitude divergence
        # in that case; this metric reports the clamped value used by the loss
        # itself. Named to be honest about that — a previous label
        # ("importance_ratio") read like the unclipped ratio on dashboards.
        "truncated_importance_ratio": _safe_mean(truncated_ratio, inputs.loss_mask),
        # Fraction of trainable tokens whose ratio hit the eps_max ceiling. Useful
        # as a sanity gauge — too high (e.g. > ~10%) suggests the recipe drifted
        # off-policy beyond what the truncation bound was tuned for.
        "ratio_truncated": _safe_mean((importance_ratio > loss_config.eps_max).float(), inputs.loss_mask),
        # Lower-tail companion: fraction of trainable tokens whose ratio
        # collapsed to ~0 (rho < 1/eps_max). CISPO's upper-only clamp means
        # those tokens are effectively zero-gradient (REINFORCE × 0); without
        # this metric, async_level=8 + slow weight broadcast can silently
        # down-weight a large fraction of the batch with no dashboard
        # signal — `ratio_truncated` only fires on the upper tail.
        "ratio_collapsed": _safe_mean(
            (importance_ratio < (1.0 / loss_config.eps_max)).float(),
            inputs.loss_mask,
        ),
    }
    return LossOutputs(loss=loss, metrics=metrics)


def sft_loss_fn(inputs: LossInputs) -> LossOutputs:
    """SFT-style masked negative log-likelihood over trainable tokens."""
    trainer_logprobs = inputs.trainer_logprobs
    loss_mask = inputs.loss_mask

    loss = -(trainer_logprobs[loss_mask]).sum()
    metrics = {
        "nll": _safe_mean(-trainer_logprobs, loss_mask),
    }
    return LossOutputs(loss=loss, metrics=metrics)


def setup_loss_fn(loss_config: LossConfig) -> LossFn:
    """Setup the loss function based on config.

    The returned callable carries `.loss_scale_mode` so train.py can pick the right
    reduction in compute_loss without a second config read.
    """
    if isinstance(loss_config, CustomLossConfig):
        custom_fn = import_object(loss_config.import_path)
        kwargs = loss_config.kwargs

        def loss_fn(inputs: LossInputs) -> LossOutputs:
            return custom_fn(inputs, **kwargs)

        loss_fn.loss_scale_mode = loss_config.loss_scale_mode
        return loss_fn

    if isinstance(loss_config, SFTLossConfig):
        # Wrap in a fresh closure rather than mutating the module-level
        # `sft_loss_fn`. Two consecutive `setup_loss_fn(SFTLossConfig(...))`
        # calls with different `loss_scale_mode` values would otherwise share
        # the same function object, and the second call would silently overwrite
        # the first's mode for any caller still holding a prior reference.
        def loss_fn(inputs: LossInputs) -> LossOutputs:
            return sft_loss_fn(inputs)

        loss_fn.loss_scale_mode = loss_config.loss_scale_mode
        return loss_fn

    if isinstance(loss_config, CISPOLossConfig):
        def loss_fn(inputs: LossInputs) -> LossOutputs:
            return cispo_loss_fn(inputs, loss_config)

        loss_fn.loss_scale_mode = loss_config.loss_scale_mode
        return loss_fn

    def loss_fn(inputs: LossInputs) -> LossOutputs:
        return default_loss_fn(inputs, loss_config)

    loss_fn.loss_scale_mode = loss_config.loss_scale_mode
    return loss_fn


def compute_loss(
    trainer_logprobs: list[Float[Tensor, " seq_i"]],
    inference_logprobs: list[Float[Tensor, " seq_i"]],
    teacher_logprobs: list[Float[Tensor, " seq_i"]] | None,
    advantages: list[Float[Tensor, " seq_i"]],
    loss_mask: list[Bool[Tensor, " seq_i"]],
    loss_fn: LossFn,
    loss_scale: int,
    sequence_loss_weights: list[float] | None = None,
    loss_scale_mode: LossScaleMode | None = None,
    fsdp_gradient_divide_factor: int = 1,
) -> tuple[Float[Tensor, ""], dict[str, Any]]:
    """
    Compute loss for packed sequences (batch size = 1, multiple sequences packed along sequence dimension).

    Args:
        trainer_logprobs: Log probabilities for each sequence
        inference_logprobs: Reference log probabilities for each sequence
        teacher_logprobs: Teacher log probabilities for each sequence, or None
        advantages: Advantages for each sequence
        loss_mask: Loss mask for each sequence
        loss_fn: Per-sequence loss function
        loss_scale: Scale factor for the "token" path (typically total trainable tokens).
        sequence_loss_weights: Optional per-sequence weights. When supplied alongside
            `loss_scale_mode in {"sequence", "none"}` the loss reduces as
            `sum_i w_i * loss_i`, with the orchestrator/packer responsible for
            encoding the desired normalization in the weights (e.g. ScaleRL
            prompt-level averaging via apply_prompt_average_sequence_weights).
        loss_scale_mode: How to combine per-sequence losses. Defaults to
            `getattr(loss_fn, "loss_scale_mode", "token")` so call sites that
            don't pass it inherit the loss-fn's preference.
        fsdp_gradient_divide_factor: the divisor that PyTorch FSDP applies
            during its all-reduce of gradients — `dp_replicate * dp_shard * cp`
            on the (dp_replicate, dp_shard_cp) hsdp mesh. Pass exactly
            `parallel_dims.fsdp_gradient_divide_factor`. NOT the same as
            `dist.get_world_size()` if you ever add pipeline parallel — PP
            does not reduce gradients across PP ranks, so `world_size` would
            shrink the gradient by `1/pp` after FSDP averaging. Only consulted
            in `sequence`/`none` mode; defaults to 1 so single-process tests
            stay simple.

    Returns:
        Tuple of (scaled_loss, aggregated_metrics)

    NOT-VERIFIED — cp>1 with `sequence`/`none` mode:
        The scaling on the `else` branch below multiplies by
        `fsdp_gradient_divide_factor = dp * cp` to cancel FSDP's gradient
        averaging. Under context parallelism, the upstream `gather_for_cp`
        step uses `dist_nn.all_gather`, whose backward is a SUM-reduce-scatter.
        When all CP ranks compute identical loss (which they do, since the
        per-rank `total_loss` is a global weighted sum after the gather), the
        gather backward injects an additional factor of `cp` on local
        activation gradients, compounding with the `dp*cp` loss multiplier
        and FSDP's `1/(dp*cp)` divisor to leave a residual factor of `cp`
        on parameter gradients vs. the desired single-process baseline.
        This branch has only been exercised under cp=1 in this fork; cp>1
        sequence/none-mode scaling has NOT been empirically verified against
        a single-process baseline. Do not deploy a cp>1 sequence/none-mode
        run without first running such a verification (e.g. compare gradient
        norms / final losses on a tiny config with cp=1 vs cp=2 holding
        global batch fixed). All shipped configs in this fork use cp=1, so
        the open question is latent for current users.
    """
    # Initialize as a 0-d tensor so an empty packed batch (len(trainer_logprobs)==0)
    # still returns a Tensor with a real device, not a Python float. A Python-float
    # `total_loss` would crash `loss.backward()` with `AttributeError: 'float'
    # object has no attribute 'backward'` and on multi-rank DP that crash is
    # asymmetric (other ranks have data) — NCCL deadlocks instead of a clean fail.
    if len(trainer_logprobs) > 0:
        device = trainer_logprobs[0].device
        dtype = trainer_logprobs[0].dtype
    else:
        device, dtype = torch.device("cpu"), torch.float32
    total_loss: Tensor = torch.zeros((), device=device, dtype=dtype)
    all_metrics: dict[str, list[Tensor]] = {}

    if teacher_logprobs is None:
        teacher_logprobs = [None] * len(trainer_logprobs)

    if loss_scale_mode is None:
        loss_scale_mode = getattr(loss_fn, "loss_scale_mode", "token")

    weights = sequence_loss_weights if sequence_loss_weights else None
    if weights is not None and len(weights) != len(trainer_logprobs):
        raise ValueError(
            f"sequence_loss_weights length {len(weights)} does not match "
            f"number of packed sequences {len(trainer_logprobs)}"
        )
    # Refuse empty weights under sequence/none modes when there ARE samples to
    # weight. Empty `[]` is the packer's "no weights ever set" sentinel; if it
    # reaches a sequence/none reduction with non-empty samples the loop below
    # would silently fall through `if weights is not None` and emit an
    # un-normalized dp-scaled raw sum — the unweighted objective the
    # prompt-average cross-validator was added to prevent. The packer fills
    # `[1.0]` per-sample by default, so production should never hit this.
    if (
        loss_scale_mode in ("sequence", "none")
        and weights is None
        and len(trainer_logprobs) > 0
    ):
        raise ValueError(
            "compute_loss: `loss_scale_mode='sequence'/'none'` requires "
            "non-empty `sequence_loss_weights` (one float per packed sample). "
            "Empty weights would produce an unweighted dp-scaled raw sum. "
            "The packer should populate weights via "
            "apply_prompt_average_sequence_weights or default `[1.0]*N`."
        )
    # Reject non-trivial weights under token mode: the loop below would
    # silently ignore them (only `sequence`/`none` consume the weights),
    # producing a token-mean reduction the caller didn't ask for. The
    # config-level cross-validator catches the canonical
    # `prompt_average_loss=True + loss_scale_mode='token'` case at load
    # time; this guard protects direct callers (custom losses, tests).
    # Trivial weights `[1.0]*N` change nothing, so allow them through —
    # that's the packer's neutral default for token-mode runs.
    if weights is not None and loss_scale_mode == "token" and any(w != 1.0 for w in weights):
        raise ValueError(
            "compute_loss: non-trivial `sequence_loss_weights` were provided "
            "with `loss_scale_mode='token'`. Token mode reduces by trainable "
            "tokens and ignores per-sequence weights — they would be silently "
            "dropped. Use loss_scale_mode='sequence' or 'none', or omit weights."
        )

    for idx, (t_logp, i_logp, teach_logp, adv, mask) in enumerate(
        zip(trainer_logprobs, inference_logprobs, teacher_logprobs, advantages, loss_mask)
    ):
        inputs = LossInputs(
            trainer_logprobs=t_logp,
            inference_logprobs=i_logp,
            teacher_logprobs=teach_logp,
            advantages=adv,
            loss_mask=mask,
        )

        result = loss_fn(inputs)

        sample_loss = result.loss
        if weights is not None and loss_scale_mode in ("sequence", "none"):
            sample_loss = sample_loss * float(weights[idx])

        total_loss = total_loss + sample_loss

        for k, v in result.metrics.items():
            if k not in all_metrics:
                all_metrics[k] = []
            all_metrics[k].append(v)

    if loss_scale_mode == "token":
        scaled_loss = total_loss / loss_scale
    else:
        # "sequence" / "none": weights already encode normalization. The packer
        # set them globally (e.g. 1/(total_p * num_prompts_global) for ScaleRL
        # prompt-level averaging), so each rank's `total_loss` is its slice of
        # the global weighted sum.
        #
        # FSDP all-reduces gradients with `gradient_divide_factor = dp_world_size`
        # (PyTorch default), turning the rank's gradient into the average across
        # ranks. With locally-summed losses that would silently produce
        # `aggregate = global_sum / dp_world_size` — i.e. effective LR shrinks
        # 1/dp_world_size in multi-rank DP. Multiplying by dp_world_size here
        # cancels the FSDP divisor so `aggregate_grad == sum_global(w_i * grad_i)`.
        #
        # ⚠️ NOT-VERIFIED for cp>1: this scaling has only been exercised at
        # cp=1 in this fork. The cp>1 path interacts with `gather_for_cp`'s
        # SUM-reduce-scatter backward in ways that may leave a residual `cp`
        # factor on parameter gradients (see the docstring note above). All
        # shipped configs use cp=1; do not deploy cp>1 sequence/none-mode
        # without an empirical single-process-vs-cp comparison first.
        # Token-mode is approximately unaffected: each rank divides by its
        # local trainable tokens, which only equals `global_tokens / dp_world_size`
        # under perfectly balanced packing. The packer pads micro-batch *count*
        # per rank, not token count, so per-rank token counts can drift; what
        # token-mode actually computes after the FSDP all-reduce is
        # `mean_rank(local_loss / local_tokens)`, which approximates but does
        # not exactly equal `global_loss / global_tokens`. Same as upstream;
        # call out if you ever change loss_scale here.
        if not isinstance(fsdp_gradient_divide_factor, int) or fsdp_gradient_divide_factor < 1:
            raise ValueError(
                "compute_loss(loss_scale_mode='sequence'/'none') requires a "
                "positive integer `fsdp_gradient_divide_factor` to cancel FSDP's "
                "gradient_divide_factor. Got "
                f"fsdp_gradient_divide_factor={fsdp_gradient_divide_factor!r}. Production callers should "
                "pass `parallel_dims.fsdp_gradient_divide_factor`; tests can pass 1."
            )
        scaled_loss = total_loss * float(fsdp_gradient_divide_factor)

    aggregated: dict[str, Any] = {}
    for k, v in all_metrics.items():
        if v[0].dim() == 0:
            aggregated[k] = torch.stack(v)
        else:
            aggregated[k] = torch.cat(v)

    return scaled_loss, aggregated
