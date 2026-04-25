from dataclasses import dataclass
from typing import Callable, Literal

import torch
import verifiers as vf
from jaxtyping import Float, Int
from torch import Tensor

from prime_rl.configs.orchestrator import AdvantageConfig, CustomAdvantageConfig
from prime_rl.orchestrator.vf_utils import get_model_completion_len
from prime_rl.utils.utils import import_object

# Std epsilon for advantage normalization. Small enough to vanish on any non-degenerate
# batch but large enough to keep the per-group "all rewards identical" path bounded.
_NORM_EPS = 1e-8


@dataclass
class AdvantageInputs:
    """Inputs for advantage computation."""

    rewards: Float[Tensor, "num_problems rollouts_per_example"]
    completion_lengths: Int[Tensor, "num_problems rollouts_per_example"]


@dataclass
class AdvantageOutputs:
    """Outputs from advantage computation."""

    advantages: Float[Tensor, "num_problems rollouts_per_example"]


AdvantageFn = Callable[..., AdvantageOutputs]
"""Type for an advantage function.

Expected signature:
    def my_advantage(inputs: AdvantageInputs, **kwargs) -> AdvantageOutputs:
        ...
"""


def default_advantage_fn(
    inputs: AdvantageInputs,
    length_shaping: bool = False,
    normalization: Literal["none", "group", "batch"] = "none",
) -> AdvantageOutputs:
    """Default GRPO advantage: reward minus per-group baseline, with optional std scaling.

    Order of operations is fixed so callers don't accidentally compose them backwards:
        raw rewards
        -> [length-shape if enabled]
        -> per-group baseline subtraction
        -> [std normalization: none | per-group]

    Note: `normalization == "batch"` is INTENTIONALLY not handled here. ScaleRL
    §3.4 / Reinforce++ describes batch-std normalization, but it must be computed
    over the SURVIVING rollouts (after `apply_filters` drops zero-advantage
    groups), otherwise the std drifts step-to-step with the filter rate and
    silently rescales the LR. The orchestrator applies the batch-std step via
    `apply_batch_advantage_normalization` after filtering.
    """
    rewards = inputs.rewards

    if length_shaping:
        completion_lengths = inputs.completion_lengths.to(dtype=rewards.dtype)
        advantages = _efficiency_length_shaping(rewards, completion_lengths)
    else:
        baseline = rewards.mean(dim=1, keepdim=True)
        advantages = rewards - baseline

    if normalization == "none" or normalization == "batch":
        # "batch" deferred to post-filter step. Emit the baseline-subtracted
        # tensor unchanged; the orchestrator finishes the job.
        pass
    elif normalization == "group":
        std = advantages.std(dim=1, keepdim=True, unbiased=False)
        advantages = advantages / (std + _NORM_EPS)
    else:
        # Pydantic Literal validates at config load; this guards direct callers.
        raise ValueError(f"Unknown normalization mode: {normalization!r}")

    return AdvantageOutputs(advantages=advantages)


def apply_batch_advantage_normalization(
    rollouts: list[vf.RolloutOutput],
    advantage_config: AdvantageConfig | None,
) -> None:
    """Post-filter step for `DefaultAdvantageConfig.normalization == "batch"`.

    Computes the std over the advantages of UNFILTERED rollouts only, divides
    every UNFILTERED rollout's advantage by that std + eps. Filtered rollouts
    are left untouched (they don't enter training; their advantage value is
    irrelevant). No-op for non-default advantage configs and for
    `normalization in {"none", "group"}` — those are already finalized in
    `default_advantage_fn`.

    Faithful to ScaleRL §3.4: the surviving gradients are scaled by their own
    cohort, not by an artifact of how many groups got filtered.
    """
    if advantage_config is None:
        return
    if not hasattr(advantage_config, "normalization"):
        return  # CustomAdvantageConfig — caller is on their own.
    if advantage_config.normalization != "batch":
        return

    surviving = [r for r in rollouts if not r.get("is_filtered", False)]
    if len(surviving) < 2:
        return  # std undefined / unstable; leave advantages as-is.

    advs = torch.tensor([float(r["advantage"]) for r in surviving])
    std = advs.std(unbiased=False).item()
    scale = 1.0 / (std + _NORM_EPS)
    for r in surviving:
        r["advantage"] = float(r["advantage"]) * scale


def _efficiency_length_shaping(
    rewards: Float[Tensor, "num_problems rollouts_per_example"],
    completion_lengths: Float[Tensor, "num_problems rollouts_per_example"],
) -> Float[Tensor, "num_problems rollouts_per_example"]:
    """Correctness-gated length shaping with bounded advantages.

    Shapes rewards with a bounded brevity bonus before standard GRPO subtraction,
    preserving zero-mean advantages per group.

    Correct rollouts get reward amplified by up to 2x based on relative brevity.
    Incorrect rollouts are untouched. Shorter correct rollouts get higher advantage.
    """
    max_reward = rewards.max(dim=1, keepdim=True).values
    correct_mask = rewards >= max_reward
    num_correct = correct_mask.sum(dim=1, keepdim=True)

    # No shaping when max reward is 0 — no correct rollouts to differentiate
    has_correct = max_reward > 0

    # Mean length of correct rollouts per problem
    correct_lengths = completion_lengths * correct_mask
    mean_correct_len = correct_lengths.sum(dim=1, keepdim=True) / num_correct.clamp(min=1)

    # Bounded brevity bonus: [0, 1], positive for below-average length, zero for above
    bonus = (1 - completion_lengths / mean_correct_len).clamp(0, 1)

    # Shape rewards: correct rollouts amplified by up to 2x, incorrect untouched
    shaped_rewards = rewards * (1 + bonus * correct_mask)
    baseline = shaped_rewards.mean(dim=1, keepdim=True)

    shaped = shaped_rewards - baseline
    unshaped = rewards - rewards.mean(dim=1, keepdim=True)
    return torch.where(has_correct, shaped, unshaped)


def setup_advantage_fn(config: AdvantageConfig) -> AdvantageFn:
    """Setup advantage function from config."""
    if isinstance(config, CustomAdvantageConfig):
        custom_fn = import_object(config.import_path)
        kwargs = config.kwargs

        def advantage_fn(inputs: AdvantageInputs) -> AdvantageOutputs:
            return custom_fn(inputs, **kwargs)

        return advantage_fn

    def advantage_fn(inputs: AdvantageInputs) -> AdvantageOutputs:
        return default_advantage_fn(
            inputs,
            length_shaping=config.length_shaping,
            normalization=config.normalization,
        )

    return advantage_fn


def compute_advantages(
    rollouts: list[vf.RolloutOutput],
    samples_per_problem: int,
    advantage_config: AdvantageConfig | None,
) -> None:
    """
    Computes advantages from rollouts, grouped by problem.
    Stores advantages in-place on the rollouts.

    Args:
        rollouts: List of rollouts to store advantages on
        samples_per_problem: Number of samples (and thus, rewards) per problem
        advantage_config: Configuration for advantage computation (DefaultAdvantageConfig or CustomAdvantageConfig)
    """
    rewards = [r["reward"] for r in rollouts]

    if not advantage_config:
        for rollout, reward in zip(rollouts, rewards):
            rollout["advantage"] = reward
        return

    advantage_fn = setup_advantage_fn(advantage_config)
    completion_lengths = [get_model_completion_len(r) for r in rollouts]

    inputs = AdvantageInputs(
        rewards=torch.tensor(rewards).view(-1, samples_per_problem),
        completion_lengths=torch.tensor(completion_lengths).view(-1, samples_per_problem),
    )

    result = advantage_fn(inputs)
    advantages = result.advantages.flatten().tolist()

    for rollout, advantage in zip(rollouts, advantages):
        rollout["advantage"] = advantage
