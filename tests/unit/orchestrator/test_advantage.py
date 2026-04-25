import pytest
import torch

from prime_rl.configs.orchestrator import CustomAdvantageConfig, DefaultAdvantageConfig
from prime_rl.orchestrator.advantage import (
    _NORM_EPS,
    AdvantageInputs,
    AdvantageOutputs,
    _efficiency_length_shaping,
    apply_batch_advantage_normalization,
    compute_advantages,
    default_advantage_fn,
    setup_advantage_fn,
)


def test_default_advantage_fn_simple_mean():
    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 0.5, 0.8], [0.2, 0.9, 0.1]]),
        completion_lengths=torch.tensor([[10, 12, 8], [15, 11, 9]]),
    )
    result = default_advantage_fn(inputs)

    assert result.advantages.shape == (2, 3)
    # Check that mean is subtracted per row
    assert torch.allclose(result.advantages.mean(dim=1), torch.zeros(2), atol=1e-6)


def test_efficiency_mixed_group():
    """Mixed group: reward shaping preserves zero-mean, shorter correct gets higher advantage."""
    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 1.0, 0.0, 1.0]]),
        completion_lengths=torch.tensor([[10, 30, 20, 20]]),
    )
    result = default_advantage_fn(inputs, length_shaping=True)

    # mean_correct_len = (10+30+20)/3 = 20
    # bonus = clamp(1 - [10,30,20,20]/20, 0, 1) = [0.5, 0, 0, 0]
    # shaped_rewards = R * (1 + bonus * correct_mask) = [1.5, 1, 0, 1]
    # baseline = mean(shaped_rewards) = 0.875
    # A = shaped_rewards - baseline = [0.625, 0.125, -0.875, 0.125]
    expected = torch.tensor([[0.625, 0.125, -0.875, 0.125]])
    assert torch.allclose(result.advantages, expected, atol=1e-6)

    # Zero-mean per group
    assert torch.allclose(result.advantages.mean(dim=1), torch.zeros(1), atol=1e-6)

    # All correct rollouts have positive advantage
    correct_mask = inputs.rewards[0] >= 1.0
    assert (result.advantages[0][correct_mask] > 0).all()


def test_efficiency_all_correct_group():
    """All-correct group: zero-mean, shorter gets higher advantage."""
    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 1.0, 1.0]]),
        completion_lengths=torch.tensor([[10, 20, 40]]),
    )
    result = default_advantage_fn(inputs, length_shaping=True)

    # mean_len = 70/3 ≈ 23.33
    # bonus = clamp(1 - [10, 20, 40] / (70/3), 0, 1) = [4/7, 1/7, 0]
    # shaped_rewards = [1+4/7, 1+1/7, 1] = [11/7, 8/7, 1]
    # baseline = mean = (11/7 + 8/7 + 1) / 3 = (11+8+7)/(7*3) = 26/21
    # A = shaped - baseline
    shaped = torch.tensor([[11.0 / 7, 8.0 / 7, 1.0]])
    baseline = shaped.mean(dim=1, keepdim=True)
    expected = shaped - baseline
    assert torch.allclose(result.advantages, expected, atol=1e-6)

    # Zero-mean
    assert torch.allclose(result.advantages.mean(dim=1), torch.zeros(1), atol=1e-6)

    # Shortest has highest advantage
    assert result.advantages[0, 0] > result.advantages[0, 1] > result.advantages[0, 2]


def test_efficiency_all_zero_rewards():
    """When all rewards are 0, no length shaping — falls back to standard GRPO."""
    inputs = AdvantageInputs(
        rewards=torch.tensor([[0.0, 0.0, 0.0]]),
        completion_lengths=torch.tensor([[10, 20, 15]]),
    )
    result_with = default_advantage_fn(inputs, length_shaping=True)
    result_without = default_advantage_fn(inputs)

    assert torch.allclose(result_with.advantages, result_without.advantages, atol=1e-6)


def test_efficiency_single_correct():
    """Single correct rollout: bonus=0 (at its own mean), same as standard GRPO."""
    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        completion_lengths=torch.tensor([[100, 50, 200, 150]]),
    )
    result = default_advantage_fn(inputs, length_shaping=True)

    expected = torch.tensor([[0.75, -0.25, -0.25, -0.25]])
    assert torch.allclose(result.advantages, expected, atol=1e-6)


def test_efficiency_shorter_correct_higher_advantage():
    """Among correct rollouts in a mixed group, shorter always gets higher advantage."""
    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]]),
        completion_lengths=torch.tensor([[50, 100, 200, 80, 120]]),
    )
    result = default_advantage_fn(inputs, length_shaping=True)

    advs = result.advantages[0]
    assert advs[0] > advs[1] > advs[2]
    assert (advs[:3] > 0).all()
    assert (advs[3:] < 0).all()


def test_efficiency_zero_mean_per_group():
    """Reward shaping preserves zero-mean advantages per group."""
    inputs = AdvantageInputs(
        rewards=torch.tensor(
            [
                [1.0, 1.0, 0.0, 1.0],  # mixed
                [1.0, 1.0, 1.0, 1.0],  # all correct
            ]
        ),
        completion_lengths=torch.tensor(
            [
                [10, 30, 20, 20],
                [10, 20, 40, 80],
            ]
        ),
    )
    result = default_advantage_fn(inputs, length_shaping=True)

    assert torch.allclose(result.advantages.mean(dim=1), torch.zeros(2), atol=1e-6)


def test_efficiency_amplification_bounded():
    """Even with extreme length outliers, reward amplification is capped at 2x."""
    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 1.0, 0.0]]),
        completion_lengths=torch.tensor([[1, 10000, 5000]]),
    )
    result = default_advantage_fn(inputs, length_shaping=True)

    # Shortest correct gets bonus ≈ 1, so shaped_reward ≈ 2
    # Standard reward = 1, so amplification ≈ 2x
    # shaped_rewards ≈ [2, 1, 0], baseline ≈ 1, max advantage ≈ 1
    assert result.advantages[0, 0] < 1.0 + 1e-3


def test_efficiency_multiple_problems():
    """Handles multiple problems independently."""
    inputs = AdvantageInputs(
        rewards=torch.tensor(
            [
                [1.0, 1.0, 0.0],  # mixed
                [1.0, 1.0, 1.0],  # all correct
            ]
        ),
        completion_lengths=torch.tensor(
            [
                [10, 20, 15],
                [10, 20, 40],
            ]
        ),
    )
    result = default_advantage_fn(inputs, length_shaping=True)

    # Row 0: mixed group — shorter correct > longer correct
    assert result.advantages[0, 0] > result.advantages[0, 1]
    assert (result.advantages[0, :2] > 0).all()
    assert result.advantages[0, 2] < 0

    # Row 1: all-correct group — shorter gets higher advantage
    assert result.advantages[1, 0] > result.advantages[1, 1] > result.advantages[1, 2]

    # Both rows have zero-mean
    assert torch.allclose(result.advantages.mean(dim=1), torch.zeros(2), atol=1e-6)


def _make_rollout(reward: float, completion_len: int) -> dict:
    """Create a minimal rollout dict for advantage testing."""
    return {
        "reward": reward,
        "trajectory": [{"tokens": {"prompt_ids": [0], "completion_ids": list(range(completion_len))}}],
    }


def test_compute_advantages_with_config():
    rewards = [1.0, 0.5, 0.8, 0.2, 0.9, 0.1]
    lengths = [10, 12, 8, 15, 11, 9]
    rollouts = [_make_rollout(r, l) for r, l in zip(rewards, lengths)]

    compute_advantages(rollouts, samples_per_problem=3, advantage_config=DefaultAdvantageConfig())

    advantages = [r["advantage"] for r in rollouts]
    assert len(advantages) == 6
    assert abs(sum(advantages[:3])) < 1e-5
    assert abs(sum(advantages[3:])) < 1e-5


def test_compute_advantages_without_config():
    rewards = [1.0, 0.5, 0.8]
    lengths = [10, 12, 8]
    rollouts = [_make_rollout(r, l) for r, l in zip(rewards, lengths)]

    compute_advantages(rollouts, samples_per_problem=3, advantage_config=None)

    advantages = [r["advantage"] for r in rollouts]
    assert advantages == rewards


def test_setup_advantage_fn_with_custom_config():
    config = CustomAdvantageConfig(
        import_path="tests.unit.orchestrator.test_advantage._dummy_custom_advantage",
        kwargs={"scale": 2.0},
    )
    advantage_fn = setup_advantage_fn(config)

    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 0.5, 0.8]]),
        completion_lengths=torch.tensor([[10, 12, 8]]),
    )

    result = advantage_fn(inputs)
    assert isinstance(result, AdvantageOutputs)
    assert torch.allclose(result.advantages, torch.tensor([[2.0, 1.0, 1.6]]))


def _dummy_custom_advantage(inputs: AdvantageInputs, scale: float = 1.0) -> AdvantageOutputs:
    """A simple custom advantage for testing."""
    return AdvantageOutputs(advantages=inputs.rewards * scale)


def test_default_advantage_normalization_none_matches_upstream():
    """normalization='none' (default) leaves the upstream center-only behavior intact."""
    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 0.5, 0.8], [0.2, 0.9, 0.1]]),
        completion_lengths=torch.tensor([[10, 12, 8], [15, 11, 9]]),
    )
    out_default = default_advantage_fn(inputs)
    out_none = default_advantage_fn(inputs, normalization="none")
    assert torch.allclose(out_default.advantages, out_none.advantages)


def test_default_advantage_normalization_group_divides_by_per_group_std():
    """normalization='group' divides by per-group std (classic GRPO)."""
    rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]])
    inputs = AdvantageInputs(
        rewards=rewards, completion_lengths=torch.zeros_like(rewards, dtype=torch.long)
    )
    out = default_advantage_fn(inputs, normalization="group")

    # First group has nonzero std; verify per-group division matches the explicit formula.
    expected_g0 = (rewards[0] - rewards[0].mean()) / (rewards[0].std(unbiased=False) + _NORM_EPS)
    assert torch.allclose(out.advantages[0], expected_g0, atol=1e-6)
    # Second group is degenerate (all 0.5 → centered all-zero, std 0); epsilon keeps it bounded.
    assert torch.allclose(out.advantages[1], torch.zeros(4), atol=1e-6)


def test_default_advantage_normalization_batch_emits_baseline_only():
    """`normalization='batch'` is intentionally a no-op inside `default_advantage_fn`
    — the batch-std step is deferred to `apply_batch_advantage_normalization`
    after rollout filtering (snowflake_poc_critique.md §1; ScaleRL §3.4
    requires the std to reflect surviving rollouts only). The function returns
    baseline-subtracted advantages, identical to `normalization='none'` for the
    pre-filter call.
    """
    rewards = torch.tensor([[1.0, 0.5, 0.8], [0.2, 0.9, 0.1]])
    inputs = AdvantageInputs(
        rewards=rewards, completion_lengths=torch.zeros_like(rewards, dtype=torch.long)
    )
    out_batch = default_advantage_fn(inputs, normalization="batch")
    out_none = default_advantage_fn(inputs, normalization="none")
    assert torch.allclose(out_batch.advantages, out_none.advantages, atol=1e-12)


def test_apply_batch_advantage_normalization_uses_post_filter_std():
    """Batch-std must be computed over UNFILTERED rollouts only. Filtered
    rollouts retain their pre-norm advantage (irrelevant; they don't train).
    """
    rollouts = [
        {"advantage": 1.0, "is_filtered": False},
        {"advantage": -1.0, "is_filtered": False},
        {"advantage": 100.0, "is_filtered": True},   # would dominate pre-filter std
        {"advantage": -100.0, "is_filtered": True},
    ]
    config = DefaultAdvantageConfig(normalization="batch")
    apply_batch_advantage_normalization(rollouts, config)
    surviving = torch.tensor([1.0, -1.0])
    expected_std = surviving.std(unbiased=False).item()
    assert rollouts[0]["advantage"] == pytest.approx(1.0 / (expected_std + _NORM_EPS))
    assert rollouts[1]["advantage"] == pytest.approx(-1.0 / (expected_std + _NORM_EPS))
    # Filtered rollouts unchanged.
    assert rollouts[2]["advantage"] == 100.0
    assert rollouts[3]["advantage"] == -100.0


def test_apply_batch_advantage_normalization_no_op_for_non_batch_modes():
    rollouts = [
        {"advantage": 1.0, "is_filtered": False},
        {"advantage": -1.0, "is_filtered": False},
    ]
    apply_batch_advantage_normalization(rollouts, DefaultAdvantageConfig(normalization="none"))
    assert rollouts[0]["advantage"] == 1.0
    apply_batch_advantage_normalization(rollouts, DefaultAdvantageConfig(normalization="group"))
    assert rollouts[0]["advantage"] == 1.0
    apply_batch_advantage_normalization(rollouts, None)
    assert rollouts[0]["advantage"] == 1.0


def test_apply_batch_advantage_normalization_handles_few_surviving():
    """Fewer than 2 surviving rollouts → std undefined; mark as filtered AND
    zero advantage. Filters key is a dict (matches `apply_filters` schema)."""
    rollouts = [
        {"advantage": 1.0, "is_filtered": False, "filters": {"gibberish": False, "zero_advantage": False}},
        {"advantage": -1.0, "is_filtered": True, "filters": {"gibberish": False, "zero_advantage": True}},
    ]
    apply_batch_advantage_normalization(rollouts, DefaultAdvantageConfig(normalization="batch"))
    assert rollouts[0]["advantage"] == 0.0
    assert rollouts[0]["is_filtered"] is True
    # CRITICAL: filters must remain a DICT (apply_filters sets it as one;
    # downstream pandas DataFrame in orchestrator.py reads dict keys as columns).
    assert isinstance(rollouts[0]["filters"], dict)
    assert rollouts[0]["filters"]["batch_norm_too_few_survivors"] is True
    assert rollouts[0]["filters"]["gibberish"] is False  # pre-existing keys preserved
    assert rollouts[1]["advantage"] == -1.0  # already-filtered rollouts untouched


def test_apply_batch_advantage_normalization_zeros_constant_advantages():
    """All surviving advantages identical (std == 0) → mark filtered and zero
    out so the orchestrator retries instead of running 1/eps explosions."""
    rollouts = [
        {"advantage": 0.5, "is_filtered": False, "filters": {"zero_advantage": False}},
        {"advantage": 0.5, "is_filtered": False, "filters": {"zero_advantage": False}},
        {"advantage": 0.5, "is_filtered": False, "filters": {"zero_advantage": False}},
    ]
    apply_batch_advantage_normalization(rollouts, DefaultAdvantageConfig(normalization="batch"))
    for r in rollouts:
        assert r["advantage"] == 0.0
        assert r["is_filtered"] is True
        assert isinstance(r["filters"], dict)
        assert r["filters"]["batch_norm_zero_std"] is True


def test_apply_batch_advantage_normalization_no_filters_key_falls_back_safely():
    """Direct callers (unit tests) may not pre-run `apply_filters`. The
    function must still mark the rollout filtered without crashing."""
    rollouts = [{"advantage": 1.0, "is_filtered": False}]
    apply_batch_advantage_normalization(rollouts, DefaultAdvantageConfig(normalization="batch"))
    assert rollouts[0]["is_filtered"] is True
    assert isinstance(rollouts[0]["filters"], dict)
    assert rollouts[0]["filters"]["batch_norm_too_few_survivors"] is True


def test_apply_batch_advantage_normalization_raises_on_nonfinite():
    rollouts = [
        {"advantage": 1.0, "is_filtered": False},
        {"advantage": float("nan"), "is_filtered": False},
    ]
    with pytest.raises(ValueError, match="non-finite advantages"):
        apply_batch_advantage_normalization(rollouts, DefaultAdvantageConfig(normalization="batch"))


def test_apply_batch_advantage_normalization_requires_is_filtered():
    """`is_filtered` is set by `apply_filters` on every rollout — missing key
    is a contract violation, not a default-to-False."""
    rollouts = [{"advantage": 1.0}, {"advantage": -1.0}]
    with pytest.raises(KeyError, match="is_filtered"):
        apply_batch_advantage_normalization(rollouts, DefaultAdvantageConfig(normalization="batch"))


def test_default_advantage_config_normalization_default_is_none():
    """Config-level default for `normalization` is `'none'` — preserves upstream behavior."""
    config = DefaultAdvantageConfig()
    assert config.normalization == "none"
