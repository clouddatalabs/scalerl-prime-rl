import random
from unittest.mock import MagicMock

import pytest
import verifiers as vf
from datasets import Dataset

from prime_rl.configs.orchestrator import BufferConfig, EnvConfig
from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.envs import Envs, TrainEnv


def make_env(name: str, vf_env: vf.Environment, **config_kwargs) -> TrainEnv:
    """Create a TrainEnv without calling vf.load_environment."""
    config = EnvConfig(id=name, name=name, **config_kwargs)
    env = TrainEnv.__new__(TrainEnv)
    env.config = config
    env._env = vf_env
    env._env_client = None
    env._env_server_process = None
    env.sampling_args = {}
    return env


def make_envs(env_dict: dict[str, TrainEnv]) -> Envs:
    """Create an Envs container from a dict of Env instances."""
    envs = Envs.__new__(Envs)
    envs._envs = env_dict
    return envs


@pytest.fixture(autouse=True)
def set_seed():
    random.seed(42)


@pytest.fixture
def mock_openai_client():
    """Return a mocked OpenAI client."""
    return MagicMock()


@pytest.fixture
def dummy_dataset() -> Dataset:
    """Return a dummy dataset with 5 examples."""
    return Dataset.from_dict(
        {
            "question": ["q0", "q1", "q2", "q3", "q4"],
            "answer": ["a0", "a1", "a2", "a3", "a4"],
        }
    )


@pytest.fixture
def dummy_envs(mock_openai_client, dummy_dataset) -> Envs:
    """Return an Envs with two dummy envs."""
    env_a = vf.SingleTurnEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=dummy_dataset,
        rubric=vf.Rubric(),
    )
    env_b = vf.SingleTurnEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=dummy_dataset,
        rubric=vf.Rubric(),
    )
    return make_envs(
        {
            "env_a": make_env("env_a", env_a),
            "env_b": make_env("env_b", env_b),
        }
    )


@pytest.fixture
def make_rollouts():
    def _make_rollouts(
        buffer: Buffer, env_name: str, indices: list[int], rewards: list[float]
    ) -> list[vf.RolloutOutput]:
        all_rollouts = []
        eb = buffer.env_buffers[env_name]
        examples = list(eb.examples.values())
        for idx, reward in zip(indices, rewards):
            example = examples[idx]
            rollouts = [
                vf.RolloutOutput(
                    example_id=example["example_id"],
                    task=example["env_name"],
                    prompt=example["prompt"],
                    prompt_ids=[0],
                    prompt_mask=[1],
                    completion_ids=[1],
                    completion_mask=[1],
                    completion_logprobs=[0.0],
                    is_truncated=False,
                    reward=reward,
                    advantage=1.0,
                    metrics={},
                )
            ] * 2
            for r in rollouts:
                r["env_name"] = env_name
            all_rollouts.extend(rollouts)
        return all_rollouts

    return _make_rollouts


def get_normal_count(buffer: Buffer) -> int:
    return sum(eb.num_normal for eb in buffer.env_buffers.values())


def test_buffer_init_and_sample(dummy_envs):
    buffer = Buffer(dummy_envs, BufferConfig())
    assert buffer.env_buffers["env_a"].num_normal == 5
    assert buffer.env_buffers["env_b"].num_normal == 5
    samples = buffer.sample_examples(2)
    assert len(samples) == 2


def test_buffer_problem_pool_assignment(dummy_envs, make_rollouts):
    """Problems are moved to easy/hard pools based on reward thresholds."""
    buffer = Buffer(dummy_envs, BufferConfig(easy_threshold=1.0, hard_threshold=0.0))
    buffer.update(make_rollouts(buffer, "env_a", list(range(5)), rewards=[1.0, 1.0, 0.5, 0.5, 0.0]))

    assert len(buffer.env_buffers["env_a"].easy_examples) == 2
    assert len(buffer.env_buffers["env_a"].hard_examples) == 1
    # 2 normal from env_a + 5 from env_b = 7
    assert get_normal_count(buffer) == 7


def test_buffer_online_difficulty_filtering(dummy_envs, make_rollouts):
    """With online_difficulty_filtering=True, only partial reward rollouts are kept."""
    buffer = Buffer(
        dummy_envs,
        BufferConfig(online_difficulty_filtering=True),
    )
    buffer.update(make_rollouts(buffer, "env_a", list(range(5)), rewards=[1.0, 0.5, 0.0, 0.5, 0.5]))

    # Only 3 problems with reward 0.5 -> 6 rollouts kept
    assert len(buffer.rollout_buffer) == 6


def test_buffer_no_filtering_by_default(dummy_envs, make_rollouts):
    """With online_difficulty_filtering=False (default), all rollouts are kept."""
    buffer = Buffer(dummy_envs, BufferConfig())
    buffer.update(make_rollouts(buffer, "env_a", list(range(5)), rewards=[1.0, 0.5, 0.0, 0.5, 0.5]))

    # All 5 problems -> 10 rollouts kept
    assert len(buffer.rollout_buffer) == 10


def test_buffer_save_load_with_conversion(dummy_envs, make_rollouts, tmp_path):
    """Easy/hard problems are partially converted to normal on load."""
    buffer = Buffer(dummy_envs, BufferConfig(easy_threshold=1.0, hard_threshold=0.0))
    buffer.update(make_rollouts(buffer, "env_a", list(range(5)), rewards=[1.0, 1.0, 0.5, 0.5, 0.0]))
    buffer.save(tmp_path / "buffer")

    new_buffer = Buffer(dummy_envs, BufferConfig(easy_fraction=0.5, hash_keys=["prompt", "env_name"]))
    new_buffer.load(tmp_path / "buffer")

    # 1 of 2 easy problems converted to normal
    assert len(new_buffer.env_buffers["env_a"].easy_examples) == 1
    # 2 were normal + 5 from env_b + 1 converted from easy = 8
    assert get_normal_count(new_buffer) == 8


def test_buffer_env_ratios(mock_openai_client, dummy_dataset):
    env_a = vf.SingleTurnEnv(client=mock_openai_client, model="test-model", dataset=dummy_dataset, rubric=vf.Rubric())
    env_b = vf.SingleTurnEnv(client=mock_openai_client, model="test-model", dataset=dummy_dataset, rubric=vf.Rubric())
    envs = make_envs(
        {
            "env_a": make_env("env_a", env_a, ratio=0.8),
            "env_b": make_env("env_b", env_b, ratio=0.2),
        }
    )

    buffer = Buffer(envs, BufferConfig())
    assert buffer.env_buffers["env_a"].num_normal == 5
    assert buffer.env_buffers["env_b"].num_normal == 5

    samples = buffer.sample_examples(100)
    env_a_count = sum(1 for p in samples if p["env_name"] == "env_a")
    assert 60 <= env_a_count <= 95


def test_buffer_env_ratios_validation():
    """Validates that env ratios must be positive and all-or-nothing."""
    from pydantic import ValidationError

    from prime_rl.configs.orchestrator import TrainConfig, TrainEnvConfig

    with pytest.raises(ValidationError):
        EnvConfig(id="env_a", ratio=-0.3)

    with pytest.raises(ValidationError, match="mix of set and unset"):
        TrainConfig(env=[TrainEnvConfig(id="a", ratio=0.5), TrainEnvConfig(id="b")])


def test_buffer_no_cross_env_pool_assignment(mock_openai_client, tmp_path):
    """Pool assignments don't transfer if example_id exists but env changed."""
    original_dataset = Dataset.from_dict({"question": ["q0"], "answer": ["a0"]})
    original_env = vf.SingleTurnEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=original_dataset,
        rubric=vf.Rubric(),
    )
    original_env_set = make_envs({"env_a": make_env("env_a", original_env)})

    buffer = Buffer(original_env_set, BufferConfig(easy_threshold=1.0))
    eb = buffer.env_buffers["env_a"]
    example_id = list(eb.examples.keys())[0]
    example = eb.examples.pop(example_id)
    eb.easy_examples.append(example)
    buffer.save(tmp_path / "buffer")

    new_dataset = Dataset.from_dict({"question": ["different_q"], "answer": ["different_a"]})
    new_env = vf.SingleTurnEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=new_dataset,
        rubric=vf.Rubric(),
    )
    new_env_set = make_envs({"env_b": make_env("env_b", new_env)})

    new_buffer = Buffer(new_env_set, BufferConfig())
    new_buffer.load(tmp_path / "buffer")

    assert len(new_buffer.env_buffers["env_b"].easy_examples) == 0
    assert new_buffer.env_buffers["env_b"].num_normal == 1


def test_buffer_no_positive_resampling_excludes_after_threshold(dummy_envs, make_rollouts):
    """A prompt whose running pass-rate crosses the threshold gets excluded permanently."""
    buffer = Buffer(
        dummy_envs,
        BufferConfig(no_positive_resampling=True, no_positive_resampling_threshold=0.9),
    )
    eb = buffer.env_buffers["env_a"]
    initial_normal = eb.num_normal

    # Single example crosses threshold on the first group: avg_reward=1.0 ≥ 0.9.
    buffer.update(make_rollouts(buffer, "env_a", [0], rewards=[1.0]))

    assert 0 not in eb.examples, "Example should be removed from the active pool."
    assert 0 in eb.excluded_examples, "Example should land in excluded_examples."
    assert eb.num_normal == initial_normal - 1
    assert eb.num_excluded_per_step == 1


def test_buffer_no_positive_resampling_min_groups_blocks_single_group_eviction(dummy_envs, make_rollouts):
    """`min_groups_for_exclusion` floor blocks single-group lucky-shot eviction.

    With Welford starting at n=0, a 14/16-correct first group on a small
    pool (Terminal-Bench: 18 train prompts, 14/16 ≈ 0.875 not even reached
    here, but 1.0 trivially crosses) would otherwise permanently evict the
    prompt before any confirming observation. With min_groups=4 the prompt
    must accumulate four groups all averaging ≥ 0.9 before exclusion.
    """
    buffer = Buffer(
        dummy_envs,
        BufferConfig(
            no_positive_resampling=True,
            no_positive_resampling_threshold=0.9,
            no_positive_resampling_min_groups=4,
        ),
    )
    eb = buffer.env_buffers["env_a"]
    initial_normal = eb.num_normal

    # First group at avg_reward=1.0 must NOT evict (n=1 < min_groups=4).
    buffer.update(make_rollouts(buffer, "env_a", [0], rewards=[1.0]))
    assert 0 in eb.examples, "Single high-reward group must not evict under min_groups=4."
    assert eb.num_normal == initial_normal

    # Three more groups at 1.0 (n becomes 4, mean still 1.0 ≥ 0.9): now evicts.
    for _ in range(3):
        buffer.update(make_rollouts(buffer, "env_a", [0], rewards=[1.0]))
    assert 0 not in eb.examples, "After 4 confirming groups, prompt must evict."
    assert 0 in eb.excluded_examples


def test_buffer_no_positive_resampling_keeps_below_threshold(dummy_envs, make_rollouts):
    """A prompt with avg_reward below threshold stays in the pool, stats accumulate."""
    buffer = Buffer(
        dummy_envs,
        BufferConfig(no_positive_resampling=True, no_positive_resampling_threshold=0.9),
    )
    eb = buffer.env_buffers["env_a"]
    initial_normal = eb.num_normal

    # avg_reward=0.5 stays below 0.9 — Welford-cumulative stays at 0.5 even after many groups.
    for _ in range(5):
        buffer.update(make_rollouts(buffer, "env_a", [0], rewards=[0.5]))

    assert 0 in eb.examples, "Example below threshold should stay in the pool."
    assert eb.num_normal == initial_normal
    h = eb.get_example_hash(eb.examples[0])
    assert eb.pass_rate_stats[h]["pass_rate"] == pytest.approx(0.5, abs=1e-6)
    assert eb.pass_rate_stats[h]["num_groups"] == 5.0


def test_buffer_no_positive_resampling_welford_cumulative_mean(dummy_envs, make_rollouts):
    """Pass rate is Welford-cumulative — old groups are not forgotten."""
    buffer = Buffer(
        dummy_envs,
        BufferConfig(no_positive_resampling=True, no_positive_resampling_threshold=0.95),
    )
    eb = buffer.env_buffers["env_a"]

    # Hard early (5x at 0.0), easy late (1x at 1.0): mean = 1/6 ≈ 0.167. Far below 0.95 threshold.
    for _ in range(5):
        buffer.update(make_rollouts(buffer, "env_a", [0], rewards=[0.0]))
    buffer.update(make_rollouts(buffer, "env_a", [0], rewards=[1.0]))

    assert 0 in eb.examples, "Cumulative mean stays below threshold; example must remain."
    h = eb.get_example_hash(eb.examples[0])
    assert eb.pass_rate_stats[h]["pass_rate"] == pytest.approx(1 / 6, abs=1e-6)


def test_buffer_no_positive_resampling_disabled_by_default(dummy_envs, make_rollouts):
    """With NPR disabled (default), no exclusions happen even at avg_reward=1.0."""
    buffer = Buffer(dummy_envs, BufferConfig())
    eb = buffer.env_buffers["env_a"]

    buffer.update(make_rollouts(buffer, "env_a", [0], rewards=[1.0]))

    assert 0 in eb.examples
    assert eb.excluded_examples == {}
    assert eb.pass_rate_stats == {}


def test_buffer_no_positive_resampling_save_load_round_trip(dummy_envs, make_rollouts, tmp_path):
    """Save+load preserves excluded_examples and pass_rate_stats.

    Note on `make_rollouts(buffer, env_name, indices, ...)`: `indices` is a
    POSITIONAL index into `list(eb.examples.values())` at call time, not an
    example_id. After update #1 excludes example 0, `eb.examples.values()`
    starts at example_id=1 — so `[0]` in the second call targets example_id=1.
    """
    buffer = Buffer(
        dummy_envs,
        BufferConfig(no_positive_resampling=True, no_positive_resampling_threshold=0.9),
    )

    # update #1: positional 0 is example_id=0 (no eviction yet). Reward 1.0 ≥ 0.9
    # → NPR moves it to excluded. Save the example dict so we can verify hash
    # round-trip after load (saved hash should match new_buffer's hash for ex0).
    excluded_example_dict = dict(buffer.env_buffers["env_a"].examples[0])
    buffer.update(make_rollouts(buffer, "env_a", [0], rewards=[1.0]))
    assert 0 in buffer.env_buffers["env_a"].excluded_examples, (
        "Sanity: the first update should have excluded example 0 (reward 1.0 ≥ threshold 0.9)."
    )

    # update #2: positional 0 NOW resolves to example_id=1, since 0 was just
    # evicted. Reward 0.5 stays below threshold → NPR records stats only.
    below_threshold_example_dict = dict(buffer.env_buffers["env_a"].examples[1])
    buffer.update(make_rollouts(buffer, "env_a", [0], rewards=[0.5]))
    assert 1 in buffer.env_buffers["env_a"].examples, (
        "Sanity: example 1 should still be in the active pool after a sub-threshold update."
    )

    buffer.save(tmp_path / "buffer")

    new_buffer = Buffer(
        dummy_envs,
        BufferConfig(no_positive_resampling=True, no_positive_resampling_threshold=0.9),
    )
    new_buffer.load(tmp_path / "buffer")
    eb = new_buffer.env_buffers["env_a"]

    assert 0 in eb.excluded_examples, "Excluded example must round-trip."
    assert 0 not in eb.examples, "Excluded example must NOT be back in the active pool."
    assert len(eb.pass_rate_stats) == 2, f"Expected stats for both examples; got {eb.pass_rate_stats}"

    h_excluded = eb.get_example_hash(excluded_example_dict)
    h_below = eb.get_example_hash(below_threshold_example_dict)
    assert eb.pass_rate_stats[h_excluded]["pass_rate"] == pytest.approx(1.0, abs=1e-6)
    assert eb.pass_rate_stats[h_excluded]["num_groups"] == 1.0
    assert eb.pass_rate_stats[h_below]["pass_rate"] == pytest.approx(0.5, abs=1e-6)
    assert eb.pass_rate_stats[h_below]["num_groups"] == 1.0
    # And the live example_buffer hash matches the saved hash for the still-active prompt.
    assert eb.get_example_hash(eb.examples[1]) == h_below


def test_buffer_no_positive_resampling_save_load_hashes_match_new_dataset(dummy_envs, make_rollouts, tmp_path):
    """Hashes stored in pass_rate_stats survive a fresh Buffer over the same dataset.

    Regression on the property the round-trip relies on: hashing
    `eb.examples[i]` from a freshly-constructed Buffer against the same envs
    yields the same bytes as the save-time hash. If `Buffer.__init__` ever
    mutates the example dict in a way that affects `hash_keys`, this fails
    and the round-trip silently loses stats.
    """
    buffer = Buffer(
        dummy_envs,
        BufferConfig(no_positive_resampling=True, no_positive_resampling_threshold=0.9),
    )
    saved_hashes = {
        eid: buffer.env_buffers["env_a"].get_example_hash(buffer.env_buffers["env_a"].examples[eid])
        for eid in (0, 1, 2)
    }
    # Drive at least one stats write so the file is non-empty.
    buffer.update(make_rollouts(buffer, "env_a", [0], rewards=[0.5]))
    buffer.save(tmp_path / "buffer")

    new_buffer = Buffer(
        dummy_envs,
        BufferConfig(no_positive_resampling=True, no_positive_resampling_threshold=0.9),
    )
    new_buffer.load(tmp_path / "buffer")
    eb = new_buffer.env_buffers["env_a"]
    for eid, expected_hash in saved_hashes.items():
        assert eb.get_example_hash(eb.examples[eid]) == expected_hash


def test_buffer_no_positive_resampling_threshold_required():
    """no_positive_resampling=True without threshold raises at config validation."""
    with pytest.raises(ValueError, match="no_positive_resampling_threshold"):
        BufferConfig(no_positive_resampling=True)


def test_buffer_config_rejects_easy_threshold_shadowing_npr():
    """`easy_threshold <= no_positive_resampling_threshold` makes NPR unreachable
    because update_pools (easy/hard) runs BEFORE update_pass_rate, so any example
    that would cross NPR is evicted to the easy pool first. Reject at config load.
    """
    with pytest.raises(ValueError, match="shadows NPR"):
        BufferConfig(
            easy_threshold=0.5,
            no_positive_resampling=True,
            no_positive_resampling_threshold=0.9,
        )


def test_buffer_config_rejects_easy_threshold_equal_to_npr_threshold():
    """Boundary case for the shadow check: easy == npr_threshold also blocks NPR
    (anything ≥ npr also ≥ easy → easy pool absorbs first). Pin the equality
    boundary so a regression flipping `<=` to `<` doesn't slip through."""
    with pytest.raises(ValueError, match="shadows NPR"):
        BufferConfig(
            easy_threshold=0.9,
            no_positive_resampling=True,
            no_positive_resampling_threshold=0.9,
        )


def test_buffer_config_rejects_online_difficulty_filtering_with_npr():
    """online_difficulty_filtering and no_positive_resampling are policy-
    overlapping ScaleRL deviations. Reject the combination so the user
    picks ONE eviction mechanism (paper-faithful is NPR alone)."""
    with pytest.raises(ValueError, match="online_difficulty_filtering=True with"):
        BufferConfig(
            online_difficulty_filtering=True,
            no_positive_resampling=True,
            no_positive_resampling_threshold=0.9,
        )


def test_buffer_config_accepts_online_difficulty_filtering_alone():
    """OD-filtering on its own (no NPR) validates — it's a non-paper but
    not-shadowing-anything choice."""
    cfg = BufferConfig(online_difficulty_filtering=True)
    assert cfg.online_difficulty_filtering is True
    assert cfg.no_positive_resampling is False


def test_buffer_no_positive_resampling_skipped_for_easy_promoted(dummy_envs, make_rollouts):
    """Sanity for the `easy <= NPR` shadowing case: with the schema ordering
    above (easy_threshold > NPR_threshold), an example whose avg_reward exceeds
    easy_threshold takes the easy path and never gets an NPR stats update.
    """
    buffer = Buffer(
        dummy_envs,
        BufferConfig(
            easy_threshold=1.0,
            no_positive_resampling=True,
            no_positive_resampling_threshold=0.9,
        ),
    )
    eb = buffer.env_buffers["env_a"]
    # avg_reward=1.0 hits easy_threshold first — promoted to easy pool, NPR skipped.
    buffer.update(make_rollouts(buffer, "env_a", [0], rewards=[1.0]))
    assert 0 not in eb.examples
    assert 0 not in eb.excluded_examples
    assert eb.easy_examples and eb.easy_examples[0]["example_id"] == 0
    assert not eb.pass_rate_stats, "NPR stats must NOT update when update_pools already evicted."
