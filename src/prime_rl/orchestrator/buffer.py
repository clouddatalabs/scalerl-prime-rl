from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast

import verifiers as vf
from verifiers.utils.save_utils import make_serializable

from prime_rl.configs.orchestrator import BufferConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import format_num, mean, mean_normalize

if TYPE_CHECKING:
    from prime_rl.orchestrator.envs import TrainEnv, TrainEnvs


POOLS = ["easy", "normal", "hard"]


class _EnvBuffer:
    """Manages examples and difficulty pools for a single env."""

    def __init__(self, env: TrainEnv, config: BufferConfig):
        self.env_name = env.name
        self.config = config

        dataset = env.get_dataset(seed=config.seed)
        if "example_id" not in dataset.column_names:
            dataset = dataset.map(lambda ex, idx: {**ex, "example_id": idx}, with_indices=True)

        assert len(dataset) > 0, f"Dataset for {env.name} must contain at least one example."
        assert "example_id" in dataset.column_names, f"Dataset for {env.name} must contain an `example_id` column."
        assert "prompt" in dataset.column_names, f"Dataset for {env.name} must contain a `prompt` column."

        self.examples: dict[int, dict] = {}
        for example in map(partial(cast, dict), dataset):
            example["env_name"] = env.name
            self.examples[example["example_id"]] = example

        self.easy_examples: list[dict] = []
        self.hard_examples: list[dict] = []

        # ScaleRL §3.6 No-Positive-Resampling. `excluded_examples` holds prompts whose
        # running pass-rate has crossed the threshold; they never come back. `pass_rate_stats`
        # tracks the Welford-cumulative pass rate per example hash so resumes restore state.
        self.excluded_examples: dict[int, dict] = {}
        self.pass_rate_stats: dict[str, dict[str, float]] = {}

        self.reset_step_metrics()

    @property
    def num_normal(self) -> int:
        return len(self.examples)

    @property
    def num_total(self) -> int:
        return self.num_normal + len(self.easy_examples) + len(self.hard_examples) + len(self.excluded_examples)

    def sample_example(self) -> dict:
        key = random.choice(tuple(self.examples))
        return self.examples[key]

    def get_example_hash(self, example: dict) -> str:
        hash_keys = [key for key in self.config.hash_keys if key in example]
        assert hash_keys, "No hashable keys found in example."
        return hashlib.sha256(json.dumps([example[key] for key in hash_keys]).encode()).hexdigest()

    def update_pools(self, example_id: int, avg_reward: float) -> str:
        """Assign example to pool based on reward. Returns pool name."""
        if self.config.easy_threshold is not None and avg_reward >= self.config.easy_threshold:
            pool = "easy"
        elif self.config.hard_threshold is not None and avg_reward <= self.config.hard_threshold:
            pool = "hard"
        else:
            pool = "normal"

        if pool != "normal" and example_id in self.examples:
            example = self.examples.pop(example_id)
            target = self.easy_examples if pool == "easy" else self.hard_examples
            target.append(example)

        self.num_examples_per_step[pool] += 1
        return pool

    def reset_step_metrics(self) -> None:
        zero = lambda: {p: 0 for p in POOLS}
        self.num_examples_per_step = zero()
        self.num_rollouts_per_step = zero()
        self.num_excluded_per_step = 0

    def update_pass_rate(self, example_id: int, avg_reward: float) -> bool:
        """Update Welford-cumulative pass rate; permanently exclude on threshold cross.

        Returns True iff this call moved the example into `excluded_examples`. The current
        rollout group's training payload still flows downstream — exclusion takes effect
        for *future* sampling only.

        Pass-rate semantics: ScaleRL §3.6 says "history of pass rates" without pinning a
        window. We use Welford-cumulative mean over the prompt's full lifetime — easy
        prompts that started hard take a long time to cross the threshold under this
        choice, vs an EMA which would cross faster. Document if you change it.
        """
        if not self.config.no_positive_resampling or self.config.no_positive_resampling_threshold is None:
            return False
        if example_id not in self.examples:
            # Already promoted to easy/hard pool or already excluded — don't double-count.
            return False

        h = self.get_example_hash(self.examples[example_id])
        stats = self.pass_rate_stats.setdefault(h, {"num_groups": 0.0, "pass_rate": 0.0})
        n = stats["num_groups"] + 1
        p = stats["pass_rate"] + (avg_reward - stats["pass_rate"]) / n
        stats["num_groups"] = n
        stats["pass_rate"] = p

        if p >= self.config.no_positive_resampling_threshold:
            self.excluded_examples[example_id] = self.examples.pop(example_id)
            self.num_excluded_per_step += 1
            return True
        return False

    def get_metrics(self) -> dict[str, float]:
        metrics = {}
        num_examples = sum(self.num_examples_per_step.values())
        num_rollouts = sum(self.num_rollouts_per_step.values())

        for pool in ["easy", "hard"]:
            if num_examples:
                metrics[f"evicted_examples/{self.env_name}/{pool}"] = self.num_examples_per_step[pool] / num_examples
            if num_rollouts:
                metrics[f"filtered_rollouts/{self.env_name}/{pool}"] = self.num_rollouts_per_step[pool] / num_rollouts

        pool_counts = [len(self.easy_examples), self.num_normal, len(self.hard_examples)]
        pool_ratios = mean_normalize(pool_counts)
        for pool, ratio in zip(POOLS, pool_ratios):
            metrics[f"pool/{self.env_name}/{pool}"] = ratio

        self.reset_step_metrics()
        return metrics


class Buffer:
    """Manages multiple Buffers with env-ratio-aware sampling."""

    def __init__(self, envs: TrainEnvs, config: BufferConfig):
        self.config = config
        self.logger = get_logger()

        if config.seed is not None:
            random.seed(config.seed)

        self.env_buffers: dict[str, _EnvBuffer] = {}
        for env in envs:
            self.env_buffers[env.name] = _EnvBuffer(env, config)
        self.env_names = envs.names

        total = sum(eb.num_total for eb in self.env_buffers.values())
        self.logger.debug(
            f"Initialized buffer with {format_num(total, precision=0)} example(s) "
            f"in {len(self.env_names)} environment(s)"
        )

        env_ratios = [env.config.ratio for env in envs]
        if any(r is not None for r in env_ratios):
            env_ratio = mean_normalize(env_ratios)
            self.env_probs = dict(zip(self.env_names, env_ratio))
            self.logger.debug(
                f"Sampling buffer according to provided environment ratios "
                f"({', '.join(f'{k}={v:.2f}' for k, v in self.env_probs.items())})"
            )
        else:
            env_counts = [self.env_buffers[name].num_normal for name in self.env_names]
            env_ratio = mean_normalize(env_counts)
            self.env_probs = dict(zip(self.env_names, env_ratio))
            self.logger.debug(
                f"Sampling buffer according to natural environment distribution "
                f"({', '.join(f'{k}={v:.2f}' for k, v in self.env_probs.items())})"
            )

        self.rollout_buffer: list[vf.RolloutOutput] = []

    def sample_examples(self, n: int) -> list[dict]:
        """Samples n examples across envs, respecting env ratios."""
        non_empty = [name for name, eb in self.env_buffers.items() if eb.examples]
        if not non_empty:
            raise ValueError("No environments left with examples.")

        weights = [self.env_probs[name] for name in non_empty]
        return [self.env_buffers[name].sample_example() for name in random.choices(non_empty, weights=weights, k=n)]

    def update(self, rollouts: list[vf.RolloutOutput]):
        """Updates buffer state with completed rollouts."""
        rollouts_by_example = defaultdict(list)
        for rollout in rollouts:
            rollouts_by_example[(rollout["env_name"], rollout["example_id"])].append(rollout)

        for (env_name, example_id), example_rollouts in rollouts_by_example.items():
            eb = self.env_buffers[env_name]
            avg_reward = mean([r["reward"] for r in example_rollouts])
            eb.update_pools(example_id, avg_reward)
            # NPR runs after pool eviction so it's a no-op for examples already in easy/hard
            # (and for anything previously excluded). The current group still flows to training;
            # the exclusion only takes effect for future sampling.
            eb.update_pass_rate(example_id, avg_reward)

            if self.config.online_difficulty_filtering:
                if avg_reward == 0.0:
                    eb.num_rollouts_per_step["hard"] += len(example_rollouts)
                    continue
                elif avg_reward == 1.0:
                    eb.num_rollouts_per_step["easy"] += len(example_rollouts)
                    continue

            eb.num_rollouts_per_step["normal"] += len(example_rollouts)
            self.rollout_buffer.extend(example_rollouts)

    def sample_rollouts(self, n: int) -> list[vf.RolloutOutput]:
        """Samples the latest n rollouts from the buffer."""
        n = min(n, len(self.rollout_buffer))
        sampled = self.rollout_buffer[-n:]
        self.rollout_buffer = self.rollout_buffer[:-n]
        return sampled

    def save(self, path: Path) -> None:
        """Saves pool assignments and rollout buffer."""
        path.mkdir(parents=True, exist_ok=True)

        def write_jsonl(lst: list, filepath: Path) -> None:
            with open(filepath, "w") as f:
                for item in lst:
                    f.write(json.dumps(item, default=make_serializable) + "\n")

        all_easy = [ex for eb in self.env_buffers.values() for ex in eb.easy_examples]
        all_hard = [ex for eb in self.env_buffers.values() for ex in eb.hard_examples]
        all_excluded = [ex for eb in self.env_buffers.values() for ex in eb.excluded_examples.values()]
        all_pass_rate_stats = [
            {"example_hash": h, **stats}
            for eb in self.env_buffers.values()
            for h, stats in eb.pass_rate_stats.items()
        ]
        write_jsonl(all_easy, path / "easy_examples.jsonl")
        write_jsonl(all_hard, path / "hard_examples.jsonl")
        write_jsonl(all_excluded, path / "excluded_examples.jsonl")
        write_jsonl(all_pass_rate_stats, path / "pass_rate_stats.jsonl")
        write_jsonl(self.rollout_buffer, path / "rollout_buffer.jsonl")

    def load(self, path: Path) -> None:
        """Loads pool assignments and rollouts from checkpoint."""

        def read_jsonl(filepath: Path, missing_ok: bool = False) -> list[dict]:
            if missing_ok and not filepath.exists():
                return []
            with open(filepath, "r") as f:
                return [json.loads(line) for line in f]

        saved_easy = read_jsonl(path / "easy_examples.jsonl")
        saved_hard = read_jsonl(path / "hard_examples.jsonl")
        # Excluded examples and pass-rate stats are missing-ok: pre-NPR checkpoints
        # don't have these files and resuming should not crash on them.
        saved_excluded = read_jsonl(path / "excluded_examples.jsonl", missing_ok=True)
        saved_pass_rate_stats = read_jsonl(path / "pass_rate_stats.jsonl", missing_ok=True)
        saved_rollouts = cast(list[vf.RolloutOutput], read_jsonl(path / "rollout_buffer.jsonl"))

        if (
            not any(saved_easy)
            and not any(saved_hard)
            and not any(saved_excluded)
            and not any(saved_pass_rate_stats)
            and not any(saved_rollouts)
        ):
            self.logger.debug("No easy/hard/excluded examples, pass-rate stats, or rollouts found in checkpoint")
            return

        # Build hash lookup across all env buffers: env -> (hash -> example_id)
        hash_lookup: dict[str, dict[str, int]] = defaultdict(dict)
        all_hashes: set[str] = set()
        for env_name, eb in self.env_buffers.items():
            for example_id, example in eb.examples.items():
                h = eb.get_example_hash(example)
                if h in all_hashes:
                    self.logger.warning(
                        f"Duplicate example hash found based on hash_keys={self.config.hash_keys}. "
                        "Overwriting with latest example. This may cause unexpected behavior when resuming the buffer."
                    )
                hash_lookup[env_name][h] = example_id
                all_hashes.add(h)

        def move_saved_pool(saved_examples: list[dict], pool_name: str) -> int:
            num_moved = 0
            for example in saved_examples:
                # Use any env buffer to compute hash (hash_keys are config-level)
                first_eb = next(iter(self.env_buffers.values()))
                h = first_eb.get_example_hash(example)
                for env_name, env_hashes in hash_lookup.items():
                    if h in env_hashes:
                        example_id = env_hashes[h]
                        eb = self.env_buffers[env_name]
                        matched = eb.examples.pop(example_id, None)
                        if matched is not None:
                            if pool_name == "easy":
                                eb.easy_examples.append(matched)
                            elif pool_name == "hard":
                                eb.hard_examples.append(matched)
                            else:
                                # NPR exclusion: keyed by example_id in a dict, not a list.
                                eb.excluded_examples[example_id] = matched
                            num_moved += 1
                            break
            return num_moved

        if any(saved_easy):
            num_moved = move_saved_pool(saved_easy, "easy")
            self.logger.debug(f"Loaded {num_moved}/{len(saved_easy)} example(s) to easy pool from checkpoint.")
            if num_moved != len(saved_easy):
                self.logger.warning(
                    f"Could not move {len(saved_easy) - num_moved} example(s) from checkpoint to easy pool. "
                    "This usually means you resumed with an env mix that does not contain all previous examples."
                )

        if any(saved_hard):
            num_moved = move_saved_pool(saved_hard, "hard")
            self.logger.debug(f"Moved {num_moved}/{len(saved_hard)} example(s) to hard pool from checkpoint.")
            if num_moved != len(saved_hard):
                self.logger.warning(
                    f"Could not move {len(saved_hard) - num_moved} example(s) from checkpoint to hard pool. "
                    "This usually means you resumed with an env mix that does not contain all previous examples."
                )

        if any(saved_excluded):
            num_moved = move_saved_pool(saved_excluded, "excluded")
            self.logger.debug(
                f"Restored {num_moved}/{len(saved_excluded)} no-positive-resampling exclusion(s) from checkpoint."
            )
            if num_moved != len(saved_excluded):
                self.logger.warning(
                    f"Could not restore {len(saved_excluded) - num_moved} no-positive-resampling exclusion(s); "
                    "the resume dataset does not contain those examples."
                )

        if any(saved_pass_rate_stats):
            restored = 0
            # Build hash → env_name lookup so we attach stats to the right buffer.
            hash_to_env = {h: env for env, env_hashes in hash_lookup.items() for h in env_hashes}
            for entry in saved_pass_rate_stats:
                h = entry.get("example_hash")
                if h is None:
                    continue
                env_name = hash_to_env.get(h)
                if env_name is None:
                    continue
                self.env_buffers[env_name].pass_rate_stats[h] = {
                    "num_groups": float(entry.get("num_groups", 0.0)),
                    "pass_rate": float(entry.get("pass_rate", 0.0)),
                }
                restored += 1
            self.logger.debug(
                f"Loaded {restored}/{len(saved_pass_rate_stats)} no-positive-resampling pass-rate stat(s) from checkpoint."
            )

        if any(saved_rollouts):
            valid = [r for r in saved_rollouts if r.get("env_name") in self.env_names]
            self.rollout_buffer.extend(valid)
            self.logger.debug(f"Loaded {len(valid)} rollout(s) from checkpoint.")

        def convert_to_normal(eb: _EnvBuffer, pool: list[dict], fraction: float) -> int:
            if fraction <= 0.0 or not pool:
                return 0
            num_to_move = round(len(pool) * fraction)
            if num_to_move <= 0:
                return 0
            for _ in range(num_to_move):
                example = random.choice(pool)
                pool.remove(example)
                eb.examples[example["example_id"]] = example
            return num_to_move

        for eb in self.env_buffers.values():
            n_easy = len(eb.easy_examples)
            moved = convert_to_normal(eb, eb.easy_examples, self.config.easy_fraction)
            self.logger.debug(f"Converted {moved}/{n_easy} example(s) back to normal from easy pool ({eb.env_name}).")
            n_hard = len(eb.hard_examples)
            moved = convert_to_normal(eb, eb.hard_examples, self.config.hard_fraction)
            self.logger.debug(f"Converted {moved}/{n_hard} example(s) back to normal from hard pool ({eb.env_name}).")

    def get_metrics(self) -> dict[str, float]:
        metrics = {}

        # Aggregate cross-env totals
        total_examples_per_pool = {p: 0 for p in POOLS}
        total_rollouts_per_pool = {p: 0 for p in POOLS}
        for eb in self.env_buffers.values():
            for p in POOLS:
                total_examples_per_pool[p] += eb.num_examples_per_step[p]
                total_rollouts_per_pool[p] += eb.num_rollouts_per_step[p]

        total_examples = sum(total_examples_per_pool.values())
        total_rollouts = sum(total_rollouts_per_pool.values())

        for pool in ["easy", "hard"]:
            if total_examples:
                metrics[f"evicted_examples/{pool}"] = total_examples_per_pool[pool] / total_examples
            if total_rollouts:
                metrics[f"filtered_rollouts/{pool}"] = total_rollouts_per_pool[pool] / total_rollouts

        total_normal = sum(eb.num_normal for eb in self.env_buffers.values())
        total_easy = sum(len(eb.easy_examples) for eb in self.env_buffers.values())
        total_hard = sum(len(eb.hard_examples) for eb in self.env_buffers.values())
        total_excluded = sum(len(eb.excluded_examples) for eb in self.env_buffers.values())
        pool_ratios = mean_normalize([total_easy, total_normal, total_hard])
        for pool, ratio in zip(POOLS, pool_ratios):
            metrics[f"pool/{pool}"] = ratio

        if self.config.no_positive_resampling:
            num_excluded_per_step = sum(eb.num_excluded_per_step for eb in self.env_buffers.values())
            if total_examples:
                metrics["excluded_examples/no_positive_resampling"] = num_excluded_per_step / total_examples
            metrics["excluded_examples/no_positive_resampling/cumulative"] = total_excluded

        # Per-env metrics
        for eb in self.env_buffers.values():
            metrics.update(eb.get_metrics())

        return metrics
