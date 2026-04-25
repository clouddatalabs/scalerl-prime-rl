"""Pin Progress save/load round-trip including next_group_id.

The `next_group_id` field exists specifically to prevent fresh-scheduler
group_ids from colliding with stamped rollouts in `rollout_buffer.jsonl` on
resume. A regression that drops the snapshot-on-save or restore-on-load
silently breaks `apply_prompt_average_sequence_weights`'s group-keyed
normalization (it would merge unrelated prompts into one prompt slot).
"""

import dataclasses

from prime_rl.orchestrator.ckpt import Progress


def test_progress_default_next_group_id_is_zero():
    """Fresh Progress starts at group_id=0 — matches Scheduler.__init__."""
    progress = Progress()
    assert progress.next_group_id == 0


def test_progress_serializes_next_group_id():
    """asdict round-trip preserves the field — `CheckpointManager.save_to_path`
    pickles the dataclass and `load_from_path` rehydrates with `setattr`. If
    the field were ever marked `init=False` or dropped from __dataclass_fields__,
    `asdict` would silently lose it on save and the restore would always read 0
    (= default), reintroducing the collision the field exists to prevent.
    """
    progress = Progress(step=42, total_tokens=1000, total_samples=50, total_problems=10, next_group_id=137)
    data = dataclasses.asdict(progress)
    assert data["next_group_id"] == 137

    # Simulate the load-side `setattr(progress, key, value)` loop.
    fresh = Progress()
    for key, value in data.items():
        setattr(fresh, key, value)
    assert fresh.next_group_id == 137


def test_progress_legacy_checkpoint_without_next_group_id():
    """A pickled Progress from a pre-field checkpoint reloads with default 0
    (so the orchestrator copies 0 → scheduler.next_group_id, matching the
    fresh-scheduler default; legacy rollouts on disk had no `group_id` so
    prompt-avg keying falls back to (env_name, example_id) anyway)."""
    legacy_data = dict(step=10, total_tokens=100, total_samples=5, total_problems=1)
    fresh = Progress()
    for key, value in legacy_data.items():
        setattr(fresh, key, value)
    assert fresh.next_group_id == 0


def test_skip_progress_with_buffer_load_still_restores_next_group_id(tmp_path):
    """skip_progress=True + skip_buffer=False (the default) must NOT reset
    next_group_id to 0 while keeping the buffer's stamped rollouts. That
    combination would silently re-issue colliding group_ids and merge
    unrelated prompts in apply_prompt_average_sequence_weights's keying.
    """
    import torch
    from unittest.mock import patch

    from prime_rl.configs.orchestrator import CheckpointConfig
    from prime_rl.orchestrator.ckpt import CheckpointManager

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    cfg = CheckpointConfig(skip_progress=True, skip_buffer=False)
    mgr = CheckpointManager(output_dir, cfg)

    # Create a fake checkpoint with next_group_id=137.
    ckpt_path = mgr.get_ckpt_path(step=42)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    saved = Progress(step=42, total_tokens=1000, total_samples=50, total_problems=10, next_group_id=137)
    with open(ckpt_path / "progress.pt", "wb") as f:
        torch.save({"progress": saved}, f)

    fresh_progress = Progress()
    # Buffer.load is irrelevant to this test — patch it to a no-op.
    with patch("prime_rl.orchestrator.ckpt.Buffer") as MockBuffer:
        MockBuffer.return_value.load = lambda *_, **__: None
        mgr.load_from_path(ckpt_path, fresh_progress, MockBuffer.return_value)

    # Other progress fields stay at fresh defaults (skip_progress=True),
    # but next_group_id must be carried over since the buffer's rollouts
    # carry stamped group_ids that depend on it.
    assert fresh_progress.step == 0, "skip_progress should leave step at default"
    assert fresh_progress.next_group_id == 137, (
        "next_group_id must be restored even when skip_progress=True, since the "
        "buffer's stamped rollouts depend on it"
    )


def test_skip_buffer_does_not_resurrect_next_group_id(tmp_path):
    """When skip_buffer=True the buffer's rollouts are discarded, so
    next_group_id starts fresh at 0 and there's nothing to collide against."""
    import torch
    from unittest.mock import patch

    from prime_rl.configs.orchestrator import CheckpointConfig
    from prime_rl.orchestrator.ckpt import CheckpointManager

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    cfg = CheckpointConfig(skip_progress=True, skip_buffer=True)
    mgr = CheckpointManager(output_dir, cfg)
    ckpt_path = mgr.get_ckpt_path(step=42)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    saved = Progress(step=42, next_group_id=137)
    with open(ckpt_path / "progress.pt", "wb") as f:
        torch.save({"progress": saved}, f)

    fresh = Progress()
    with patch("prime_rl.orchestrator.ckpt.Buffer") as MockBuffer:
        MockBuffer.return_value.load = lambda *_, **__: None
        mgr.load_from_path(ckpt_path, fresh, MockBuffer.return_value)
    assert fresh.next_group_id == 0
