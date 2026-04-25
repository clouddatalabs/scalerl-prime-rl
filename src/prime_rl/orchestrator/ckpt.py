import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from prime_rl.configs.orchestrator import CheckpointConfig
from prime_rl.orchestrator.buffer import Buffer
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import get_ckpt_dir, get_step_path


@dataclass
class Progress:
    step: int = 0
    total_tokens: int = 0
    total_samples: int = 0
    total_problems: int = 0
    # Monotonic group_id counter from the scheduler. Persisted across resume so
    # rollouts saved by `Buffer.save` (with stamped `group_id`s in {0..N})
    # don't collide with freshly-issued ids on resume — that collision would
    # silently merge unrelated prompts in `apply_prompt_average_sequence_weights`'s
    # group-keyed normalization. The orchestrator copies into / out of
    # `Scheduler.next_group_id` around save/load.
    next_group_id: int = 0


class CheckpointManager:
    """Utility class to save and load orchestrator checkpoints to resume orchestrator."""

    def __init__(self, output_dir: Path, config: CheckpointConfig):
        self.config = config
        self.ckpt_dir = get_ckpt_dir(output_dir)
        self.logger = get_logger()

    def get_ckpt_path(self, step: int) -> Path:
        return get_step_path(self.ckpt_dir, step) / "orchestrator"

    def save_to_path(
        self,
        ckpt_path: Path,
        progress: Progress,
        buffer: Buffer,
    ):
        self.logger.debug(f"Saving orchestrator checkpoint to {ckpt_path}")
        start_time = time.perf_counter()

        # Save progress
        with open(ckpt_path / "progress.pt", "wb") as f:
            torch.save({"progress": progress}, f)

        # Save buffer
        buffer.save(ckpt_path / "buffer")

        self.logger.debug(f"Orchestrator checkpoint saved in {time.perf_counter() - start_time:.2f} seconds")

    def load_from_path(self, ckpt_path: Path, progress: Progress, buffer: Buffer) -> None:
        """Loads a checkpoint from a given path in-place."""
        self.logger.debug(f"Loading checkpoint from {ckpt_path}")
        start_time = time.perf_counter()

        # `next_group_id` is logically a buffer-side datum (it stamps rollouts
        # in `rollout_buffer.jsonl`), even though it lives on Progress for
        # serialization. Restore it iff the buffer is being restored — otherwise
        # `skip_progress=True + skip_buffer=False` (the default) would reset
        # the counter to 0 while old rollouts on disk still carry stamped
        # group_ids in {0..N}, silently colliding in `apply_prompt_average_sequence_weights`.
        loaded_progress = None
        with open(ckpt_path / "progress.pt", "rb") as f:
            loaded_progress = torch.load(f, weights_only=False)["progress"]

        if self.config.skip_progress:
            self.logger.info("Skipping progress loading from checkpoint")
        else:
            for key, value in asdict(loaded_progress).items():
                setattr(progress, key, value)

        # Load buffer
        if self.config.skip_buffer:
            self.logger.info("Skipping buffer loading from checkpoint")
        else:
            buffer.load(ckpt_path / "buffer")
            # If skip_progress dropped next_group_id but we're keeping the
            # buffer, restore just that field so the scheduler doesn't reissue
            # colliding ids against the restored rollouts.
            if self.config.skip_progress:
                progress.next_group_id = loaded_progress.next_group_id
                self.logger.info(
                    "Restored next_group_id=%d from checkpoint (skip_progress is set "
                    "but the buffer was loaded; the field is logically buffer-side).",
                    progress.next_group_id,
                )

        self.logger.debug(f"Orchestrator checkpoint loaded in {time.perf_counter() - start_time:.2f} seconds")

    def load(self, progress: Progress, buffer: Buffer, step: int) -> None:
        """Loads a checkpoint from a given path."""
        ckpt_path = self.get_ckpt_path(step)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        self.load_from_path(ckpt_path, progress, buffer)

    def save(
        self,
        progress: Progress,
        buffer: Buffer,
        step: int,
    ) -> None:
        """Saves the full checkpoint state for a specified step."""
        ckpt_path = self.get_ckpt_path(step)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        self.save_to_path(ckpt_path, progress, buffer)


def setup_ckpt_manager(output_dir: Path, config: CheckpointConfig | None) -> CheckpointManager | None:
    if config is None:
        return None
    return CheckpointManager(output_dir, config)
