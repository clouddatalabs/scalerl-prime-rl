"""Multi-run checkpointing for RL training.

MultiCheckpointManager owns per-run CheckpointManagers and AppStates,
each saving to its own run directory.
"""

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful

from prime_rl.configs.trainer import CheckpointConfig
from prime_rl.trainer.ckpt import CheckpointManager
from prime_rl.trainer.runs import Progress, get_multi_run_manager
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_stable_ckpt_steps

if TYPE_CHECKING:
    from prime_rl.trainer.optim import MultiLoRAOptimizer
    from prime_rl.trainer.scheduler import MultiLoRAScheduler


class RunState(Stateful):
    """Per-run state wrapper - just like AppState but for adapter weights."""

    def __init__(
        self,
        model_state_dict: dict[str, Any],
        optimizer,
        scheduler,
        progress: Progress,
    ):
        self.model_state_dict = model_state_dict
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.progress = progress

    def state_dict(self) -> dict[str, Any]:
        state = {
            "model": self.model_state_dict,
            "progress": asdict(self.progress),
        }
        if self.optimizer is not None:
            state["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        return state

    @torch.no_grad()
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Load adapter weights
        for key, value in state_dict["model"].items():
            if key in self.model_state_dict:
                self.model_state_dict[key].copy_(value)
        # Load optimizer
        if "optimizer" in state_dict and self.optimizer is not None:
            self.optimizer.load_state_dict(state_dict["optimizer"])
        # Load scheduler
        if "scheduler" in state_dict and self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict["scheduler"])
        # Don't load progress because it resets packers count
        # There will be a step issue if we load progress


class MultiCheckpointManager:
    """Owns per-run CheckpointManagers and AppStates."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.multi_run_manager = get_multi_run_manager()
        self.world = get_world()
        self.logger = get_logger()
        self.managers: list[CheckpointManager | None] = [None] * self.multi_run_manager.max_runs
        self.multi_run_manager.register_deletion_hook(self._run_deletion_hook)
        self.multi_run_manager.register_creation_hook(self._run_creation_hook)

    def _run_deletion_hook(self, idx: int, run_id: str) -> None:
        self.managers[idx] = None

    def _run_creation_hook(self, idx: int, run_id: str) -> None:
        self.managers[idx] = self._maybe_create_manager(idx)

    def _maybe_create_manager(self, idx: int) -> CheckpointManager | None:
        ckpt_config = self.multi_run_manager.config[idx].ckpt
        if ckpt_config is None:
            return None

        config = CheckpointConfig(
            interval=ckpt_config.interval,
            keep_last=ckpt_config.keep_last,
            keep_interval=ckpt_config.keep_interval,
        )
        run_dir = self.multi_run_manager.get_run_dir(idx)
        manager = CheckpointManager(run_dir, config)
        self.managers[idx] = manager
        return manager

    def _should_save(self, idx: int, step: int) -> bool:
        """Determine if a checkpoint should be saved for a given run and step."""
        ckpt_config = self.multi_run_manager.config[idx].ckpt
        if ckpt_config is None or ckpt_config.interval is None:
            return False
        if step <= 0 or step % ckpt_config.interval != 0:
            return False
        # Check if already saved this step
        return step not in self.managers[idx].ckpt_steps

    def save(
        self,
        optimizer: "MultiLoRAOptimizer",
        scheduler: "MultiLoRAScheduler",
    ) -> None:
        for idx in self.multi_run_manager.used_idxs:
            step = self.multi_run_manager.progress[idx].step
            if not self._should_save(idx, step):
                continue

            manager = self.managers[idx]

            # Per-rank `saved_ok` flag — `mark_stable` (master-only STABLE
            # touch) and `ckpt_steps.append` (per-rank list) must only run
            # on ranks that actually finished their save. The barrier below
            # is OUTSIDE the try-except so every rank reaches it regardless
            # of success/failure: if it were inside the try (the original
            # structure), a per-rank exception (ENOSPC, FS race, run-dir
            # deleted mid-save) would skip the barrier on the failing rank
            # while peers blocked at it forever — a hard NCCL deadlock that
            # `clean_exit` cannot rescue because the exception is swallowed
            # internally.
            saved_ok = False

            # We have a very wide try-except because we dont want to crash the trainer over one run having issues
            try:
                model_state_dict = {
                    k: v.data.detach().clone() for k, v in self.multi_run_manager.get_named_parameters_for_run(idx)
                }
                run_state = RunState(
                    model_state_dict,
                    optimizer.optimizers[idx],
                    scheduler.schedulers[idx],
                    self.multi_run_manager.progress[idx],
                )
                ckpt_path = manager.get_ckpt_path(step)
                ckpt_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(
                    f"Saving checkpoint for run {idx} at step {step} to {ckpt_path / f'rank_{self.world.rank}.pt'}"
                )
                torch.save(run_state.state_dict(), ckpt_path / f"rank_{self.world.rank}.pt")

                # Copy broadcast folder to checkpoint
                # This way, we only need to save the checkpoint folder
                if self.world.is_master:
                    run_dir = self.multi_run_manager.get_run_dir(idx)
                    broadcast_src = run_dir / "broadcasts" / f"step_{step}"
                    weight_dst = run_dir / "checkpoints" / f"step_{step}" / "weight"
                    try:
                        shutil.copytree(broadcast_src, weight_dst)
                    except FileNotFoundError:
                        # Broadcast folder absent means weight isn't pinned
                        # under the checkpoint dir — the rank shards are
                        # still on disk, but eval/resume scanning for the
                        # broadcast folder will miss this step. Logged for
                        # visibility, but treated as soft (saved_ok stays True).
                        self.logger.error(
                            f"Broadcast folder not found for run {idx} at step {step}. Looking for it in {broadcast_src}"
                        )
                    except OSError as e:
                        # Anything else master-only (FileExistsError if
                        # weight_dst exists from a prior partial save,
                        # ENOSPC mid-copy, EACCES) used to escape to the
                        # outer `except Exception` and quietly mark master
                        # as failed while workers stayed `saved_ok=True`.
                        # That divergence skipped the STABLE marker (master-
                        # only) without diverging the rank-shard saves on
                        # disk: the rank files exist but eval/resume cannot
                        # see them. Catch it locally so master's saved_ok
                        # is consistent with the workers'.
                        self.logger.error(
                            f"Master broadcast-copy failed for run {idx} at step {step}: {type(e).__name__}: {e}"
                        )
                saved_ok = True
            except FileNotFoundError:
                self.logger.warning(f"Run {idx} deleted during checkpoint, skipping")
            except Exception as e:
                # Catch ALL exceptions — narrowing to `OSError` previously left
                # logical bugs (None-deref in `run_state`, schema regression
                # that makes a value non-picklable, KeyError, RuntimeError
                # from torch.save) to escape the per-`idx` block, skipping
                # the rank's `dist.barrier()` (line below) and
                # `all_reduce(saved_ok)` collectives. Peers blocked on those
                # collectives until NCCL watchdog fired (~10 min). Catching
                # broad-Exception ensures every rank reaches the barrier,
                # but log with full type+traceback so operators see logical
                # bugs in the run log on the first occurrence — and the
                # all-reduce-MIN converts "any rank failed" to a global
                # `saved_ok=False` so `mark_stable` is skipped consistently
                # rather than silently writing a STABLE marker over a torn
                # save.
                import traceback as _tb

                self.logger.error(
                    f"Error checkpointing run {idx}: {type(e).__name__}: {e}\n"
                    f"{_tb.format_exc()}"
                )

            # Single sync point — every rank reaches this regardless of
            # try-except outcome. Replaces the prior pair of barriers (one
            # inside the try, one after) which deadlocked on per-rank failure.
            dist.barrier()

            # Reduce per-rank `saved_ok` to a global "all ranks succeeded"
            # bool so ranks agree before mark_stable / ckpt_steps.append.
            # Without this, a rank-asymmetric failure (e.g. master-only
            # copytree error escaping the inner catch) leaves master with
            # `saved_ok=False` and workers with True — workers append step
            # to their per-rank ckpt_steps and master doesn't, so the
            # master-only `mark_stable` STABLE marker is never written and
            # the step is invisible to eval/resume even though rank shards
            # exist on disk.
            saved_ok_tensor = torch.tensor(int(saved_ok), device="cuda")
            dist.all_reduce(saved_ok_tensor, op=dist.ReduceOp.MIN)
            saved_ok = bool(saved_ok_tensor.item())

            if saved_ok:
                manager.mark_stable(step)
                manager.ckpt_steps.append(step)
            # If the run is deleted, remove the run directory
            # This is avoid the creation of zombie runs when the directory is deleted while we are checkpointing which recreates the directory
            # Ideally we move this to discover but lets have here for now
            if (
                self.multi_run_manager.get_orchestrator_config(self.multi_run_manager.idx_2_id[idx]) is None
                and self.world.is_master
            ):
                try:
                    self.logger.warning(f"Run {idx} deleted during checkpoint, removing run directory")
                    shutil.rmtree(self.multi_run_manager.get_run_dir(idx))
                except Exception as e:
                    self.logger.error(f"Error removing run directory for run {idx}: {e}")
        dist.barrier()

    def load_run(
        self,
        idx: int,
        optimizer: "MultiLoRAOptimizer",
        scheduler: "MultiLoRAScheduler",
    ) -> bool:
        if (
            self.multi_run_manager.config[idx].ckpt is None
            or self.multi_run_manager.config[idx].ckpt.resume_step is None
        ):
            return False

        manager = self.managers[idx]
        if manager is None:
            return False

        step = self.multi_run_manager.config[idx].ckpt.resume_step
        if step == -1:
            stable_steps = get_stable_ckpt_steps(manager.ckpt_dir)
            if not stable_steps:
                return False
            step = max(stable_steps)

        load_ok = False
        try:
            model_state_dict = dict(self.multi_run_manager.get_named_parameters_for_run(idx))
            run_state = RunState(
                model_state_dict,
                optimizer.optimizers[idx],
                scheduler.schedulers[idx],
                self.multi_run_manager.progress[idx],
            )
            ckpt_path = manager.get_ckpt_path(step)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
            self.logger.info(f"Loading checkpoint from {ckpt_path}")
            state_dict = torch.load(ckpt_path / f"rank_{self.world.rank}.pt", weights_only=False)
            run_state.load_state_dict(state_dict)

            self.logger.info(f"Resumed run {self.multi_run_manager.idx_2_id[idx]} from step {step}")
            load_ok = True
        except Exception as e:
            self.logger.error(f"Error loading checkpoint for run {idx}: {e}")

        # Reduce per-rank `load_ok` to a global "every rank loaded
        # successfully" — mirrors the `save()` MIN-reduce pattern. Without
        # this, 7/8 ranks loading successfully but rank K hitting a corrupt
        # `rank_K.pt` or transient FS error left training continuing with
        # rank K's freshly-zeroed/stale adapter weights against 7 ranks
        # holding the loaded weights. Subsequent FSDP all-gather mixed
        # them silently, corrupting that adapter for the rest of the run.
        load_ok_tensor = torch.tensor(int(load_ok), device="cuda")
        dist.all_reduce(load_ok_tensor, op=dist.ReduceOp.MIN)
        return bool(load_ok_tensor.item())

    def maybe_clean(self) -> None:
        if not self.world.is_master:
            return
        for idx in self.multi_run_manager.used_idxs:
            if self.managers[idx] is None:
                continue
            self.managers[idx].maybe_clean()


def setup_multi_checkpoint_manager(output_dir: Path) -> tuple[MultiCheckpointManager, None]:
    return MultiCheckpointManager(output_dir), None
