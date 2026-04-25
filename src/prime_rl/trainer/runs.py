import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import tomli
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

from prime_rl.configs.trainer import LoRAConfig
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_all_ckpt_steps, get_stable_ckpt_steps

if TYPE_CHECKING:
    from prime_rl.configs.orchestrator import OrchestratorConfig
    from prime_rl.trainer.models.layers.lora import MultiLoRALinear


@dataclass
class Progress:
    step: int = 0
    total_tokens: int = 0
    total_samples: int = 0


class MultiRunManager:
    """This class stores information about the runs in the system."""

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(self, output_dir: Path, max_runs: int, device: torch.device = torch.device("cpu")):
        self.output_dir = output_dir
        self.max_runs = max_runs
        self.logger = get_logger()

        self.idx_2_id: dict[int, str] = {}
        self.id_2_idx: dict[str, int] = {}
        self.unused_idxs = {i for i in range(self.max_runs)}

        self.progress: dict[int, Progress] = {}
        self.config: dict[int, "OrchestratorConfig"] = {}
        self.ready_to_update = [False] * max_runs

        self._creation_hooks: list[Callable[[int, str], None]] = []
        self._deletion_hooks: list[Callable[[int, str], None]] = []
        self._discovered_hooks: list[Callable[[int, str, "OrchestratorConfig"], None]] = []
        self._forgotten_hooks: list[Callable[[int, str], None]] = []
        self._config_validation_hooks: list[Callable[["OrchestratorConfig"], tuple[bool, str]]] = []

        # We use the store to keep other ranks in sync with master
        self.store = c10d._get_default_store()
        self.world = get_world()
        # Track id_2_idx state at last synchronize_state to calculate diffs
        self._last_synced_id_2_idx: dict[str, int] = {}

        # Store modules with their FQN prefixes for parameter management
        self._modules: list[tuple[str, "MultiLoRALinear"]] = []

        # Optional conversion applied to adapter state dicts (e.g. PrimeRL -> HF key rename)
        self._adapter_state_dict_converter: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None

        # Initialize lora globals on device so runs.* ARE the global tensors
        from prime_rl.trainer.models.layers.lora import (
            get_lora_num_tokens,
            get_multilora_scaling,
            set_lora_num_tokens,
            set_multilora_scaling,
        )

        # We first set to None so we dont check the shape
        # Might be better to not have the check at all but this will do for now
        set_lora_num_tokens(None, reset_reference=True)
        set_multilora_scaling(None, reset_reference=True)

        set_lora_num_tokens(torch.zeros(max_runs, dtype=torch.int32, device=device), reset_reference=True)
        self.lora_num_tokens = get_lora_num_tokens()

        set_multilora_scaling(
            torch.ones(max_runs, dtype=torch.bfloat16, device=device) * 1000_000.0, reset_reference=True
        )
        self.scaling_factors = get_multilora_scaling()

    # =========================================================================
    # Hook Registration
    # =========================================================================

    def register_creation_hook(self, hook: Callable[[int, str], None]) -> None:
        """Register a hook to be called when a new run is created.

        Args:
            hook: A callable that takes (idx: int, run_id: str) as arguments.
                  Called on all ranks when a new run is added to the system.
        """
        self._creation_hooks.append(hook)

    def register_deletion_hook(self, hook: Callable[[int, str], None]) -> None:
        """Register a hook to be called when a run is deleted.

        Args:
            hook: A callable that takes (idx: int, run_id: str) as arguments.
                  Called on all ranks when a run is removed from the system.
        """
        self._deletion_hooks.append(hook)

    def register_discovered_hook(self, hook: Callable[[int, str, "OrchestratorConfig"], None]) -> None:
        """Register a hook to be called when a new run is discovered (master only).

        Args:
            hook: A callable that takes (idx: int, run_id: str, config: OrchestratorConfig).
                  Called only on master rank in discover_runs() when a new run is found.
        """
        if not self.world.is_master:
            raise RuntimeError("register_discovered_hook() must only be called on the master rank")
        self._discovered_hooks.append(hook)

    def register_forgotten_hook(self, hook: Callable[[int, str], None]) -> None:
        """Register a hook to be called when a run is forgotten/removed (master only).

        Args:
            hook: A callable that takes (idx: int, run_id: str).
                  Called only on master rank in discover_runs() when a run is removed.
        """
        if not self.world.is_master:
            raise RuntimeError("register_forgotten_hook() must only be called on the master rank")
        self._forgotten_hooks.append(hook)

    def register_config_validation_hook(self, hook: Callable[["OrchestratorConfig"], tuple[bool, str]]) -> None:
        """Register a hook to validate orchestrator config when a run is created.

        Args:
            hook: A callable that takes (config: OrchestratorConfig) and returns
                  (is_valid: bool, error_message: str). Error message is used when invalid.
        """
        if not self.world.is_master:
            raise RuntimeError("register_config_validation_hook() must only be called on the master rank")
        self._config_validation_hooks.append(hook)

    # =========================================================================
    # Module Registration and Parameter Management
    # =========================================================================

    def register_adapter_state_dict_converter(
        self, converter: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]
    ) -> None:
        """Register a converter applied in-place to adapter state dicts (e.g. model.convert_adapter_to_hf)."""
        self._adapter_state_dict_converter = converter

    def register_module(self, prefix: str, module: "MultiLoRALinear") -> None:
        """Register a MultiLoRALinear module with its FQN prefix.

        This allows MultiRunManager to manage parameter access, reset, and state dict slicing
        for multi-adapter LoRA modules.

        Args:
            prefix: The module's fully qualified name in the model
                   (e.g., "model.layers.0.self_attn.q_proj")
            module: The MultiLoRALinear module to register
        """
        self._modules.append((prefix, module))

    def get_named_parameters_for_run(self, idx: int) -> list[tuple[str, torch.nn.Parameter]]:
        """Get named parameters for a specific run index.

        Args:
            idx: The run index to get parameters for

        Returns:
            List of (name, parameter) tuples for the specified run index
        """
        params = []
        for prefix, module in self._modules:
            for name, param in module.named_parameters_for_adapter(idx):
                params.append((f"{prefix}.{name}.weight", param))
        return params

    def get_state_dict_for_run(self, idx: int) -> dict[str, torch.Tensor]:
        """Get state dict for a specific run index.

        Args:
            idx: The run index to get state dict for

        Returns:
            State dict for the specified run index
        """
        state_dict = {}
        for prefix, module in self._modules:
            # Check if module has a custom state_dict_for_adapter method (e.g., MoE modules)
            # which returns vLLM-compatible per-expert format
            if hasattr(module, "state_dict_for_adapter"):
                for name, tensor in module.state_dict_for_adapter(idx).items():
                    state_dict[f"{prefix}.{name}"] = tensor.detach()
            else:
                # Default: use named_parameters_for_adapter
                for name, param in module.named_parameters_for_adapter(idx):
                    state_dict[f"{prefix}.{name}.weight"] = param.detach()

        if self._adapter_state_dict_converter is not None:
            state_dict = self._adapter_state_dict_converter(state_dict)
        return state_dict

    def reset_run_parameters(self, idx: int) -> None:
        """Reset parameters for a specific run index.

        Called when a new run is created to initialize fresh adapter weights.

        Args:
            idx: The run index to reset parameters for
        """
        for _, module in self._modules:
            module.reset_parameters(idx)

    # =========================================================================
    # Config Loading
    # =========================================================================

    def get_orchestrator_config(self, run_id: str) -> Optional["OrchestratorConfig"]:
        """Load and validate orchestrator config for a run.

        Returns None if config doesn't exist, fails to parse, or fails validation.
        Writes error to config dir for orchestrator to consume.
        """
        config_path = self.output_dir / run_id / "control" / "orch.toml"
        config_dir = config_path.parent
        error_path = config_dir / "config_validation_error.txt"

        if not config_path.exists():
            self.logger.error(f"Run {run_id}: No orchestrator config found at {config_path}")
            return None

        try:
            with open(config_path, "rb") as f:
                config_dict = tomli.load(f)

            from prime_rl.configs.orchestrator import OrchestratorConfig

            config = OrchestratorConfig(**config_dict)
        except Exception as e:
            if error_path.parent.exists():
                with open(error_path, "w") as f:
                    f.write(f"Error parsing orchestrator config:\n{str(e)}\n")
            self.logger.error(f"Run {run_id}: Error parsing orchestrator config: {e}")
            return None

        # Run registered validation hooks
        for hook in self._config_validation_hooks:
            is_valid, error_message = hook(config)
            if not is_valid:
                self.logger.error(f"Run {run_id}: {error_message}")
                if error_path.parent.exists():
                    with open(error_path, "w") as f:
                        f.write(f"{error_message}\n")
                return None

        # Config is valid, remove any stale error file
        if error_path.exists():
            error_path.unlink()

        return config

    # =========================================================================
    # Internal Run Data Helpers
    # =========================================================================

    def _create_run_data(self, new_run: str, new_id: int, config: "OrchestratorConfig") -> None:
        """Update data structures for a new run (no hooks or param reset)."""
        self.id_2_idx[new_run] = new_id
        self.unused_idxs.remove(new_id)
        self.idx_2_id[new_id] = new_run

        # Set progress based on resume_step config (match orchestrator behavior)
        self.progress[new_id] = Progress()
        if config.ckpt is None or config.ckpt.resume_step is None:
            self.progress[new_id].step = 0
        elif config.ckpt.resume_step == -1:
            ckpt_dir = self.get_run_dir(new_id) / "checkpoints"
            # In multi-run, the trainer writes STABLE after saving LoRA weights to the run's checkpoint dir.
            # In single-run, only the orchestrator writes checkpoints here (trainer has its own dir), so no STABLE exists.
            steps = get_stable_ckpt_steps(ckpt_dir) if self.max_runs > 1 else get_all_ckpt_steps(ckpt_dir)
            self.progress[new_id].step = max(steps) if steps else 0
        else:
            self.progress[new_id].step = config.ckpt.resume_step

        # Store the parsed config
        self.config[new_id] = config

    def _delete_run_data(self, deleted_run: str, deleted_idx: int) -> None:
        """Update data structures for a deleted run (internal cleanup only, no hooks)."""
        del self.progress[deleted_idx]
        if deleted_idx in self.config:
            del self.config[deleted_idx]

        # A big value might help make it error loudly
        self.scaling_factors[deleted_idx] = 1000_000.0

        # Process mappings
        self.unused_idxs.add(deleted_idx)
        del self.idx_2_id[deleted_idx]
        del self.id_2_idx[deleted_run]

    def _create_run_hooks(self, new_id: int, new_run: str) -> None:
        """Reset parameters and call creation hooks for a run."""
        self.reset_run_parameters(new_id)
        for hook in self._creation_hooks:
            hook(new_id, new_run)

    def _delete_run_hooks(self, deleted_idx: int, deleted_run: str) -> None:
        """Call deletion hooks for a run."""
        for hook in self._deletion_hooks:
            hook(deleted_idx, deleted_run)

    # =========================================================================
    # Run Discovery, Synchronization, and Eviction
    # =========================================================================

    def evict_run(self, idx: int, reason: str) -> None:
        """Evict a run by writing the reason to a file for the orchestrator to read.

        The orchestrator will error and surface this reason. The run will be
        ignored by future discover_runs() calls.
        Note that the run is not deleted on master until the next discover_runs() call.
        And not deleted on other ranks until the next synchronize_state() call.

        Args:
            idx: The run index to evict
            reason: The reason for eviction (will be shown to user)
        """
        if not self.world.is_master:
            raise RuntimeError("evict_run() must only be called on the master rank")

        if idx not in self.idx_2_id:
            self.logger.warning(f"Run index {idx} not found, cannot evict")
            return

        run_id = self.idx_2_id[idx]
        run_dir = self.output_dir / run_id
        config_dir = run_dir / "control"
        config_dir.mkdir(parents=True, exist_ok=True)

        evicted_path = config_dir / "evicted.txt"
        with open(evicted_path, "w") as f:
            f.write(reason)

        self.logger.warning(f"Evicted run {run_id} (idx={idx}): {reason}")

    def discover_runs(self) -> None:
        """Detect run changes and update data structures (master only). Must be followed by synchronize_state().

        Scans for new/deleted runs, calls forgotten/discovered hooks (master only),
        and updates internal data structures. Hooks and parameter resets for all ranks
        are deferred to synchronize_state().
        """
        if not self.world.is_master:
            raise RuntimeError("discover_runs() must only be called on the master rank")
        run_ids = {run_path.stem for run_path in self.output_dir.glob("run_*")}

        # Filter out evicted runs
        evicted_runs = {run_id for run_id in run_ids if (self.output_dir / run_id / "control" / "evicted.txt").exists()}
        run_ids = run_ids - evicted_runs

        deleted_runs = self.id_2_idx.keys() - run_ids
        new_runs = run_ids - self.id_2_idx.keys()

        for deleted_run in deleted_runs:
            deleted_idx = self.id_2_idx[deleted_run]
            # Call forgotten hooks (master only) before cleanup
            for hook in self._forgotten_hooks:
                hook(deleted_idx, deleted_run)
            self._delete_run_data(deleted_run, deleted_idx)

        for new_run in new_runs:
            try:
                new_id = next(iter(self.unused_idxs))

                config = self.get_orchestrator_config(new_run)
                if config is None:
                    continue

                self._create_run_data(new_run, new_id, config)
                # Call discovered hooks (master only) after data setup
                for hook in self._discovered_hooks:
                    hook(new_id, new_run, config)
            except StopIteration:
                continue

    def synchronize_state(self) -> None:
        """Sync run state across ranks and execute hooks.

        Master calculates what changed since last sync using _last_synced_id_2_idx.
        This matches what non-master ranks will calculate as new/deleted runs.
        All ranks then execute hooks and parameter resets together.
        """

        # Wrap the master-only `pickle.dumps + store.set` in try/finally so
        # the barrier is reached on every rank regardless of master failure.
        # Without finally, `pickle.dumps(sync_data)` raising on master (e.g.
        # an OrchestratorConfig carrying a non-picklable hook closure, a
        # custom Path subclass, or any future schema change that introduces
        # a non-picklable field) skips the barrier on master while peer
        # ranks block at `dist.barrier()` until NCCL watchdog kills the
        # job (~10 min). This validator runs every step on the RL trainer
        # via `DataLoader.wait_for_batch`, so a one-shot pickling regression
        # would brick every subsequent step.
        # Use a step-versioned key so peers can detect a stale read. Without
        # the version, if `pickle.dumps(sync_data)` raises on master BEFORE
        # the `store.set("runs", ...)` write, the store key keeps its
        # PREVIOUS sync's payload — peers happily decode the stale data and
        # run `_create_run_data`/`_delete_run_data`/creation+deletion hooks
        # on phantom runs for one-or-more steps before the next collective
        # crashes. Tagging the payload with a counter master increments on
        # every successful set lets peers detect "no new data this step"
        # and fail fast.
        sync_call_idx = getattr(self, "_sync_call_idx", 0) + 1
        self._sync_call_idx = sync_call_idx
        try:
            if self.world.is_master:
                # Include configs for new runs so non-master ranks have them
                new_runs = self.id_2_idx.keys() - self._last_synced_id_2_idx.keys()
                new_configs = {new_run: self.config[self.id_2_idx[new_run]] for new_run in new_runs}

                sync_data = {
                    "id_2_idx": self.id_2_idx,
                    "ready_to_update": self.ready_to_update,
                    "scaling_factors": self.scaling_factors.cpu(),
                    "new_configs": new_configs,
                    "progress": self.progress,
                    "_sync_call_idx": sync_call_idx,
                }
                self.store.set("runs", pickle.dumps(sync_data))
        finally:
            dist.barrier()

        if self.world.is_master:
            # Calculate changes since last sync (this is what other ranks will see)
            new_runs = self.id_2_idx.keys() - self._last_synced_id_2_idx.keys()
            deleted_runs = self._last_synced_id_2_idx.keys() - self.id_2_idx.keys()
            # Capture deleted indices from last synced state (already removed from id_2_idx)
            deleted_run_idxs = {run: self._last_synced_id_2_idx[run] for run in deleted_runs}
        else:
            sync_data: dict = pickle.loads(self.store.get("runs"))
            received_idx = sync_data.get("_sync_call_idx")
            if received_idx != sync_call_idx:
                # Master failed to write the current sync's payload —
                # peers are about to act on a stale snapshot from a
                # previous sync. Bail loudly instead of mutating run
                # state on phantom data. Master's exception will surface
                # via clean_exit; peers raise here so the whole world
                # fails together.
                raise RuntimeError(
                    f"MultiRunManager.synchronize_state: stale store payload "
                    f"(expected sync #{sync_call_idx}, got #{received_idx}). "
                    "Master likely raised before writing. Refusing to apply "
                    "stale state to peer ranks."
                )
            new_id_2_idx: dict[str, int] = sync_data["id_2_idx"]
            self.ready_to_update = sync_data["ready_to_update"]
            self.scaling_factors.copy_(sync_data["scaling_factors"])
            new_configs: dict[str, "OrchestratorConfig"] = sync_data["new_configs"]
            master_progress: dict[int, Progress] = sync_data["progress"]

            new_runs = new_id_2_idx.keys() - self.id_2_idx.keys()
            deleted_runs = self.id_2_idx.keys() - new_id_2_idx.keys()
            # Capture deleted indices before removing from id_2_idx
            deleted_run_idxs = {run: self.id_2_idx[run] for run in deleted_runs}

            # Other ranks catch up with master's data state
            for deleted_run in deleted_runs:
                deleted_idx = deleted_run_idxs[deleted_run]
                self._delete_run_data(deleted_run, deleted_idx)

            for new_run in new_runs:
                new_id = new_id_2_idx[new_run]
                self._create_run_data(new_run, new_id, new_configs[new_run])

            # Sync progress from master
            self.progress = master_progress

        # Call deletion hooks on all ranks
        for deleted_run in deleted_runs:
            deleted_idx = deleted_run_idxs[deleted_run]
            self._delete_run_hooks(deleted_idx, deleted_run)

        for new_run in new_runs:
            new_id = self.id_2_idx[new_run]
            self._create_run_hooks(new_id, new_run)

        # Update last synced state for master
        if self.world.is_master:
            self._last_synced_id_2_idx = self.id_2_idx.copy()

    # =========================================================================
    # Properties and Accessors
    # =========================================================================

    @property
    def used_idxs(self):
        return sorted(self.idx_2_id.keys())

    @property
    def ready_to_update_idxs(self):
        return [idx for idx, ready in enumerate(self.ready_to_update) if ready]

    def run_dirs(self) -> list[Path]:
        return [self.output_dir / run_id for run_id in self.id_2_idx.keys()]

    def get_run_dir(self, idx: int) -> Path:
        return self.output_dir / self.idx_2_id[idx]

    def __repr__(self):
        return f"MultiRunManager(max={self.max_runs})[{self.idx_2_id.keys()}]"


# Singleton instance of MultiRunManager
_MULTI_RUN_MANAGER: MultiRunManager | None = None


def get_multi_run_manager() -> MultiRunManager:
    """Returns the MultiRunManager singleton. Must be initialized first via setup_multi_run_manager()."""
    global _MULTI_RUN_MANAGER
    if _MULTI_RUN_MANAGER is None:
        raise RuntimeError("MultiRunManager not initialized. Please call `setup_multi_run_manager` first.")
    return _MULTI_RUN_MANAGER


def _validate_orch_lora_against_trainer(
    orch_config: "OrchestratorConfig", trainer_lora: LoRAConfig
) -> tuple[bool, str]:
    """Validate an orchestrator config's LoRA section against the trainer's LoRA.

    Mutates the orch config in place to fill in defaults from the trainer
    (rank/alpha may be omitted in the orch TOML to inherit). Returns the
    standard `(is_valid, error_message)` contract used by the multi-run
    config-validation hook chain. Extracted from the closure inside
    `setup_multi_run_manager` so unit tests can pin the contract — the
    `orch_config.model.lora is None` rejection is the load-bearing path
    that prevents an AttributeError on the trainer's master rank.
    """
    if orch_config.model.lora is None:
        return (
            False,
            "[model.lora] section is required in the orchestrator config when "
            "the trainer is configured with LoRA. Add `[model.lora]` (rank "
            "and alpha may be omitted to inherit the trainer's values).",
        )
    # Read explicit per-config values BEFORE applying defaults — the
    # trainer-equality checks below need to see what the operator actually
    # wrote, not what the inheritance pass filled in. Mutating the config
    # before a possible False return also leaves a half-edited config in
    # the caller's hands; checking first keeps the contract pure: validate,
    # THEN inherit, THEN return.
    requested_rank = orch_config.model.lora.rank
    requested_alpha = orch_config.model.lora.alpha
    effective_rank = requested_rank if requested_rank is not None else trainer_lora.rank
    effective_alpha = requested_alpha if requested_alpha is not None else trainer_lora.alpha

    if effective_rank > trainer_lora.rank:
        return (
            False,
            f"model.lora.rank ({effective_rank}) exceeds trainer max rank ({trainer_lora.rank})",
        )
    # Per-run alpha must match the trainer's (the single-run path's
    # `auto_setup_lora` already enforces this; multi-run was strictly
    # weaker without justification). The scaling factor in
    # `MultiRunManager.scaling_factors` is `alpha/rank`, used by the
    # trainer's optimizer step. If the orchestrator's saved adapter
    # config — which vLLM consumes at adapter-load time — disagrees,
    # train and inference apply different scales and the adapter
    # silently mistrains.
    if effective_alpha != trainer_lora.alpha:
        return (
            False,
            f"model.lora.alpha ({effective_alpha}) does not match "
            f"trainer.model.lora.alpha ({trainer_lora.alpha}). The trainer "
            "applies alpha/rank as the LoRA scaling factor; vLLM reads alpha "
            "from the saved adapter config at load time. Disagreement here "
            "silently mistrains the adapter (train and inference scale by "
            "different factors). Either omit alpha from the orchestrator "
            "config to inherit, or set it equal to trainer.model.lora.alpha.",
        )
    # Apply inherited defaults only on the success path so a False return
    # never leaves the caller with a half-mutated config.
    if requested_rank is None:
        orch_config.model.lora.rank = trainer_lora.rank
    if requested_alpha is None:
        orch_config.model.lora.alpha = trainer_lora.alpha
    return True, ""


def _validate_orch_optim_lr_explicit(orch_config: "OrchestratorConfig") -> tuple[bool, str]:
    """Refuse multi-run orchestrator configs that didn't explicitly set `[orchestrator.optim] lr`.

    Per-run optimizer LR is consumed by the multi-scheduler; its default
    `1e-4` is 200x ScaleRL §3.3's `5e-7`. A copy-pasted multi-run TOML that
    sets `[trainer.optim] lr = 5e-7` but forgets to mirror that on the
    per-run orchestrator config silently trains every adapter at 1e-4. The
    multi-run path discovers orchestrator configs out-of-process so a
    single RLConfig-level validator can't see them — register this hook
    inside setup_multi_run_manager so each per-run config is rejected at
    discovery time when `lr` was inherited from the schema default rather
    than explicitly written.
    """
    if "lr" not in orch_config.optim.model_fields_set:
        return (
            False,
            "[orchestrator.optim] lr is required in multi-run mode (it is "
            "consumed by the per-run multi-scheduler). The schema default "
            "1e-4 is 200x ScaleRL §3.3's 5e-7 and silently mistraining a "
            "multi-run sweep is a real footgun. Set `[orchestrator.optim] "
            "lr = <value>` explicitly for every per-run TOML."
        )
    return True, ""


def setup_multi_run_manager(
    output_dir: Path, max_runs: int, device: torch.device, lora_config: LoRAConfig | None = None
) -> MultiRunManager:
    """Initialize the MultiRunManager singleton.

    Args:
        output_dir: Directory containing run outputs
        max_runs: Maximum number of concurrent runs
        device: Device for LoRA tensors
        lora_config: Optional trainer LoRA config. If provided, registers validation
            and scaling hooks for LoRA rank and alpha.

    Returns:
        The initialized MultiRunManager instance.
    """
    global _MULTI_RUN_MANAGER
    _MULTI_RUN_MANAGER = MultiRunManager(output_dir, max_runs, device)

    # Register validation and scaling hooks for LoRA
    if lora_config is not None and _MULTI_RUN_MANAGER.world.is_master:
        trainer_lora = lora_config

        def validate_lora_rank(orch_config: "OrchestratorConfig") -> tuple[bool, str]:
            return _validate_orch_lora_against_trainer(orch_config, trainer_lora)

        def on_run_discovered(idx: int, run_id: str, orch_config: "OrchestratorConfig") -> None:
            # Validation hook above guarantees orch_config.model.lora is not None
            # before this discovered hook fires; keep an explicit assert so a
            # future refactor that reorders or skips validation surfaces the
            # contract loudly rather than crashing on attribute access.
            assert orch_config.model.lora is not None, (
                "on_run_discovered requires validate_lora_rank to have run first"
            )
            _MULTI_RUN_MANAGER.scaling_factors[idx] = orch_config.model.lora.alpha / orch_config.model.lora.rank

        _MULTI_RUN_MANAGER.register_config_validation_hook(validate_lora_rank)
        _MULTI_RUN_MANAGER.register_discovered_hook(on_run_discovered)

    # Always register the per-run LR explicit-set check in multi-run mode.
    # max_runs == 1 is the single-run path where orchestrator.optim.lr is
    # dead — only multi-run mode consumes it (multi-scheduler at scheduler.py
    # line ~149, multi-optimizer at optim.py line ~294).
    if max_runs > 1 and _MULTI_RUN_MANAGER.world.is_master:
        _MULTI_RUN_MANAGER.register_config_validation_hook(_validate_orch_optim_lr_explicit)

    return _MULTI_RUN_MANAGER
