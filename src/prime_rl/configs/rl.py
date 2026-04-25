from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.inference import WeightBroadcastConfig as InferenceWeightBroadcastConfig
from prime_rl.configs.orchestrator import (
    CheckpointConfig as OrchestratorCheckpointConfig,
)
from prime_rl.configs.orchestrator import (
    FileSystemWeightBroadcastConfig as OrchestratorFileSystemWeightBroadcastConfig,
)
from prime_rl.configs.orchestrator import (
    NCCLWeightBroadcastConfig as OrchestratorNCCLWeightBroadcastConfig,
)
from prime_rl.configs.orchestrator import (
    OrchestratorConfig,
)
from prime_rl.configs.shared import (
    SlurmConfig,
    VLMConfig,
    WandbConfig,
    WandbWithExtrasConfig,
)
from prime_rl.configs.trainer import (
    BenchConfig,
    FakeDataLoaderConfig,
    TokenizerConfig,
    TrainerConfig,
)
from prime_rl.configs.trainer import (
    CheckpointConfig as TrainerCheckpointConfig,
)
from prime_rl.configs.trainer import (
    FileSystemWeightBroadcastConfig as TrainerFileSystemWeightBroadcastConfig,
)
from prime_rl.configs.trainer import (
    NCCLWeightBroadcastConfig as TrainerNCCLWeightBroadcastConfig,
)
from prime_rl.utils.config import BaseConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.validation import (
    validate_shared_ckpt_config,
    validate_shared_max_async_level,
    validate_shared_max_steps,
    validate_shared_model_name,
    validate_shared_output_dir,
    validate_shared_tokenizer,
    validate_shared_wandb_config,
    validate_shared_weight_broadcast,
)


class RLExperimentalConfig(BaseConfig):
    """Experimental features for RL training."""


class SharedLogConfig(BaseConfig):
    """Configures shared logging."""

    level: Annotated[
        str | None,
        Field(
            description="The log level to use. When unset, the trainer and orchestrator log levels are used as-is (which themselves default to the PRIME_LOG_LEVEL env var if set, else 'info').",
        ),
    ] = None

    json_logging: Annotated[
        bool,
        Field(description="Emit JSON logs (newline-delimited) for log aggregation (Loki, Grafana, etc.)."),
    ] = False


class SharedWandbConfig(BaseConfig):
    """Configures shared W&B configs."""

    project: Annotated[str | None, Field(description="The W&B project to use.")] = "prime-rl"

    name: Annotated[str | None, Field(description="The W&B run name to use.")] = None

    offline: Annotated[bool | None, Field(description="Whether to run W&B in offline mode.")] = False

    shared: Annotated[
        bool,
        Field(
            description="Use shared W&B mode to log trainer and orchestrator metrics to a single run. "
            "Requires wandb SDK >= 0.19.9. Incompatible with offline mode.",
        ),
    ] = True

    @model_validator(mode="after")
    def validate_shared_not_offline(self):
        if self.shared and self.offline:
            raise ValueError("W&B shared mode requires server connectivity and is incompatible with offline mode")
        return self


class SharedCheckpointConfig(BaseConfig):
    """Configures shared checkpoint configs."""

    output_dir: Annotated[
        Path | None,
        Field(
            description="Override directory for checkpoints and weights. When set, checkpoints and weight snapshots are written here instead of under the trainer output_dir.",
        ),
    ] = None

    interval: Annotated[int | None, Field(description="The interval at which to save checkpoints.")] = None

    resume_step: Annotated[
        int | None, Field(description="The step to resume from. If None, will not resume from a checkpoint.")
    ] = None

    keep_last: Annotated[
        int | None,
        Field(
            ge=1,
            description="Keep at most this many recent step checkpoints on disk. If None, never clean old checkpoints based on recency.",
        ),
    ] = None

    keep_interval: Annotated[
        int | None,
        Field(
            ge=1,
            description="Keep checkpoints at every N steps permanently (e.g., keep_interval=100 keeps step 100, 200, ...). If None, no interval-based keeping.",
        ),
    ] = None


class SharedModelConfig(BaseConfig):
    """Configures shared model settings."""

    name: Annotated[
        str,
        Field(description="The name of the model to use."),
    ] = "Qwen/Qwen3-0.6B"

    vlm: Annotated[
        "VLMConfig | None",
        Field(description="VLM configuration. Set to enable vision-language model support."),
    ] = None


class SharedWeightBroadcastConfig(BaseConfig):
    """Configures shared weight broadcast settings."""

    type: Annotated[Literal["nccl", "filesystem"], Field(description="The type of weight broadcast to use.")] = (
        "filesystem"
    )

    port: Annotated[int, Field(description="The port to use for NCCL weight broadcast.")] = 29501
    timeout: Annotated[int, Field(description="The timeout in seconds for NCCL weight broadcast.")] = 1200
    quantize_in_weight_transfer: Annotated[
        bool,
        Field(
            description=(
                "Use kernel-format FP8 quantized NCCL transfer for weight updates. "
                "When disabled, uses default HF checkpoint-format transfer."
            ),
        ),
    ] = False


class BaseDeploymentConfig(BaseModel):
    """Configures a base deployment."""

    model_config = ConfigDict(extra="forbid")

    gpus_per_node: Annotated[int, Field(description="Number of GPUs per node.")] = 8


class SingleNodeDeploymentConfig(BaseDeploymentConfig):
    """Configures a single node deployment."""

    type: Literal["single_node"] = "single_node"

    num_train_gpus: Annotated[int, Field(description="Number of training GPUs")] = 1
    num_infer_gpus: Annotated[int, Field(description="Number of inference GPUs")] = 1
    num_teacher_gpus: Annotated[int | None, Field(description="Number of teacher inference GPUs")] = None

    @model_validator(mode="after")
    def validate_gpu_count(self):
        total = self.num_train_gpus + self.num_infer_gpus + (self.num_teacher_gpus or 0)
        if total > self.gpus_per_node:
            raise ValueError(
                f"Total GPU count ({total} = {self.num_train_gpus} train + {self.num_infer_gpus} infer"
                f" + {self.num_teacher_gpus or 0} teacher) exceeds gpus_per_node ({self.gpus_per_node})."
            )
        return self


class MultiNodeDeploymentConfig(BaseDeploymentConfig):
    """Configures a multi node deployment."""

    type: Literal["multi_node"] = "multi_node"

    num_train_nodes: Annotated[int, Field(description="Number of training nodes.")]
    num_infer_nodes: Annotated[
        int,
        Field(
            ge=0,
            description="Number of inference nodes per replica. Set to 0 to skip inference and orchestrator (requires fake data).",
        ),
    ]
    num_infer_replicas: Annotated[
        int,
        Field(
            ge=1,
            description="Number of independent inference replicas. Total inference nodes = num_infer_nodes * num_infer_replicas.",
        ),
    ] = 1
    num_teacher_nodes: Annotated[int | None, Field(description="Number of teacher inference nodes.")] = None

    nodes_per_fsdp_group: Annotated[
        int | None,
        Field(
            description="Number of training nodes per FSDP island. Auto-sets trainer.dp_replicate = num_train_nodes / nodes_per_fsdp_group."
        ),
    ] = None

    @property
    def total_infer_nodes(self) -> int:
        return self.num_infer_nodes * self.num_infer_replicas

    @model_validator(mode="after")
    def teacher_inference_not_supported(self):
        if self.num_teacher_nodes is not None:
            raise ValueError("Teacher inference is not yet supported in multi node deployment.")
        return self


DeploymentConfig: TypeAlias = Annotated[
    SingleNodeDeploymentConfig | MultiNodeDeploymentConfig, Field(discriminator="type")
]


class RLConfig(BaseConfig):
    """Configures an RL training run."""

    trainer: TrainerConfig
    orchestrator: OrchestratorConfig
    inference: Annotated[
        InferenceConfig | None,
        Field(
            description="The inference config. If None, the rl entrypoint will not start an inference server (useful for elastic inference pools or manually started servers)."
        ),
    ] = None

    teacher_inference: Annotated[
        InferenceConfig | None,
        Field(
            description="Teacher inference config. If None, will use the same config as inference or a default config. Only used when teacher GPUs or nodes are set."
        ),
    ] = None

    output_dir: Annotated[
        Path,
        Field(
            description="The directory to store the outputs. Should be set to a unique directory identifying the experiment."
        ),
    ] = Path("outputs")

    clean_output_dir: Annotated[
        bool,
        Field(
            description="If true, delete the output directory before starting training. Required to overwrite an output directory that contains checkpoints from a previous run when not resuming.",
        ),
    ] = False

    ### Shared configurations

    log: Annotated[
        SharedLogConfig,
        Field(
            description="Shared log configs. If None, will fallback to the log configs specified on submodule configs."
        ),
    ] = SharedLogConfig()

    ckpt: Annotated[
        SharedCheckpointConfig | None,
        Field(
            description="Shared checkpoint configs. If None, will fallback to the checkpoint configs specified on submodule configs."
        ),
    ] = None

    wandb: Annotated[
        SharedWandbConfig | None,
        Field(
            description="Shared W&B configs. If None, will fallback to the W&B configs specified on submodule configs."
        ),
    ] = None

    model: Annotated[
        SharedModelConfig | None,
        Field(
            description="Shared model configs. If None, will fallback to the model configs specified on submodule configs."
        ),
    ] = None

    tokenizer: Annotated[
        TokenizerConfig | None,
        Field(
            description="Shared tokenizer config. Propagated to trainer, orchestrator, and inference. "
            "If None, each component uses its own tokenizer config (defaulting to model name).",
        ),
    ] = None

    max_steps: Annotated[
        int | None,
        Field(
            description="The maximum number of steps to train for. If None, will fallback to the max steps specified on submodule configs."
        ),
    ] = None

    max_model_len: Annotated[
        int | None,
        Field(
            description="The maximum model length to use. If None, will fallback to the max model length specified on submodule configs."
        ),
    ] = None

    seq_len: Annotated[
        int | None,
        Field(
            description="Shared sequence length. Propagates to trainer.model.seq_len and orchestrator.seq_len, "
            "but only for those not explicitly set in the config. "
            "Explicitly set per-component values always take precedence."
        ),
    ] = None

    max_async_level: Annotated[
        int | None,
        Field(
            description="The async level to use. If None, will fallback to the async level specified on submodule configs."
        ),
    ] = None

    weight_broadcast: Annotated[
        SharedWeightBroadcastConfig | None, Field(description="The weight broadcast config.")
    ] = None

    bench: Annotated[
        bool,
        Field(
            description="Whether to run in benchmark mode. Automatically sets the trainer and orchestrator to benchmark mode and, if present, suffixes the W&B project with `-bench`.",
        ),
    ] = False

    deployment: DeploymentConfig = SingleNodeDeploymentConfig()

    slurm: Annotated[SlurmConfig | None, Field(description="SLURM configuration. If None, will run locally.")] = None

    dry_run: Annotated[bool, Field(description="Only validate and dump resolved configs and exit early.")] = False

    experimental: Annotated[
        RLExperimentalConfig,
        Field(description="Experimental features for RL training."),
    ] = RLExperimentalConfig()

    ### Validate configs (e.g. raise for unsupported (combinations of) configs)

    @model_validator(mode="after")
    def validate_deployment(self):
        if self.deployment.type == "multi_node":
            if self.slurm is None:
                raise ValueError("Must use SLURM for multi-node deployment.")
            if self.deployment.num_infer_nodes > 0 and not self.inference:
                raise ValueError("Must configure inference when using multi-node deployment with inference nodes.")
            if self.deployment.num_infer_nodes == 0 and self.inference:
                raise ValueError(
                    "Cannot configure inference with num_infer_nodes = 0. "
                    "Either set num_infer_nodes > 0 or remove the inference config."
                )
            if self.deployment.num_infer_nodes == 0 and not self.trainer.data.fake and not self.bench:
                raise ValueError(
                    "Must use fake data (trainer.data.fake or bench = true) when num_infer_nodes = 0, "
                    "since no orchestrator or inference server will be running."
                )
        return self

    # TODO: fix this
    @model_validator(mode="after")
    def validate_no_teacher_in_multinode(self):
        if self.deployment.type == "multi_node" and self.teacher_inference is not None:
            raise ValueError(
                "Teacher inference is not supported in multi-node deployment. "
                "The SLURM template only handles inference and training nodes."
            )
        return self

    @model_validator(mode="after")
    def validate_enough_devices_for_nccl(self):
        if self.deployment.type == "single_node":
            if self.trainer.weight_broadcast.type == "nccl":
                if self.deployment.num_train_gpus + self.deployment.num_infer_gpus < 2:
                    raise ValueError(
                        "NCCL weight broadcast requires at least 2 GPUs to build the broadcast process group."
                    )
        return self

    @model_validator(mode="after")
    def validate_quantize_in_weight_transfer(self):
        if self.weight_broadcast is None or not self.weight_broadcast.quantize_in_weight_transfer:
            return self

        if self.weight_broadcast.type != "nccl":
            raise ValueError("weight_broadcast.quantize_in_weight_transfer requires weight_broadcast.type = 'nccl'.")

        if self.inference is None:
            raise ValueError("weight_broadcast.quantize_in_weight_transfer requires an inference config.")

        if self.trainer.model.impl != "custom":
            raise ValueError("weight_broadcast.quantize_in_weight_transfer requires trainer.model.impl = 'custom'.")

        return self

    @model_validator(mode="after")
    def validate_teacher_model(self):
        if (
            self.trainer.loss.type == "default" and self.trainer.loss.teacher_tau > 0
        ) and not self.orchestrator.teacher_model:
            raise ValueError(
                "teacher_model must be configured when teacher_tau > 0. "
                "Either set teacher_tau = 0, set deployment.num_teacher_gpus, or configure teacher_model manually."
            )
        return self

    @model_validator(mode="after")
    def validate_cispo_no_teacher(self):
        """CISPO ignores `teacher_logprobs` — `cispo_loss_fn` does not consume
        them. Configuring a teacher_model with `loss.type == "cispo"` would
        silently pay the teacher-prefill compute cost on every example, ship
        the tensors through TrainingSample, and have the trainer drop them.
        Reject the combination loudly.
        """
        if self.trainer.loss.type == "cispo" and self.orchestrator.teacher_model is not None:
            raise ValueError(
                "trainer.loss.type='cispo' with [orchestrator.teacher_model] set: "
                "CISPO does not consume teacher logprobs (no teacher_tau / KL term). "
                "The teacher prefill cost would be paid for nothing. Either drop "
                "[orchestrator.teacher_model] or switch to trainer.loss.type='default' "
                "with teacher_tau > 0."
            )
        return self

    @model_validator(mode="after")
    def validate_sft_no_teacher(self):
        """Same footgun as CISPO+teacher: `sft_loss_fn` does not consume
        teacher_logprobs. A `loss.type == "sft"` config with a teacher_model
        configured would silently pay the prefill compute cost.
        """
        if self.trainer.loss.type == "sft" and self.orchestrator.teacher_model is not None:
            raise ValueError(
                "trainer.loss.type='sft' with [orchestrator.teacher_model] set: "
                "SFT does not consume teacher logprobs. Drop [orchestrator.teacher_model] "
                "or switch to trainer.loss.type='default' with teacher_tau > 0 if you want "
                "teacher distillation."
            )
        return self

    # Note: a previously-added `validate_is_ratio_loss_with_external_rollout_string_client`
    # was redundant — `validate_external_rollout_mode` (below) already rejects
    # `teacher_rollout_model` with any non-SFT loss. The remaining gap is when an
    # operator overrides `[orchestrator.client] base_url` to point at an external
    # service WITHOUT setting `teacher_rollout_model` — neither validator catches
    # that, and `pretokenize_rollout_trajectory` reconstructs zero logprobs that
    # would silently corrupt CISPO/Default. The runtime check in
    # `prime_rl.trainer.rl.train` raises before compute_loss when a micro-batch
    # carries `inference_logprobs_synthesized=True` and the loss is CISPO/Default —
    # that's the load-bearing guard for the bypass path.

    @model_validator(mode="after")
    def validate_fp32_lm_head_matmul_precision(self):
        """fp32_lm_head=True is silently downgraded to TF32 unless
        matmul_precision="highest".

        PyTorch's `torch.set_float32_matmul_precision("high")` (the
        TrainerConfig default) relaxes FP32 matmuls to TF32 (10-bit
        mantissa) on NVIDIA Ampere+. The LM-head matmul in
        `trainer.models.layers.lm_head` is a plain `F.linear(hidden.float(),
        weight.float())`, so it honors that global flag — defeating the
        very §3.2 / MiniMax-M1 §3.2 train/inference logprob parity that
        fp32_lm_head exists to enforce. The shipped scalerl_*/rl.toml
        configs already override to "highest", but a child config that
        copy-pastes only `fp32_lm_head=true` would silently get TF32.
        Reject the combination at config load.
        """
        if not self.trainer.model.fp32_lm_head:
            return self
        if self.trainer.matmul_precision != "highest":
            raise ValueError(
                "trainer.model.fp32_lm_head=True requires "
                f"trainer.matmul_precision='highest' (got "
                f"{self.trainer.matmul_precision!r}). 'high'/'medium' "
                "silently relaxes FP32 matmuls to TF32 (10-bit mantissa) "
                "on NVIDIA Ampere+, which defeats the §3.2 "
                "train/inference logprob-parity ingredient that "
                "fp32_lm_head exists for. Set [trainer] matmul_precision "
                "= \"highest\" or disable fp32_lm_head."
            )
        return self

    @model_validator(mode="after")
    def validate_fp32_lm_head_no_fp8_transfer(self):
        """fp32_lm_head + quantize_in_weight_transfer=true silently undoes
        the §3.2 ingredient.

        With NCCL FP8 quantized weight transfer, the trainer broadcasts FP8
        LM-head weights and `inference/patches.py:promote_parallel_lm_head_to_fp32`
        casts the FP8 result to fp32 — baking in the quantization noise the
        fp32 LM-head was supposed to prevent. Reject the combination.
        """
        if not self.trainer.model.fp32_lm_head:
            return self
        # Check both the top-level `[weight_broadcast]` (the user's TOML
        # source-of-truth) and the post-propagation `trainer.weight_broadcast`
        # — `auto_setup_weight_broadcast` copies the RLConfig-level value into
        # both trainer and orchestrator, but the top-level field is what the
        # user actually sets.
        for wb in (
            getattr(self, "weight_broadcast", None),
            getattr(self.trainer, "weight_broadcast", None),
        ):
            if wb is None:
                continue
            if getattr(wb, "type", None) == "nccl" and getattr(wb, "quantize_in_weight_transfer", False):
                raise ValueError(
                    "trainer.model.fp32_lm_head=True with "
                    "weight_broadcast.quantize_in_weight_transfer=True is "
                    "self-defeating: NCCL FP8 weight transfer would quantize the "
                    "LM-head weights to FP8 before vLLM's promote-to-fp32 path runs, "
                    "baking in the quantization noise that fp32 LM-head exists to "
                    "prevent (ScaleRL §3.2 / MiniMax-M1 §3.2). Either disable "
                    "fp32_lm_head or set quantize_in_weight_transfer=False."
                )
        return self

    @model_validator(mode="after")
    def validate_external_rollout_mode(self):
        if self.orchestrator.teacher_rollout_model is None:
            return self

        if self.trainer.loss.type != "sft":
            raise ValueError('orchestrator.teacher_rollout_model is only supported when trainer.loss.type = "sft".')

        if self.inference is not None:
            raise ValueError(
                "inference must be omitted when orchestrator.teacher_rollout_model is configured. "
                "External rollout mode does not use the local inference server."
            )

        if self.orchestrator.use_token_client:
            raise ValueError(
                "orchestrator.use_token_client must be false when orchestrator.teacher_rollout_model is configured."
            )

        return self

    ### Auto-setup and validate shared configs

    @model_validator(mode="after")
    def auto_setup_output_dir(self):
        """Auto-setup shared output directory for trainer and orchestrator."""
        self.trainer.output_dir = self.output_dir
        self.orchestrator.output_dir = self.output_dir / "run_default"

        validate_shared_output_dir(self.trainer, self.orchestrator)

        return self

    @model_validator(mode="after")
    def auto_setup_logs(self):
        """Auto-setup shared log config for trainer and orchestrator."""
        if self.log is not None:
            if self.log.level is not None:
                self.trainer.log.level = self.log.level
                self.orchestrator.log.level = self.log.level
            self.trainer.log.json_logging = self.log.json_logging
            self.orchestrator.log.json_logging = self.log.json_logging

        return self

    @model_validator(mode="after")
    def auto_setup_ckpt(self):
        """Auto-setup shared checkpoint config for trainer and orchestrator."""
        if self.ckpt is not None:
            # Create checkpoint configs if not specified
            if self.trainer.ckpt is None:
                self.trainer.ckpt = TrainerCheckpointConfig()
            if self.orchestrator.ckpt is None:
                self.orchestrator.ckpt = OrchestratorCheckpointConfig()

            # If specified, override checkpoint output directory
            if self.ckpt.output_dir is not None:
                self.trainer.ckpt.output_dir = self.ckpt.output_dir

            # If specified, use the same ckpt interval
            if self.ckpt.interval is not None:
                self.trainer.ckpt.interval = self.ckpt.interval
                self.orchestrator.ckpt.interval = self.ckpt.interval

            # If resuming training, ensure orchestrator resume from the same step
            if self.ckpt.resume_step is not None:
                self.trainer.ckpt.resume_step = self.ckpt.resume_step
                self.orchestrator.ckpt.resume_step = self.ckpt.resume_step

            # If specified, propagate keep policy
            if self.ckpt.keep_last is not None:
                self.trainer.ckpt.keep_last = self.ckpt.keep_last
                self.orchestrator.ckpt.keep_last = self.ckpt.keep_last

            if self.ckpt.keep_interval is not None:
                self.trainer.ckpt.keep_interval = self.ckpt.keep_interval
                self.orchestrator.ckpt.keep_interval = self.ckpt.keep_interval

        validate_shared_ckpt_config(self.trainer, self.orchestrator)

        return self

    @model_validator(mode="after")
    def auto_setup_wandb(self):
        """Auto-setup shared W&B config for trainer and orchestrator."""
        if self.wandb is not None:
            if not self.trainer.wandb:
                self.trainer.wandb = WandbConfig()
            if not self.orchestrator.wandb:
                self.orchestrator.wandb = WandbWithExtrasConfig()

            if self.wandb.project:
                self.trainer.wandb.project = self.wandb.project
                self.orchestrator.wandb.project = self.wandb.project

            if self.wandb.shared:
                if self.wandb.name:
                    self.trainer.wandb.name = self.wandb.name
                    self.orchestrator.wandb.name = self.wandb.name
            else:
                if self.wandb.name:
                    self.trainer.wandb.name = f"{self.wandb.name}-trainer"
                    self.orchestrator.wandb.name = f"{self.wandb.name}-orchestrator"

            if self.wandb.offline:
                self.trainer.wandb.offline = self.wandb.offline
                self.orchestrator.wandb.offline = self.wandb.offline

        validate_shared_wandb_config(self.trainer, self.orchestrator)

        if self.orchestrator.prime_monitor is not None and self.orchestrator.prime_monitor.run_name is None:
            if self.wandb and self.wandb.name:
                self.orchestrator.prime_monitor.run_name = self.wandb.name

        return self

    @model_validator(mode="after")
    def auto_setup_model(self):
        """Auto-setup shared model config for trainer, orchestrator, and inference."""
        if self.model is not None:
            self.trainer.model.name = self.model.name
            if self.inference is not None:
                inference_model_explicitly_set = "name" in self.inference.model.model_fields_set
                if not inference_model_explicitly_set:
                    self.inference.model.name = self.model.name
                self.orchestrator.model.name = self.inference.model.name
            else:
                self.orchestrator.model.name = self.model.name

            if self.model.vlm is not None:
                self.trainer.model.vlm = self.model.vlm
                self.orchestrator.model.vlm = self.model.vlm
                if self.inference is not None:
                    self.inference.model.vlm = self.model.vlm

        validate_shared_model_name(self.trainer, self.orchestrator, self.inference)

        return self

    @model_validator(mode="after")
    def auto_setup_tokenizer(self):
        """Auto-setup shared tokenizer config for trainer, orchestrator, and inference."""
        if self.tokenizer is not None:
            # Shared tokenizer config: propagate to all components, then fill
            # in name/trust_remote_code from model config where still unset.
            self.trainer.tokenizer = self.tokenizer.model_copy()
            self.orchestrator.tokenizer = self.tokenizer.model_copy()
            for component in (self.trainer, self.orchestrator):
                if component.tokenizer.name is None:
                    component.tokenizer.name = component.model.name
                if component.tokenizer.trust_remote_code is None:
                    component.tokenizer.trust_remote_code = component.model.trust_remote_code
        else:
            # No shared tokenizer: re-derive from (now-correct) model names,
            # since auto_setup_tokenizer on sub-configs already ran with defaults.
            for component in (self.trainer, self.orchestrator):
                component.tokenizer.name = component.model.name
                component.tokenizer.trust_remote_code = component.model.trust_remote_code

        # Propagate chat_template to inference (vLLM --chat-template)
        if self.inference is not None:
            chat_template = self.trainer.tokenizer.chat_template
            if chat_template is not None and self.inference.model.chat_template is None:
                self.inference.model.chat_template = chat_template

        validate_shared_tokenizer(self.trainer, self.orchestrator, self.inference)

        return self

    @model_validator(mode="after")
    def auto_setup_max_steps(self):
        """Auto-setup shared max steps for trainer and orchestrator."""
        if self.max_steps is not None:
            self.trainer.max_steps = self.max_steps
            self.orchestrator.max_steps = self.max_steps

        validate_shared_max_steps(self.trainer, self.orchestrator)

        return self

    @model_validator(mode="after")
    def auto_setup_async_level(self):
        """Auto-setup shared async level for trainer and orchestrator."""
        if self.max_async_level is not None:
            self.trainer.max_async_level = self.max_async_level
            self.orchestrator.max_async_level = self.max_async_level

        validate_shared_max_async_level(self.trainer, self.orchestrator)

        return self

    @model_validator(mode="after")
    def propagate_nccl_async_override(self):
        """If one side opts into the experimental NCCL override, propagate to the other.

        RLConfig owns max_async_level for both subconfigs, so the experimental opt-in must be
        consistent across them. Setting it on either side propagates; this avoids the "fix
        the orchestrator validator but the trainer validator still rejects" footgun.

        Honor explicit `false` on both sides: a user who set both to false on
        purpose to assert "definitely no override" should not have their value
        silently flipped. Only propagate when AT MOST one side was explicitly
        set; reject conflicting explicit values.
        """
        orch_set = "allow_nccl_async_level_override" in self.orchestrator.experimental.model_fields_set
        trainer_set = "allow_nccl_async_level_override" in self.trainer.experimental.model_fields_set
        orch_flag = self.orchestrator.experimental.allow_nccl_async_level_override
        trainer_flag = self.trainer.experimental.allow_nccl_async_level_override
        if orch_set and trainer_set and orch_flag != trainer_flag:
            raise ValueError(
                "[orchestrator.experimental] allow_nccl_async_level_override "
                f"({orch_flag}) and [trainer.experimental] allow_nccl_async_level_override "
                f"({trainer_flag}) conflict. Set both to the same value, or set only one and "
                "let RLConfig propagate."
            )
        flag = orch_flag if orch_set else trainer_flag if trainer_set else orch_flag  # default False
        self.orchestrator.experimental.allow_nccl_async_level_override = flag
        self.trainer.experimental.allow_nccl_async_level_override = flag
        return self

    @model_validator(mode="after")
    def validate_fp32_lm_head_consistency(self):
        """Reject mismatched fp32_lm_head between trainer and inference.

        The whole point of FP32 LM-head (ScaleRL §3.2) is to keep train/inference logprob
        correlation > 0.99. Setting it on one side without the other defeats the purpose
        AND injects silent bias into any IS-based loss. Fail loud at config load.

        When [inference] is omitted (externally managed inference pools or
        num_infer_nodes=0 fake-data runs) we cannot validate the inference side.
        If the trainer wants fp32 LM-head, warn loudly: `setup_vllm_env` does not
        run on an externally launched vLLM, so the operator must export
        `PRIME_RL_VLLM_FP32_LM_HEAD=1` themselves before starting the server,
        or vLLM will silently run the LM-head matmul in bf16 and break IS parity.
        """
        if self.inference is None:
            if self.trainer.model.fp32_lm_head:
                import warnings

                warnings.warn(
                    "trainer.model.fp32_lm_head=True with [inference] omitted: "
                    "the FP32 LM-head env-var bridge in prime_rl.inference.server "
                    "does NOT run on externally launched vLLM. Export "
                    "PRIME_RL_VLLM_FP32_LM_HEAD=1 in the vLLM environment yourself, "
                    "or train/inference logprob parity will break (ScaleRL §3.2).",
                    stacklevel=2,
                )
            return self
        trainer_fp32 = self.trainer.model.fp32_lm_head
        inference_fp32 = self.inference.model.fp32_lm_head
        if trainer_fp32 != inference_fp32:
            raise ValueError(
                f"fp32_lm_head must match between trainer and inference: "
                f"trainer.model.fp32_lm_head={trainer_fp32}, "
                f"inference.model.fp32_lm_head={inference_fp32}. "
                "Mismatch defeats the train/inference logprob parity that FP32 LM-head exists "
                "to provide (ScaleRL §3.2). Set both true or both false."
            )
        return self

    @model_validator(mode="after")
    def validate_prompt_average_loss_scale_mode(self):
        """Cross-validate `orchestrator.prompt_average_loss` and trainer
        `loss_scale_mode` so neither setting silently produces a different
        objective than the user expected.

        Forward direction:
            prompt_average_loss=True with loss_scale_mode='token' silently
            drops the packer's per-sequence weights (compute_loss only
            consumes them in 'sequence'/'none' mode) and trains under
            token-mean reduction.

        Reverse direction:
            loss_scale_mode in {'sequence','none'} without
            prompt_average_loss=True leaves the per-sequence weights at the
            neutral 1.0 default. compute_loss then computes
            `sum_i loss_i * fsdp_gradient_divide_factor`, which after FSDP averaging is
            the raw (unnormalized) sum over the global batch — effective LR
            scales linearly with batch_size * avg_completion_len. CISPO
            defaults to 'sequence', so an out-of-the-box CISPO config without
            prompt_average_loss=True silently lacks any normalization.

        CISPO + prompt_average_loss=True + loss_scale_mode='sequence' (the
        smoke config) validates and is the recipe ScaleRL §3.3 prescribes.
        """
        scale_mode = self.trainer.loss.loss_scale_mode
        if self.orchestrator.prompt_average_loss and scale_mode not in ("sequence", "none"):
            raise ValueError(
                "orchestrator.prompt_average_loss=true requires "
                "trainer.loss.loss_scale_mode in {'sequence', 'none'} so the "
                "packer's per-sequence weights actually reach compute_loss. "
                f"Got loss_scale_mode={scale_mode!r}. Either set "
                "[trainer.loss] loss_scale_mode = \"sequence\" "
                "(the CISPO default), or disable prompt_average_loss."
            )
        if scale_mode in ("sequence", "none") and not self.orchestrator.prompt_average_loss:
            raise ValueError(
                f"trainer.loss.loss_scale_mode={scale_mode!r} relies on the "
                "packer to encode per-sequence normalization in "
                "`sequence_loss_weights`. With `[orchestrator] "
                "prompt_average_loss = false` the weights stay at the neutral "
                "1.0 default and the loss reduces to a raw global sum scaled "
                "by dp_world_size — effective LR scales with batch size and "
                "DP size, almost certainly not what you want. Either set "
                "[orchestrator] prompt_average_loss = true (the ScaleRL "
                "default), or set [trainer.loss] loss_scale_mode = \"token\" "
                "(token-mean reduction). CustomLossConfig that supplies its "
                "own normalization is no exception — the trainer cannot tell "
                "whether your custom loss does — so make the choice explicit."
            )
        return self

    @model_validator(mode="after")
    def auto_setup_seq_len(self):
        """Auto-setup shared seq_len for trainer and orchestrator.

        Only propagates to components that weren't explicitly set in the config.
        Uses model_fields_set to detect explicit assignment.
        """
        if self.seq_len is not None:
            if "seq_len" not in self.trainer.model.model_fields_set:
                self.trainer.model.seq_len = self.seq_len
            if "seq_len" not in self.orchestrator.model_fields_set:
                self.orchestrator.seq_len = self.seq_len

        if self.trainer.model.seq_len < self.orchestrator.seq_len:
            raise ValueError(
                f"Trainer model seq_len ({self.trainer.model.seq_len}) must be >= orchestrator seq_len ({self.orchestrator.seq_len}). "
                f"The trainer needs to be able to handle sequences at least as long as those produced by the orchestrator."
            )

        return self

    @model_validator(mode="after")
    def auto_setup_weight_broadcast(self):
        """Auto-setup shared weight broadcast config for trainer, orchestrator, and inference."""
        if self.weight_broadcast is not None:
            if self.weight_broadcast.type == "nccl":
                inference_world_size = self.inference.parallel.dp * self.inference.parallel.tp if self.inference else 1
                self.trainer.weight_broadcast = TrainerNCCLWeightBroadcastConfig(
                    type=self.weight_broadcast.type,
                    inference_world_size=inference_world_size,
                    port=self.weight_broadcast.port,
                    timeout=self.weight_broadcast.timeout,
                    quantize_in_weight_transfer=self.weight_broadcast.quantize_in_weight_transfer,
                )
                self.orchestrator.weight_broadcast = OrchestratorNCCLWeightBroadcastConfig(
                    type=self.weight_broadcast.type,
                    port=self.weight_broadcast.port,
                    timeout=self.weight_broadcast.timeout,
                    inference_world_size=inference_world_size,
                    quantize_in_weight_transfer=self.weight_broadcast.quantize_in_weight_transfer,
                )
            elif self.weight_broadcast.type == "filesystem":
                self.trainer.weight_broadcast = TrainerFileSystemWeightBroadcastConfig()
                self.orchestrator.weight_broadcast = OrchestratorFileSystemWeightBroadcastConfig()
            if self.inference is not None:
                self.inference.weight_broadcast = InferenceWeightBroadcastConfig(type=self.weight_broadcast.type)

        validate_shared_weight_broadcast(self.trainer, self.orchestrator, self.inference)

        return self

    @model_validator(mode="after")
    def validate_root_nccl_async_override(self):
        """Re-check the NCCL + max_async_level > 1 guardrail at the root level.

        The trainer and orchestrator subconfigs each enforce this on their own, but
        `auto_setup_weight_broadcast` reassigns their `weight_broadcast` after their
        validators have already run. A root config with `[weight_broadcast].type = "nccl"`,
        `max_async_level = 8`, and no override on either subconfig would otherwise sneak
        through.
        """
        broadcast = self.trainer.weight_broadcast
        if broadcast is None or broadcast.type != "nccl":
            return self
        async_level = max(self.trainer.max_async_level, self.orchestrator.max_async_level)
        if async_level <= 1:
            return self
        override = (
            self.orchestrator.experimental.allow_nccl_async_level_override
            or self.trainer.experimental.allow_nccl_async_level_override
        )
        if not override:
            raise ValueError(
                "NCCL weight broadcast with max_async_level > 1 is rejected by default. "
                "ScaleRL Pipeline-RL §3.1 requires it; set the experimental override on "
                "either side — they propagate to both — using EITHER\n"
                "    [orchestrator.experimental]\n"
                "    allow_nccl_async_level_override = true\n"
                "OR\n"
                "    [trainer.experimental]\n"
                "    allow_nccl_async_level_override = true\n"
                "Validated on B200/EFA. For other hardware prefer "
                "`[weight_broadcast] type = \"filesystem\"`."
            )
        return self

    @model_validator(mode="after")
    def auto_setup_bench(self):
        if self.bench:
            self.trainer.bench = BenchConfig()
            self.orchestrator.bench = True
            self.trainer.data.fake = FakeDataLoaderConfig(
                batch_size=self.orchestrator.batch_size or 32,
            )

        trainer_bench_enabled = self.trainer.bench is not None
        if trainer_bench_enabled != self.orchestrator.bench:
            raise ValueError(
                f"Trainer benchmark mode ({self.trainer.bench}) and orchestrator benchmark mode "
                f"({self.orchestrator.bench}) must match. Use the top-level bench = true to set both."
            )

        return self

    @model_validator(mode="after")
    def auto_setup_lora(self):
        if self.trainer.model.lora is not None:
            if self.trainer.weight_broadcast.type == "nccl":
                raise ValueError("NCCL weight broadcast does not support LoRA yet.")

            if self.orchestrator.model.lora is None:
                from prime_rl.configs.orchestrator import LoRAConfig

                self.orchestrator.model.lora = LoRAConfig()

            if (
                self.orchestrator.model.lora.rank is not None
                and self.orchestrator.model.lora.rank != self.trainer.model.lora.rank
            ):
                raise ValueError(
                    f"orchestrator.model.lora.rank ({self.orchestrator.model.lora.rank}) conflicts with "
                    f"trainer.model.lora.rank ({self.trainer.model.lora.rank}). "
                    f"Remove orchestrator.model.lora.rank to inherit from trainer, or update trainer.model.lora.rank to match."
                )

            if (
                self.orchestrator.model.lora.alpha is not None
                and self.orchestrator.model.lora.alpha != self.trainer.model.lora.alpha
            ):
                raise ValueError(
                    f"orchestrator.model.lora.alpha ({self.orchestrator.model.lora.alpha}) conflicts with "
                    f"trainer.model.lora.alpha ({self.trainer.model.lora.alpha}). "
                    f"Remove orchestrator.model.lora.alpha to inherit from trainer, or update trainer.model.lora.alpha to match."
                )

            if self.orchestrator.model.lora.rank is None:
                self.orchestrator.model.lora.rank = self.trainer.model.lora.rank

            if self.orchestrator.model.lora.alpha is None:
                self.orchestrator.model.lora.alpha = self.trainer.model.lora.alpha

            if self.orchestrator.model.lora.name is None:
                self.orchestrator.model.lora.name = (
                    f"r{self.orchestrator.model.lora.rank}-a{self.orchestrator.model.lora.alpha}"
                )

            if self.inference is not None:
                self.inference.enable_lora = True
                self.inference.max_lora_rank = self.trainer.model.lora.rank
            else:
                get_logger().warning(
                    "LoRA is enabled, but inference is not configured. When manually starting the inference server, "
                    "make sure to set --enable_lora and --max-lora-rank."
                )

        return self

    @model_validator(mode="after")
    def auto_setup_session_headers(self):
        """Ensure X-Session-ID header is always set for sticky DP-aware routing at the inference router."""
        self.orchestrator.client.extra_headers_from_state.setdefault("X-Session-ID", "example_id")
        return self

    @model_validator(mode="after")
    def auto_setup_router_replay(self):
        if self.trainer.enable_router_replay:
            if self.inference is not None:
                if self.inference.enable_return_routed_experts is False:
                    get_logger().warning(
                        "Router replay is enabled, but inference.enable_return_routed_experts is False. Setting to True."
                    )
                self.inference.enable_return_routed_experts = True
            else:
                get_logger().warning(
                    "Router replay is enabled, but inference is not configured. When manually starting the inference server, make sure to pass `--enable-return-routed-experts` to the vLLM server."
                )
        return self

    @model_validator(mode="after")
    def auto_setup_deployment(self):
        if self.deployment.type == "single_node":  # single-node
            # set num_train_workers to the number of data replicas
            non_data_parallel_size = self.trainer.model.cp
            if self.deployment.num_train_gpus > 1:
                self.orchestrator.num_train_workers = self.deployment.num_train_gpus // non_data_parallel_size

            # fill up inference capacity with dp ranks
            if self.inference is not None:
                num_infer_gpus = self.deployment.num_infer_gpus
                if num_infer_gpus != self.inference.parallel.dp * self.inference.parallel.tp:
                    assert num_infer_gpus % self.inference.parallel.tp == 0, (
                        "Number of inference GPUs must be divisible by the tensor parallel size"
                    )
                    self.inference.parallel.dp = num_infer_gpus // self.inference.parallel.tp
                # Ensure api_server_count matches DP so all workers are created.
                # Without this, the NCCL broadcast group expects dp*tp workers
                # but only api_server_count*tp exist, causing a deadlock.
                dp = self.inference.parallel.dp
                if self.inference.api_server_count < dp and not self.inference.enable_lora:
                    self.inference.api_server_count = dp

        elif self.deployment.type == "multi_node":  # multi-node
            self.orchestrator.num_train_workers = self.deployment.num_train_nodes * self.deployment.gpus_per_node

            if self.deployment.nodes_per_fsdp_group is not None:
                if self.deployment.num_train_nodes % self.deployment.nodes_per_fsdp_group != 0:
                    raise ValueError(
                        f"deployment.num_train_nodes ({self.deployment.num_train_nodes}) must be divisible by "
                        f"deployment.nodes_per_fsdp_group ({self.deployment.nodes_per_fsdp_group})"
                    )
                self.trainer.model.dp_replicate = (
                    self.deployment.num_train_nodes // self.deployment.nodes_per_fsdp_group
                )

            if (
                self.inference is not None
                and self.inference.enable_expert_parallel
                and self.inference.deployment.type != "disaggregated"
            ):
                inference_tp = self.inference.parallel.tp
                if self.deployment.gpus_per_node % inference_tp != 0:
                    raise ValueError(
                        "deployment.gpus_per_node must be divisible by inference.parallel.tp "
                        "when inference.enable_expert_parallel is enabled in multi-node deployment."
                    )

                inferred_dp_local = self.deployment.gpus_per_node // inference_tp
                total_infer_gpus = self.deployment.num_infer_nodes * self.deployment.gpus_per_node
                expected_global_world_size = self.inference.parallel.dp * inference_tp
                if expected_global_world_size != total_infer_gpus:
                    raise ValueError(
                        "For multi-node expert parallel inference, inference.parallel.dp * inference.parallel.tp "
                        f"must match total inference GPUs ({total_infer_gpus}), got {expected_global_world_size}."
                    )

                if self.inference.data_parallel_size_local is None:
                    self.inference.data_parallel_size_local = inferred_dp_local
                elif self.inference.data_parallel_size_local != inferred_dp_local:
                    raise ValueError(
                        "inference.data_parallel_size_local must equal deployment.gpus_per_node / inference.parallel.tp "
                        f"({inferred_dp_local}) when inference.enable_expert_parallel is enabled in multi-node deployment."
                    )

                if not self.inference.enable_lora and self.inference.api_server_count == self.inference.parallel.dp:
                    self.inference.api_server_count = inferred_dp_local

            # Auto-infer DP and api_server_count for standard multi-node inference.
            # Without EP, vLLM only creates api_server_count * tp workers per node,
            # not gpus_per_node workers. If DP isn't set, the broadcast group expects
            # more workers than exist, deadlocking NCCL init.
            if (
                self.inference is not None
                and not self.inference.enable_expert_parallel
                and self.inference.deployment.type != "disaggregated"
            ):
                dp_per_node = self.deployment.gpus_per_node // self.inference.parallel.tp
                if self.inference.parallel.dp == 1 and dp_per_node > 1:
                    self.inference.parallel.dp = dp_per_node
                if self.inference.data_parallel_size_local is None and dp_per_node > 1:
                    self.inference.data_parallel_size_local = dp_per_node
                if self.inference.api_server_count == 1 and dp_per_node > 1:
                    self.inference.api_server_count = dp_per_node

            if self.weight_broadcast is not None and self.weight_broadcast.type == "nccl":
                # Compute inference_world_size from actual worker count per server:
                # each api_server runs tp workers that participate in collective_rpc.
                api_server_count = self.inference.api_server_count if self.inference else 1
                tp = self.inference.parallel.tp if self.inference else 1
                total_infer_workers = self.deployment.total_infer_nodes * api_server_count * tp
                assert self.trainer.weight_broadcast.type == "nccl"
                self.trainer.weight_broadcast.host = "0.0.0.0"
                self.trainer.weight_broadcast.inference_world_size = total_infer_workers
                assert self.orchestrator.weight_broadcast.type == "nccl"
                self.orchestrator.weight_broadcast.inference_world_size = total_infer_workers

        return self

    @model_validator(mode="after")
    def auto_setup_disaggregated_inference(self):
        """Auto-setup for disaggregated P/D inference within a multi-node deployment."""
        if self.inference is None or self.inference.deployment.type != "disaggregated":
            return self
        if self.deployment.type != "multi_node":
            return self

        infer_deploy = self.inference.deployment
        expected_infer_nodes = infer_deploy.num_prefill_nodes + infer_deploy.num_decode_nodes
        if self.deployment.num_infer_nodes != expected_infer_nodes:
            raise ValueError(
                f"deployment.num_infer_nodes ({self.deployment.num_infer_nodes}) must equal "
                f"inference.deployment.num_prefill_nodes ({infer_deploy.num_prefill_nodes}) + "
                f"inference.deployment.num_decode_nodes ({infer_deploy.num_decode_nodes}) = {expected_infer_nodes}"
            )

        total_infer_gpus = self.deployment.total_infer_nodes * self.deployment.gpus_per_node
        if self.weight_broadcast is not None and self.weight_broadcast.type == "nccl":
            assert self.trainer.weight_broadcast.type == "nccl"
            self.trainer.weight_broadcast.inference_world_size = total_infer_gpus
            assert self.orchestrator.weight_broadcast.type == "nccl"
            self.orchestrator.weight_broadcast.inference_world_size = total_infer_gpus

        return self

    @model_validator(mode="after")
    def auto_setup_dp_rank_count(self):
        """Auto-set orchestrator client dp_rank_count from inference DP size.

        Uses data_parallel_size_local (per-node DP) when set, since each base URL
        points to a single node whose API server only knows about its local ranks.
        Falls back to the global parallel.dp for single-node setups.
        """
        if self.inference is not None and "dp_rank_count" not in self.orchestrator.client.model_fields_set:
            self.orchestrator.client.dp_rank_count = (
                self.inference.data_parallel_size_local or self.inference.parallel.dp
            )
        return self

    @model_validator(mode="after")
    def auto_setup_teacher_inference(self):
        """Auto-configure teacher inference server and orchestrator teacher_model client."""
        if self.deployment.type != "single_node":
            return self
        if self.deployment.num_teacher_gpus is None or self.deployment.num_teacher_gpus == 0:
            return self

        import copy

        from prime_rl.configs.orchestrator import TeacherModelConfig

        if self.teacher_inference is None:
            if self.inference is None:
                self.teacher_inference = InferenceConfig()
            else:
                self.teacher_inference = copy.deepcopy(self.inference)
            self.teacher_inference.server.port = (self.inference.server.port if self.inference else 8000) + 1
        elif self.inference is not None and self.teacher_inference.server.port == self.inference.server.port:
            raise ValueError(
                f"teacher_inference.server.port ({self.teacher_inference.server.port}) conflicts with "
                f"inference.server.port ({self.inference.server.port}). "
                "Either use different ports or let teacher_inference be auto-configured."
            )

        tp = self.teacher_inference.parallel.tp
        num_teacher_gpus = self.deployment.num_teacher_gpus
        if num_teacher_gpus != self.teacher_inference.parallel.dp * tp:
            assert num_teacher_gpus % tp == 0, "Number of teacher GPUs must be divisible by tensor parallel size"
            assert num_teacher_gpus > 0, "num_teacher_gpus cannot be zero"
            self.teacher_inference.parallel.dp = num_teacher_gpus // tp

        if self.orchestrator.teacher_model is None:
            self.orchestrator.teacher_model = TeacherModelConfig()
        host = self.teacher_inference.server.host or "localhost"
        port = self.teacher_inference.server.port
        self.orchestrator.teacher_model.client.base_url = [f"http://{host}:{port}/v1"]
        self.orchestrator.teacher_model.model.name = self.teacher_inference.model.name

        return self

    @model_validator(mode="after")
    def auto_setup_slurm_template(self):
        """Auto-setup the default single-node/multi-node SLURM template if no custom template is provided."""
        if self.slurm is not None and self.slurm.template_path is None:
            import prime_rl

            templates_dir = Path(prime_rl.__file__).parent / "templates"
            if self.deployment.type == "single_node":
                self.slurm.template_path = templates_dir / "single_node_rl.sbatch.j2"
            else:
                self.slurm.template_path = templates_dir / "multi_node_rl.sbatch.j2"
        return self

    ### Warnings
