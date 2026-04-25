import msgspec


# Orchestrator -> Packer
class TrainingSample(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A single training example."""

    prompt_ids: list[int]
    prompt_mask: list[bool]
    completion_ids: list[int]
    completion_mask: list[bool]
    completion_logprobs: list[float]
    completion_temperatures: list[float]  # Per-token temperatures used during generation
    teacher_logprobs: list[float] | None = None
    advantage: float | None = None
    reward: float | None = None

    # Multimodal fields (Qwen3-VL) — pixel_values stored as raw float32 bytes for efficient serialization
    pixel_values: bytes | None = None
    pixel_values_shape: list[int] | None = None  # [num_patches, patch_dim]
    # image_grid_thw: grid dimensions [num_images, 3] where each entry is [temporal, height, width]
    image_grid_thw: list[list[int]] | None = None

    routed_experts: list[list[list[int]]] | None = None  # [seq_len, layers, topk]

    # mm_token_type_ids: token type ids per token [batch seq], int64 (0=text, 1=image, 2=video)
    mm_token_type_ids: list[int] | None = None

    # Stable per-prompt id used by the trainer to compute prompt-level loss weights when
    # OrchestratorConfig.prompt_average_loss is True. `example_id` is only unique within
    # an env, so the trainer pairs it with `env_name` to disambiguate cross-env batches.
    # Fields appended at the END so msgspec array_like=True backward-compat with old
    # serialized batches survives.
    example_id: str | None = None
    prompt_average_loss: bool = False
    env_name: str | None = None
    # True iff `completion_logprobs` were reconstructed from chat-template render
    # (no real generator logprobs). Set by `pretokenize_rollout_trajectory` when the
    # chat client did not preserve token data. Importance-ratio losses (CISPO,
    # DefaultLoss) MUST refuse to train on these — `rho = exp(trainer_lp - 0)`
    # is not the IS ratio. SFT can train on them (it does not consume the field).
    inference_logprobs_synthesized: bool = False
    # Scheduler-assigned group_id for this rollout. Each scheduler call to
    # `_schedule_next_request` increments group_id, so duplicate samples of
    # the same `(env_name, example_id)` (which is allowed under sampling-with-
    # replacement on small training pools) get distinct group_ids. The trainer
    # uses this for prompt-level loss averaging so duplicates count as
    # separate prompt slots — compute_advantages treats them that way too.
    group_id: int | None = None


class TrainingBatch(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A batch of training examples with metadata for transport."""

    examples: list[TrainingSample]
    step: int
    run_idx: int | None = None


# Packer -> Trainer
class MicroBatch(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A micro batch of data for training."""

    input_ids: list[int]
    loss_mask: list[bool]
    advantages: list[float]
    inference_logprobs: list[float]
    position_ids: list[int]
    temperatures: list[float]  # Per-token temperatures used during generation
    teacher_logprobs: list[float] | None = None
    lora_num_tokens: list[int] | None = None
    routed_experts: list[list[list[int]]] | None = None

    # Multimodal fields (Qwen3-VL) — pixel_values stored as raw float32 bytes for efficient serialization
    pixel_values: bytes | None = None
    pixel_values_shape: list[int] | None = None  # [num_patches, patch_dim]
    # image_grid_thw: grid dimensions [num_images, 3] where each entry is [temporal, height, width]
    image_grid_thw: list[list[int]] | None = None
    # mm_token_type_ids: token type ids per token [batch seq], int64 (0=text, 1=image, 2=video)
    mm_token_type_ids: list[int] | None = None

    # Per-sequence loss weights (one float per packed sample) used when the loss
    # function's loss_scale_mode is "sequence" / "none" (e.g. ScaleRL prompt-level
    # averaging). Default `[]` keeps msgspec backward-compat with old serialized
    # batches — readers fall through the `not weights` branch in compute_loss.
    sequence_loss_weights: list[float] = []
    # True iff ANY of the packed samples in this micro-batch has
    # `inference_logprobs_synthesized=True` (i.e. completion_logprobs were
    # reconstructed from chat-template render with no real generator
    # logprobs). The trainer's `compute_loss` rejects this case for
    # importance-ratio losses (CISPO, Default) — `rho = exp(trainer_lp - 0)`
    # is not the IS ratio. SFT can train on it (does not consume the field).
    inference_logprobs_synthesized: bool = False
