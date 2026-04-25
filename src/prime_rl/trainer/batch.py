import copy

from prime_rl.transport.types import MicroBatch, TrainingSample


def prepare_sample(training_example: TrainingSample, seq_len: int) -> MicroBatch:
    """
    Prepare a problem for sequence packing training.
    Tokenize and prepare tensors.
    """
    input_ids = training_example.prompt_ids + training_example.completion_ids
    loss_mask = training_example.prompt_mask + training_example.completion_mask
    # Latent-bug guard: `inference_logprobs` for prompt tokens is hard-zeroed
    # below. CISPO's importance ratio rho = exp(trainer_lp - inference_lp);
    # if `prompt_mask` ever turns trainable on a prompt token, that token's
    # IS ratio collapses (trainer_lp is real, inference_lp is 0 → rho ≈ 0)
    # and the gradient on prompt-trainable tokens silently dies. Verifiers'
    # bundled clients hard-code `prompt_mask = [0]*len(prompt_ids)`, but a
    # custom client that flips this would silently corrupt training. Fail
    # loudly here instead. If you legitimately need prompt-trainable
    # tokens, plumb real prompt-side inference logprobs through and remove
    # the guard.
    if any(training_example.prompt_mask):
        raise ValueError(
            "prepare_sample: prompt-side trainable tokens (prompt_mask=True) "
            "are not supported — `inference_logprobs` for prompt positions is "
            "hard-zeroed below, which would silently zero CISPO's importance "
            "ratio and kill the gradient on those tokens. Plumb real prompt "
            "logprobs through and lift this guard before re-enabling."
        )
    inference_logprobs = [0.0] * len(training_example.prompt_ids) + training_example.completion_logprobs
    advantages = [training_example.advantage] * len(input_ids)
    position_ids = list(range(len(input_ids)))
    mm_token_type_ids = training_example.mm_token_type_ids

    # Per-token temperatures: prompt tokens use first completion temp (masked out anyway)
    # Default to 1.0 if completion is empty (e.g., model generated only tool calls with no text)
    prompt_temp = training_example.completion_temperatures[0] if training_example.completion_temperatures else 1.0
    temperatures = [prompt_temp] * len(training_example.prompt_ids) + training_example.completion_temperatures

    # Teacher logprobs already cover the full sequence (prompt + completion),
    # computed via prefill in the orchestrator when a teacher model is configured
    teacher_logprobs = training_example.teacher_logprobs
    routed_experts = training_example.routed_experts

    if len(input_ids) > seq_len:
        input_ids = input_ids[:seq_len]
        loss_mask = loss_mask[:seq_len]
        inference_logprobs = inference_logprobs[:seq_len]
        position_ids = position_ids[:seq_len]
        advantages = advantages[:seq_len]
        temperatures = temperatures[:seq_len]
        if teacher_logprobs is not None:
            teacher_logprobs = teacher_logprobs[:seq_len]
        if routed_experts is not None:
            routed_experts = routed_experts[:seq_len]
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids[:seq_len]

    assert (
        len(input_ids)
        == len(advantages)
        == len(loss_mask)
        == len(position_ids)
        == len(inference_logprobs)
        == len(temperatures)
    ), (
        f"input_ids: {len(input_ids)}, advantages: {len(advantages)}, loss_mask: {len(loss_mask)}, position_ids: {len(position_ids)}, inference_logprobs: {len(inference_logprobs)}, temperatures: {len(temperatures)}"
    )
    if teacher_logprobs is not None:
        assert len(teacher_logprobs) == len(input_ids), f"teacher_logprobs: {len(teacher_logprobs)}"

    if routed_experts is not None:
        assert len(routed_experts) == len(input_ids), (
            f"routed_experts: {len(routed_experts)}, input_ids: {len(input_ids)}"
        )

    if mm_token_type_ids is not None:
        assert len(mm_token_type_ids) == len(input_ids), (
            f"mm_token_type_ids: {len(mm_token_type_ids)}, input_ids: {len(input_ids)}"
        )

    return MicroBatch(
        input_ids=input_ids,
        advantages=advantages,
        loss_mask=loss_mask,
        position_ids=position_ids,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=teacher_logprobs,
        temperatures=temperatures,
        routed_experts=routed_experts,
        mm_token_type_ids=mm_token_type_ids,
        # Multimodal fields (Qwen3-VL) - passed through without modification
        pixel_values=training_example.pixel_values,
        pixel_values_shape=training_example.pixel_values_shape,
        image_grid_thw=training_example.image_grid_thw,
        # Default to a neutral 1.0 weight per packed sequence; overwritten by
        # apply_prompt_average_sequence_weights when prompt_average_loss is set.
        sequence_loss_weights=[1.0],
        # Carry forward the synthesized-logprobs flag so the train-loop's
        # runtime check can refuse importance-ratio losses against zero-fill
        # inference_logprobs. Static `validate_external_rollout_mode` rejects
        # the canonical `teacher_rollout_model + CISPO` combination at config
        # load, but a user pointing `[orchestrator.client]` at an external
        # service WITHOUT setting `teacher_rollout_model` bypasses the static
        # check; the runtime guard in trainer/rl/train.py is the load-bearing
        # second line of defense, and it depends on this carry-forward. Read
        # strictly — the field is a non-optional bool on TrainingSample, so a
        # getattr fallback would silently disable the guard if the schema ever
        # changed.
        inference_logprobs_synthesized=training_example.inference_logprobs_synthesized,
    )


def _is_multimodal_sample(sample: MicroBatch) -> bool:
    """Check if a sample contains multimodal data (images)."""
    return sample.pixel_values is not None


def packed_samples_into_micro_bs(
    samples: list[tuple[int, MicroBatch]], max_seq_len: int, num_loras: int
) -> list[MicroBatch]:
    """
    Pack samples into micro_batch efficiently.
    We follow the First Fit Decreasing algorithm to pack the samples into bins and minimize potential padding while never truncating.
    With per-token temperatures, samples can be packed together regardless of their temperature values.

    NOTE: Multimodal samples (with pixel_values) are NOT packed together as they have variable-sized
    vision data that doesn't pack well. Each multimodal sample becomes its own micro batch.
    """
    # Sort by (lora_idx, -length) for packing efficiency
    samples.sort(key=lambda x: (x[0], -len(x[1].input_ids)))

    ## we create bins
    micro_batches: list[MicroBatch] = []

    for idx, sample in samples:
        # Multimodal samples cannot be packed - each becomes its own micro batch
        if _is_multimodal_sample(sample):
            sample.lora_num_tokens = [0] * num_loras
            sample.lora_num_tokens[idx] = len(sample.input_ids)
            micro_batches.append(sample)
            continue

        # Try to find a bin that can fit this sequence (only pack text-only samples)
        for bin_content in micro_batches:
            # Don't pack into multimodal micro batches
            if _is_multimodal_sample(bin_content):
                continue
            # Check if sequence fits in this bin
            if len(bin_content.input_ids) + len(sample.input_ids) <= max_seq_len:
                bin_content.input_ids.extend(sample.input_ids)
                bin_content.loss_mask.extend(sample.loss_mask)
                bin_content.advantages.extend(sample.advantages)
                bin_content.inference_logprobs.extend(sample.inference_logprobs)
                bin_content.temperatures.extend(sample.temperatures)
                if sample.teacher_logprobs is not None:
                    if bin_content.teacher_logprobs is None:
                        bin_content.teacher_logprobs = []
                    bin_content.teacher_logprobs.extend(sample.teacher_logprobs)
                if sample.routed_experts is not None:
                    if bin_content.routed_experts is None:
                        bin_content.routed_experts = []
                    bin_content.routed_experts.extend(sample.routed_experts)
                if sample.mm_token_type_ids is not None:
                    if bin_content.mm_token_type_ids is None:
                        bin_content.mm_token_type_ids = []
                    bin_content.mm_token_type_ids.extend(sample.mm_token_type_ids)
                bin_content.position_ids.extend(sample.position_ids)
                bin_content.sequence_loss_weights.extend(sample.sequence_loss_weights)
                # Bin is "synthesized" if ANY contributing sample has the flag —
                # one corrupted sample in the pack contaminates the whole bin's
                # importance-ratio computation.
                if sample.inference_logprobs_synthesized:
                    bin_content.inference_logprobs_synthesized = True
                bin_content.lora_num_tokens[idx] += len(sample.input_ids)
                break
        else:
            sample.lora_num_tokens = [0] * num_loras
            sample.lora_num_tokens[idx] = len(sample.input_ids)
            micro_batches.append(sample)

    return micro_batches


def pad_micro_batch(micro_batch: MicroBatch, pad_to_multiple_of: int) -> MicroBatch:
    """
    Pad a micro batch with the given padding size sample
    Return the padded micro batch.
    Args:
        micro_batch: The micro batch to pad.
        padding_size: The number of padding tokens to add.
    Returns:
        The padded micro batch.
    """

    padding_size = (pad_to_multiple_of - (len(micro_batch.input_ids) % pad_to_multiple_of)) % pad_to_multiple_of

    if not (pad_to_multiple_of > 1 and padding_size > 0):
        return micro_batch

    micro_batch.input_ids.extend([1] * padding_size)
    micro_batch.advantages.extend([0.0] * padding_size)
    micro_batch.loss_mask.extend([False] * padding_size)
    micro_batch.position_ids.extend(list(range(padding_size)))
    micro_batch.inference_logprobs.extend([0.0] * padding_size)
    # Use temperature 1.0 for padding tokens (doesn't matter since loss_mask is False)
    micro_batch.temperatures.extend([1.0] * padding_size)
    if micro_batch.teacher_logprobs is not None:
        micro_batch.teacher_logprobs.extend([0.0] * padding_size)
    micro_batch.lora_num_tokens[-1] += (
        padding_size  # We send padding to the last lora so that tokens have ascending lora idx
    )
    if micro_batch.mm_token_type_ids is not None:
        micro_batch.mm_token_type_ids.extend([0] * padding_size)
    # routed_experts is per-token; without padding here, the downstream
    # `torch.tensor(...).reshape(...)` in `_micro_batch_to_tensor` would
    # either crash on shape mismatch or silently misalign router-replay
    # supervision with token positions. Pad with **0** (valid expert index),
    # NOT -1: the MoE forward pass calls `scores.gather(dim=1, index=routed_experts)`
    # unconditionally before any loss masking, so any out-of-range index
    # crashes the gather (RuntimeError on CPU, device-side assert on CUDA).
    # `loss_mask=False` does not gate the gather, only the loss; the comment
    # in the prior version got that wrong. Filler 0 matches the convention
    # used by `_align_routed_experts` in orchestrator/trajectories.py and
    # only over-counts expert 0 in the load-balancing histogram by the
    # padding-token count — pad tokens are loss-masked so the bias is benign.
    if micro_batch.routed_experts is not None and len(micro_batch.routed_experts) > 0:
        layers_topk = micro_batch.routed_experts[0]
        pad_entry = [[0 for _ in row] for row in layers_topk]
        micro_batch.routed_experts.extend([pad_entry] * padding_size)
    # The padding tokens get position_ids = range(padding_size) above, which restarts at 0.
    # `get_response_lengths` (utils.py) reads position_id resets as sequence boundaries — but
    # only when the NEXT position_id is 1 (otherwise the 0 is treated as trailing padding of
    # the previous sequence). So a phantom sequence is created only when padding_size >= 2.
    # When padding_size == 1 the trailing single `0` is folded into the last real sequence
    # and num_packed_sequences is unchanged; appending a phantom weight would produce a
    # length-mismatch crash in compute_loss.
    if micro_batch.sequence_loss_weights and padding_size >= 2:
        # Match the existing weight semantics: if the packer set non-trivial
        # weights (prompt-avg path), append `0.0` so the phantom sequence
        # contributes zero. If the packer left them at the neutral `[1.0]*N`
        # default (token-mode path), append `1.0` instead — `compute_loss`
        # rejects non-trivial weights under token mode and would otherwise
        # raise on the legitimate cp>1 padding case.
        existing = micro_batch.sequence_loss_weights
        is_neutral = all(w == 1.0 for w in existing)
        micro_batch.sequence_loss_weights.append(1.0 if is_neutral else 0.0)

    return micro_batch


def _make_dummy_batch(source: MicroBatch) -> MicroBatch:
    """Create a zero-loss dummy batch from an existing batch, preserving its modality."""
    dummy = copy.deepcopy(source)
    dummy.advantages = [0.0] * len(dummy.input_ids)
    dummy.loss_mask = [False] * len(dummy.input_ids)
    # The IS-ratio runtime guard rejects any micro-batch carrying the synth
    # flag against CISPO/Default. A dummy is loss-mask-zero, so it contributes
    # nothing — but the deepcopy would otherwise inherit the source's flag and
    # trip the guard before the loop short-circuits on zero loss_mask. Reset it.
    dummy.inference_logprobs_synthesized = False
    # Zero the sequence weights so the dummy contributes nothing to the loss when
    # loss_scale_mode is "sequence" / "none". For token-mode runs the source's
    # weights are the neutral `[1.0]*N` default, and `compute_loss` rejects any
    # non-1.0 weight under token mode — preserve the `1.0`s in that case (the
    # loss_mask is all-False so the dummy contributes zero regardless).
    is_neutral = all(w == 1.0 for w in dummy.sequence_loss_weights)
    fill = 1.0 if is_neutral else 0.0
    dummy.sequence_loss_weights = [fill] * len(dummy.sequence_loss_weights)
    return dummy


def _pad_group_for_distribution(group: list[MicroBatch], num_train_workers: int) -> list[MicroBatch]:
    """Pad a group of micro batches so its length is divisible by num_train_workers."""
    num_padding = -len(group) % num_train_workers
    if num_padding > 0 and len(group) > 0:
        dummy = _make_dummy_batch(group[0])
        group.extend([dummy] * num_padding)
    return group


def _trainable_completion_tokens(rollout: TrainingSample, seq_len: int) -> int:
    """Count tokens that actually enter the loss after prepare_sample().

    prepare_sample builds `loss_mask = prompt_mask + completion_mask` then
    truncates the concatenation to seq_len. compute_loss weights summed
    losses against the FULL loss_mask, so prompt-side trainable tokens
    would count if any existed.

    Today verifiers' bundled clients hard-zero `prompt_mask` and
    `prepare_sample` raises if any prompt position is trainable (CISPO's
    `inference_logprobs=0` for prompt tokens would otherwise zero the IS
    ratio). The prompt-side branch below is therefore zero in practice
    but kept for a future env that legitimately plumbs prompt-side
    inference logprobs and lifts the guard.

    The truncation rule: positions [0, prompt_len) come from prompt_mask,
    positions [prompt_len, prompt_len + completion_len) from completion_mask.
    After truncation to seq_len, prompt-side keeps the first
    `min(prompt_len, seq_len)` entries; completion-side keeps the first
    `max(0, seq_len - prompt_len)` entries. Sum the surviving True bits.
    """
    prompt_len = len(rollout.prompt_ids)
    surviving_prompt = min(prompt_len, seq_len)
    surviving_completion = max(0, seq_len - prompt_len)
    prompt_trainable = int(sum(rollout.prompt_mask[:surviving_prompt]))
    completion_trainable = int(sum(rollout.completion_mask[:surviving_completion]))
    return prompt_trainable + completion_trainable


def apply_prompt_average_sequence_weights(
    rollouts: list[TrainingSample], seq_len: int
) -> list[float] | None:
    """Compute ScaleRL §3.3 prompt-level loss averaging weights.

    The per-sample loss callbacks in `trainer/rl/loss.py` already sum over
    trainable tokens (CISPO `pg_per_token.sum()`, SFT `(-trainer_logprobs[loss_mask]).sum()`).
    To turn that into "every prompt contributes equally, every token within a
    prompt contributes equally," each non-empty sample in prompt p gets

        w_i = 1 / (total_trainable_tokens_in_p * num_prompts)

    so that
        sum_i in p (w_i * loss_i) = sum_p_tokens / total_p / num_prompts
                                  = (per-token mean loss within p) / num_prompts
    and summing over prompts gives `mean_p (per-token mean within p)`.

    Token counts come from `_trainable_completion_tokens`, which excludes
    non-trainable completion tokens and tokens dropped by `prepare_sample`'s
    seq_len truncation.

    Group key: prefer the scheduler-assigned `group_id`. Each call to
    `Scheduler._schedule_next_request` increments next_group_id by one, so a
    prompt sampled twice in the same step (which happens whenever
    `batch_size / rollouts_per_example > num_unique_examples`, e.g. TB's 18
    prompts × 48 groups) gets distinct group_ids. compute_advantages already
    treats those as separate groups; matching the prompt-avg path here
    keeps the two consistent. Fall back to `(env_name, example_id)` when
    `group_id` is None (older serialized batches, custom callers).

    Empty samples (zero trainable tokens after truncation) get weight 0 — they
    contribute nothing under any choice of weight, so this just makes that
    explicit. Empty PROMPTS (every sample empty) are excluded from the
    `num_prompts` denominator: counting them would dilute every surviving
    prompt's contribution by `1 - num_empty/num_total`, deviating from
    ScaleRL §3.3's "every prompt contributes equally" objective. With one
    all-empty prompt in a 4-prompt batch, surviving prompts would otherwise
    lose 25% mass for no semantic reason.

    Returns None when no rollouts have prompt_average_loss set (caller leaves the
    default neutral 1.0 weight in place). Raises if the flag is set inconsistently
    across the batch — mixing schemes would silently yield wrong gradients.
    """
    if not any(rollout.prompt_average_loss for rollout in rollouts):
        return None
    if not all(rollout.prompt_average_loss for rollout in rollouts):
        raise ValueError(
            "prompt_average_loss must be set consistently for every sample in a batch; "
            "mixing prompt-avg and other reductions in the same step yields wrong gradients."
        )

    # If any rollout has a group_id, every rollout must — partial assignment
    # would silently mix the two keying schemes and miscount num_prompts.
    # Strict access: TrainingSample.group_id is declared on the msgspec schema
    # (default None); a getattr fallback would silently downgrade to the
    # legacy keying path on a schema rename.
    have_gid = [r.group_id is not None for r in rollouts]
    if any(have_gid) and not all(have_gid):
        raise ValueError(
            "prompt_average_loss requires consistent group_id assignment: some rollouts "
            "have group_id set and others don't. Either every rollout has a scheduler-assigned "
            "group_id or none do (legacy fallback to (env_name, example_id))."
        )
    use_gid = bool(have_gid and all(have_gid))

    by_prompt: dict[tuple[str, ...], list[int]] = {}
    for i, rollout in enumerate(rollouts):
        if use_gid:
            key: tuple[str, ...] = (str(rollout.group_id),)
        else:
            if rollout.example_id is None:
                raise ValueError("example_id is required when prompt_average_loss is enabled")
            # Strict access — env_name is a schema field; getattr fallback
            # would silently lose the cross-env disambiguation contract.
            env_name = rollout.env_name
            if not env_name:
                raise ValueError(
                    "env_name is required on every sample when prompt_average_loss is enabled "
                    "(needed to disambiguate example_id across envs)"
                )
            key = (str(env_name), str(rollout.example_id))
        by_prompt.setdefault(key, []).append(i)

    if len(by_prompt) == 0:
        raise ValueError("prompt_average_loss requires at least one prompt in the batch")

    # First pass: compute per-prompt token totals so we can exclude empty
    # prompts from `num_prompts`. ScaleRL §3.3 prescribes "mean over prompts"
    # of the per-prompt token-mean. An empty prompt has no per-prompt mean
    # (no tokens), so it's not part of the outer mean's domain — counting
    # it in the denominator would shrink every other prompt's contribution
    # by the empty-prompt fraction.
    prompt_buckets: list[tuple[list[int], list[int], int]] = []
    for indices in by_prompt.values():
        sample_tokens = [_trainable_completion_tokens(rollouts[i], seq_len) for i in indices]
        total = sum(sample_tokens)
        prompt_buckets.append((indices, sample_tokens, total))

    num_prompts = sum(1 for _, _, total in prompt_buckets if total > 0)
    weights = [0.0] * len(rollouts)
    if num_prompts == 0:
        # Every prompt was empty — no signal to weight. Caller's neutral
        # 1.0 default never reaches here (we'd return None upstream), but
        # the all-empty-yet-prompt-avg-flagged case still needs a valid
        # `weights` list.
        return weights

    for indices, sample_tokens, total in prompt_buckets:
        if total == 0:
            continue
        per_sample_weight = 1.0 / (total * num_prompts)
        for i, n in zip(indices, sample_tokens):
            weights[i] = per_sample_weight if n > 0 else 0.0
    return weights


def prepare_batch(
    rollouts: list[TrainingSample],
    seq_len: int,
    num_train_workers: int,
    idxs: list[int],
    num_loras: int,
    pad_to_multiple_of: int = 1,
) -> list[list[MicroBatch]]:
    """
    Prepare a batch of problems for each GPU. Each batch is a list of micro batches.
    Each micro batch is shape [1, seq_len], the number of samples is not fixed per micro batch.

    FSDP requires all ranks to execute the same operations at each step. If one rank
    processes a multimodal batch (triggering the vision encoder) while another processes
    a text-only batch, the all-gather will hang. We separate micro batches by modality
    and distribute them so that at each step index, all ranks see the same modality.
    """
    prompt_weights = apply_prompt_average_sequence_weights(rollouts, seq_len)
    all_samples = [(idx, prepare_sample(rollout, seq_len)) for idx, rollout in zip(idxs, rollouts)]
    # Overwrite the default neutral 1.0 weight with the prompt-avg weight when set.
    if prompt_weights is not None:
        for (_, micro_batch), w in zip(all_samples, prompt_weights):
            micro_batch.sequence_loss_weights = [float(w)]

    micro_batches = packed_samples_into_micro_bs(all_samples, seq_len, num_loras)
    micro_batches = [pad_micro_batch(micro_batch, pad_to_multiple_of) for micro_batch in micro_batches]

    # Separate by modality so each step index has uniform modality across all ranks
    mm_batches = [b for b in micro_batches if _is_multimodal_sample(b)]
    text_batches = [b for b in micro_batches if not _is_multimodal_sample(b)]

    # Pad each group independently so its count is divisible by num_train_workers
    mm_batches = _pad_group_for_distribution(mm_batches, num_train_workers)
    text_batches = _pad_group_for_distribution(text_batches, num_train_workers)

    # Combine: all multimodal first, then all text-only. Since each group's length is
    # divisible by num_train_workers, the modality boundary aligns with distribution rows.
    ordered = mm_batches + text_batches

    assert len(ordered) % num_train_workers == 0, "Number of micro batches is not divisible by number of data ranks"

    # Distribute in strided order so each step index has the same modality across ranks
    batches_per_gpu: list[list[MicroBatch]] = [[] for _ in range(num_train_workers)]
    for i, batch in enumerate(ordered):
        batches_per_gpu[i % num_train_workers].append(batch)

    return batches_per_gpu
