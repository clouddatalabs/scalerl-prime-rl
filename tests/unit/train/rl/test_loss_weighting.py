"""CPU-runnable tests for ScaleRL prompt-level loss averaging + canonical CISPO."""

import math

import pytest
import torch

from prime_rl.configs.trainer import CISPOLossConfig, CustomLossConfig, DefaultLossConfig, SFTLossConfig
from prime_rl.trainer.batch import apply_prompt_average_sequence_weights
from prime_rl.trainer.rl.loss import LossInputs, LossOutputs, cispo_loss_fn, compute_loss, setup_loss_fn
from prime_rl.transport.types import TrainingSample


def _make_sample(
    *,
    example_id: str,
    completion_len: int,
    prompt_average_loss: bool = True,
    advantage: float = 1.0,
    env_name: str = "env_a",
) -> TrainingSample:
    return TrainingSample(
        prompt_ids=[1, 2, 3],
        prompt_mask=[True, True, True],
        completion_ids=[10] * completion_len,
        completion_mask=[True] * completion_len,
        completion_logprobs=[0.0] * completion_len,
        completion_temperatures=[1.0] * completion_len,
        advantage=advantage,
        reward=0.0,
        example_id=example_id,
        prompt_average_loss=prompt_average_loss,
        env_name=env_name,
    )


def test_apply_prompt_average_sequence_weights_balances_prompts():
    """Weights make every prompt contribute equally and every token-within-prompt contribute equally.

    The per-sample loss callbacks (`cispo_loss_fn`, `sft_loss_fn`) return summed
    losses over trainable tokens. So each non-empty sample in prompt p must get
    `w_i = 1 / (total_p * num_prompts)` — uniform within a prompt — for the sum
    `sum_i (w_i * loss_i)` to equal the per-token mean within p divided by
    num_prompts.
    """
    rollouts = [
        _make_sample(example_id="A", completion_len=10),
        _make_sample(example_id="A", completion_len=30),
        _make_sample(example_id="B", completion_len=20),
        _make_sample(example_id="B", completion_len=20),
    ]
    weights = apply_prompt_average_sequence_weights(rollouts, seq_len=4096)
    assert weights is not None
    # Prompt A: total = 40 trainable tokens; per-sample weight = 1 / (40 * 2) = 0.0125
    # Prompt B: total = 40 trainable tokens; per-sample weight = 1 / (40 * 2) = 0.0125
    assert weights == pytest.approx([0.0125, 0.0125, 0.0125, 0.0125])


def test_apply_prompt_average_sequence_weights_yields_per_token_mean_for_uniform_loss():
    """End-to-end sanity for snowflake_poc_critique.md §1.

    With uniform per-token loss = 1, summed-loss-per-sample equals
    `trainable_tokens_i`. The expected total loss is 1 (per-token mean = 1
    averaged over equal prompts = 1). The previous formula
    `w_i = trainable_tokens_i / (total_p * num_prompts)` produced
    `sum_p sum_i n_i^2 / (total_p * num_prompts)`, which over-weights long
    completions — concretely 25 instead of 1 for the [10, 30] example below.
    """
    rollouts = [
        _make_sample(example_id="A", completion_len=10),
        _make_sample(example_id="A", completion_len=30),
        _make_sample(example_id="B", completion_len=20),
        _make_sample(example_id="B", completion_len=20),
    ]
    weights = apply_prompt_average_sequence_weights(rollouts, seq_len=4096)
    # Per-sample summed losses for uniform per-token loss of 1.
    summed_losses = [10, 30, 20, 20]
    total = sum(w * l for w, l in zip(weights, summed_losses))
    # Two prompts × 1.0 per-token mean / 2 = 1.0
    assert total == pytest.approx(1.0)
    # Sanity: the previously-shipped formula `w_i = n_i / (total_p * num_prompts)`
    # would have given 22.5 here (= (100+900)/80 + (400+400)/80 = 12.5 + 10).
    # Pin a wide margin so this test fails loudly if the regression returns.
    broken_weights = [10 / 80, 30 / 80, 20 / 80, 20 / 80]
    broken_total = sum(w * l for w, l in zip(broken_weights, summed_losses))
    assert broken_total == pytest.approx(22.5)
    assert abs(broken_total - 1.0) > 5.0


def test_apply_prompt_average_sequence_weights_disambiguates_envs():
    """Same example_id across different envs must not collide.

    `example_id` is only unique within an env, so a multi-env training batch
    where env_a/example_id=0 and env_b/example_id=0 are both present would
    fold them into one prompt under the old code (snowflake_poc_critique.md §3).
    Group key is now (env_name, example_id) so the samples below count as
    TWO prompts.

    Use UNEQUAL completion lengths across envs (env_a samples 10 tokens each,
    env_b samples 20 tokens each) so the correct and broken behaviors yield
    different weights — a uniform-shapes test would coincide and let the bug
    sneak back.
    """
    rollouts = [
        _make_sample(example_id="0", completion_len=10, env_name="env_a"),
        _make_sample(example_id="0", completion_len=10, env_name="env_a"),
        _make_sample(example_id="0", completion_len=20, env_name="env_b"),
        _make_sample(example_id="0", completion_len=20, env_name="env_b"),
    ]
    weights = apply_prompt_average_sequence_weights(rollouts, seq_len=4096)
    # Correct: 2 prompts. env_a total = 20 → w = 1/(20*2) = 0.025; env_b total = 40 → w = 0.0125.
    assert weights == pytest.approx([0.025, 0.025, 0.0125, 0.0125])
    # If the key were just example_id (broken), num_prompts=1 and total=60 →
    # uniform w = 1/60 ≈ 0.01667 across all four samples. Pin the divergence.
    broken_uniform_weight = 1.0 / 60.0
    assert weights[0] != pytest.approx(broken_uniform_weight)
    assert weights[2] != pytest.approx(broken_uniform_weight)
    # Sanity: with summed losses = trainable_tokens, total = mean over prompts of per-token mean = 1.
    summed_losses = [10, 10, 20, 20]
    total = sum(w * l for w, l in zip(weights, summed_losses))
    assert total == pytest.approx(1.0)


def test_apply_prompt_average_sequence_weights_requires_env_name():
    sample = _make_sample(example_id="A", completion_len=5, env_name="env_a")
    sample.env_name = None
    with pytest.raises(ValueError, match="env_name is required"):
        apply_prompt_average_sequence_weights([sample], seq_len=4096)


def test_apply_prompt_average_sequence_weights_no_op_when_disabled():
    rollouts = [_make_sample(example_id="A", completion_len=5, prompt_average_loss=False)]
    assert apply_prompt_average_sequence_weights(rollouts, seq_len=4096) is None


def test_apply_prompt_average_sequence_weights_rejects_mixed_flags():
    rollouts = [
        _make_sample(example_id="A", completion_len=5, prompt_average_loss=True),
        _make_sample(example_id="A", completion_len=5, prompt_average_loss=False),
    ]
    with pytest.raises(ValueError, match="set consistently"):
        apply_prompt_average_sequence_weights(rollouts, seq_len=4096)


def test_apply_prompt_average_sequence_weights_requires_example_id():
    sample = _make_sample(example_id="A", completion_len=5, prompt_average_loss=True)
    sample.example_id = None
    with pytest.raises(ValueError, match="example_id is required"):
        apply_prompt_average_sequence_weights([sample], seq_len=4096)


def test_apply_prompt_average_sequence_weights_excludes_non_trainable_completion_tokens():
    """completion_mask=False tokens (e.g. orchestrator-injected prompt extensions) should
    not contribute to the prompt-level weight — only trainable tokens enter the loss."""
    sample = TrainingSample(
        prompt_ids=[1, 2, 3],
        prompt_mask=[True, True, True],
        completion_ids=[10, 11, 12, 13, 14, 15],
        # 4 of 6 completion tokens are trainable; 2 are interleaved tool-call prompt
        # tokens with completion_mask=False.
        completion_mask=[True, True, False, False, True, True],
        completion_logprobs=[0.0] * 6,
        completion_temperatures=[1.0] * 6,
        advantage=1.0,
        reward=0.0,
        example_id="A",
        prompt_average_loss=True,
        env_name="env_a",
    )
    sample2 = _make_sample(example_id="A", completion_len=4)  # 4 trainable tokens too
    weights = apply_prompt_average_sequence_weights([sample, sample2], seq_len=4096)
    # Both samples have 4 trainable tokens; prompt A total = 8; num_prompts = 1.
    # Per-sample weight = 1 / (8 * 1) = 0.125 (uniform within prompt).
    assert weights == pytest.approx([0.125, 0.125])
    # Sanity: with summed losses = trainable_tokens = [4, 4], total = 4*0.125 + 4*0.125 = 1.0.
    assert sum(w * 4 for w in weights) == pytest.approx(1.0)


def test_apply_prompt_average_sequence_weights_excludes_truncated_tokens():
    """seq_len truncation drops tail tokens — they must not count toward the weight."""
    # prompt_len=3, completion_len=20. At seq_len=10 only 10 - 3 = 7 completion tokens
    # survive prepare_sample's truncation.
    sample = _make_sample(example_id="A", completion_len=20)
    weights = apply_prompt_average_sequence_weights([sample], seq_len=10)
    # Per-sample weight = 1 / (7 trainable * 1 prompt). The summed loss for that
    # sample under uniform per-token loss=1 is 7, so total contribution = 1.0.
    assert weights == pytest.approx([1.0 / 7])
    assert sum(w * 7 for w in weights) == pytest.approx(1.0)

    # If prompt itself is at-or-over seq_len, no completion tokens are trainable.
    # Total = 0 → prompt contributes nothing (correct: no tokens, no loss).
    sample_no_completion = _make_sample(example_id="A", completion_len=5)
    weights = apply_prompt_average_sequence_weights([sample_no_completion], seq_len=2)
    assert weights == pytest.approx([0.0])


def test_compute_loss_token_mode_default_unchanged():
    """loss_scale_mode='token' (default for DefaultLossConfig) sums then divides by loss_scale."""
    trainer_logprobs = [torch.tensor([-1.0, -2.0]), torch.tensor([-3.0])]
    inference_logprobs = [torch.zeros(2), torch.zeros(1)]
    advantages = [torch.zeros(2), torch.zeros(1)]
    loss_mask = [torch.tensor([True, True]), torch.tensor([True])]

    loss_fn = setup_loss_fn(SFTLossConfig())
    assert getattr(loss_fn, "loss_scale_mode") == "token"
    loss, _ = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_fn=loss_fn,
        loss_scale=3,
    )
    # SFT: -sum(logprobs) / 3 = -(-1 + -2 + -3) / 3 = 6 / 3 = 2.0
    assert torch.isclose(loss, torch.tensor(2.0), atol=1e-6)


def test_compute_loss_sequence_mode_uses_weights_no_token_divide():
    """loss_scale_mode='sequence' applies per-sequence weights and skips loss_scale divide."""
    trainer_logprobs = [torch.tensor([-1.0, -2.0]), torch.tensor([-3.0])]
    inference_logprobs = [torch.zeros(2), torch.zeros(1)]
    advantages = [torch.zeros(2), torch.zeros(1)]
    loss_mask = [torch.tensor([True, True]), torch.tensor([True])]

    loss_fn = setup_loss_fn(SFTLossConfig(loss_scale_mode="sequence"))
    assert getattr(loss_fn, "loss_scale_mode") == "sequence"
    loss, _ = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_fn=loss_fn,
        loss_scale=999,  # ignored under "sequence"
        sequence_loss_weights=[0.25, 0.5],
    )
    # SFT per-sample loss: 3.0 (sum of -[-1,-2]) and 3.0 (sum of -[-3]). With weights 0.25, 0.5:
    # total = 0.25*3 + 0.5*3 = 0.75 + 1.5 = 2.25; loss_scale ignored.
    assert torch.isclose(loss, torch.tensor(2.25), atol=1e-6)


def test_compute_loss_rejects_mismatched_weight_count():
    trainer_logprobs = [torch.tensor([-1.0]), torch.tensor([-2.0])]
    loss_fn = setup_loss_fn(SFTLossConfig(loss_scale_mode="sequence"))
    with pytest.raises(ValueError, match="length"):
        compute_loss(
            trainer_logprobs=trainer_logprobs,
            inference_logprobs=[torch.zeros(1), torch.zeros(1)],
            teacher_logprobs=None,
            advantages=[torch.zeros(1), torch.zeros(1)],
            loss_mask=[torch.tensor([True]), torch.tensor([True])],
            loss_fn=loss_fn,
            loss_scale=1,
            sequence_loss_weights=[0.5],  # one too few
        )


def test_cispo_loss_matches_paper_formula_uniform_ratio():
    """When ratio == 1 everywhere, CISPO reduces to -sum(adv * log pi) — clean reference case."""
    inputs = LossInputs(
        trainer_logprobs=torch.tensor([-1.0, -2.0, -3.0]),
        inference_logprobs=torch.tensor([-1.0, -2.0, -3.0]),  # ratio = 1
        teacher_logprobs=None,
        advantages=torch.tensor([0.5, -0.5, 1.0]),
        loss_mask=torch.tensor([True, True, True]),
    )
    out = cispo_loss_fn(inputs, CISPOLossConfig(eps_max=4.0, adv_tau=1.0))
    # min(1, 4) = 1; loss = -sum(0.5*-1 + -0.5*-2 + 1.0*-3) = -(-0.5 + 1 + -3) = -(-2.5) = 2.5
    assert torch.isclose(out.loss, torch.tensor(2.5), atol=1e-6)


def test_cispo_loss_truncates_high_ratio():
    """High ratios get clamped at eps_max — verify the truncation actually fires."""
    inputs = LossInputs(
        # ratio = exp(0 - log(0.01)) = 1/0.01 = 100, way above eps_max=4
        trainer_logprobs=torch.tensor([0.0]),
        inference_logprobs=torch.tensor([math.log(0.01)]),
        teacher_logprobs=None,
        advantages=torch.tensor([1.0]),
        loss_mask=torch.tensor([True]),
    )
    out = cispo_loss_fn(inputs, CISPOLossConfig(eps_max=4.0, adv_tau=1.0))
    # Truncated ratio = 4; loss = -sum(4 * 1.0 * 0.0) = 0.0 (because trainer_logprob is 0).
    assert torch.isclose(out.loss, torch.tensor(0.0), atol=1e-6)
    # The metric should report ratio_truncated = 1.0 (one of one tokens truncated).
    assert torch.isclose(out.metrics["ratio_truncated"], torch.tensor(1.0), atol=1e-6)


def test_cispo_loss_stop_gradient_on_ratio():
    """The IS ratio is detached — gradient flows only through trainer_logprobs."""
    trainer_lp = torch.tensor([-1.0, -2.0], requires_grad=True)
    inference_lp = torch.tensor([0.5, 1.5], requires_grad=True)  # also requires_grad to detect leak
    inputs = LossInputs(
        trainer_logprobs=trainer_lp,
        inference_logprobs=inference_lp,
        teacher_logprobs=None,
        advantages=torch.tensor([1.0, 1.0]),
        loss_mask=torch.tensor([True, True]),
    )
    out = cispo_loss_fn(inputs, CISPOLossConfig(eps_max=4.0))
    out.loss.backward()
    # trainer side has gradient; inference side does NOT (stop-gradient).
    assert trainer_lp.grad is not None
    assert torch.all(trainer_lp.grad != 0)
    assert inference_lp.grad is None or torch.all(inference_lp.grad == 0)


def test_cispo_loss_default_scale_mode_is_sequence():
    config = CISPOLossConfig()
    loss_fn = setup_loss_fn(config)
    assert getattr(loss_fn, "loss_scale_mode") == "sequence"


def test_setup_loss_fn_attaches_loss_scale_mode_to_all_paths():
    for cfg in [
        DefaultLossConfig(),
        DefaultLossConfig(loss_scale_mode="sequence"),
        SFTLossConfig(),
        SFTLossConfig(loss_scale_mode="none"),
        CISPOLossConfig(),
        CISPOLossConfig(loss_scale_mode="token"),
    ]:
        loss_fn = setup_loss_fn(cfg)
        assert hasattr(loss_fn, "loss_scale_mode")
        assert getattr(loss_fn, "loss_scale_mode") == cfg.loss_scale_mode


def _identity_custom_loss(inputs: LossInputs, **_: object) -> LossOutputs:
    return LossOutputs(loss=torch.tensor(1.0), metrics={})


def test_pad_micro_batch_extends_sequence_loss_weights_for_phantom_split():
    """Padding restarts position_ids at 0, which get_response_lengths reads as a new
    sequence. We must add a [0.0] weight so length matches num_packed and the phantom
    contributes nothing under loss_scale_mode='sequence'/'none'.
    """
    from prime_rl.trainer.batch import pad_micro_batch
    from prime_rl.transport.types import MicroBatch

    mb = MicroBatch(
        input_ids=[1, 2, 3, 4, 5],
        loss_mask=[True, True, True, True, True],
        advantages=[1.0, 1.0, 1.0, 1.0, 1.0],
        inference_logprobs=[0.0] * 5,
        position_ids=[0, 1, 2, 3, 4],
        temperatures=[1.0] * 5,
        sequence_loss_weights=[0.5],
        lora_num_tokens=[5],
    )
    pad_micro_batch(mb, pad_to_multiple_of=8)
    # Padding adds 3 tokens with position_ids=[0,1,2] — a phantom sequence boundary.
    assert mb.sequence_loss_weights == [0.5, 0.0]
    assert len(mb.input_ids) == 8


def test_pad_micro_batch_padding_size_one_does_not_create_phantom_sequence():
    """When `padding_size == 1`, the trailing single `0` is folded by
    get_response_lengths into the previous sequence (it only treats `0` as a
    boundary when the NEXT position_id is 1). Appending a phantom weight in
    that case would produce a length-mismatch crash in compute_loss.
    """
    from prime_rl.trainer.batch import pad_micro_batch
    from prime_rl.trainer.utils import get_response_lengths
    from prime_rl.transport.types import MicroBatch

    mb = MicroBatch(
        input_ids=[1, 2, 3, 4, 5, 6, 7],
        loss_mask=[True] * 7,
        advantages=[1.0] * 7,
        inference_logprobs=[0.0] * 7,
        position_ids=[0, 1, 2, 3, 4, 5, 6],
        temperatures=[1.0] * 7,
        sequence_loss_weights=[0.5],
        lora_num_tokens=[7],
    )
    pad_micro_batch(mb, pad_to_multiple_of=8)
    # padding_size = 1 → position_ids end with [..., 6, 0]. The trailing 0 is folded
    # into the last sequence by get_response_lengths.
    assert len(mb.input_ids) == 8
    assert mb.position_ids[-1] == 0
    num_packed = len(get_response_lengths(torch.tensor(mb.position_ids)))
    assert num_packed == 1, f"padding_size=1 must not create a phantom sequence; got {num_packed}"
    assert len(mb.sequence_loss_weights) == num_packed, (
        "sequence_loss_weights length must match num_packed_sequences for compute_loss to accept it"
    )


def test_pad_micro_batch_packed_two_samples_then_padding_size_one():
    """Two real packed sequences plus padding_size=1: phantom-detection still works.
    The intra-sequence-2 boundary at position_ids[3]==0 is detected because
    position_ids[4]==1; the trailing pad `0` is folded into seq 2.
    """
    from prime_rl.trainer.batch import pad_micro_batch
    from prime_rl.trainer.utils import get_response_lengths
    from prime_rl.transport.types import MicroBatch

    mb = MicroBatch(
        # Two packed sequences of length 3 and 4: positions [0,1,2, 0,1,2,3] (len 7)
        input_ids=[1, 2, 3, 4, 5, 6, 7],
        loss_mask=[True] * 7,
        advantages=[1.0] * 7,
        inference_logprobs=[0.0] * 7,
        position_ids=[0, 1, 2, 0, 1, 2, 3],
        temperatures=[1.0] * 7,
        sequence_loss_weights=[0.3, 0.7],
        lora_num_tokens=[7],
    )
    pad_micro_batch(mb, pad_to_multiple_of=8)
    assert len(mb.input_ids) == 8
    num_packed = len(get_response_lengths(torch.tensor(mb.position_ids)))
    assert num_packed == 2
    assert len(mb.sequence_loss_weights) == num_packed


def test_setup_loss_fn_custom_loss_attaches_loss_scale_mode():
    cfg = CustomLossConfig(
        import_path="tests.unit.train.rl.test_loss_weighting._identity_custom_loss",
        loss_scale_mode="none",
    )
    loss_fn = setup_loss_fn(cfg)
    assert getattr(loss_fn, "loss_scale_mode") == "none"
