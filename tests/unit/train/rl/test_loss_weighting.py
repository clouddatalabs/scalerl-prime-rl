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
    )


def test_apply_prompt_average_sequence_weights_balances_prompts():
    """Weights make every prompt contribute equally and every token-within-prompt contribute equally."""
    rollouts = [
        _make_sample(example_id="A", completion_len=10),
        _make_sample(example_id="A", completion_len=30),
        _make_sample(example_id="B", completion_len=20),
        _make_sample(example_id="B", completion_len=20),
    ]
    weights = apply_prompt_average_sequence_weights(rollouts, seq_len=4096)
    assert weights is not None
    # Prompt A: total 40 tokens; weights = 10/40/2 = 0.125 and 30/40/2 = 0.375.
    # Prompt B: total 40 tokens; weights = 20/40/2 = 0.25 and 20/40/2 = 0.25.
    assert weights == pytest.approx([0.125, 0.375, 0.25, 0.25])
    # Sum within each prompt = 0.5; total over batch = 1.0 (i.e. average over prompts).
    assert sum(weights[:2]) == pytest.approx(0.5)
    assert sum(weights[2:]) == pytest.approx(0.5)
    assert sum(weights) == pytest.approx(1.0)


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
    )
    sample2 = _make_sample(example_id="A", completion_len=4)  # 4 trainable tokens too
    weights = apply_prompt_average_sequence_weights([sample, sample2], seq_len=4096)
    # Both samples have 4 trainable tokens; prompt A total = 8; num_prompts = 1.
    # Each weight = 4 / (8 * 1) = 0.5. Mass-on-loss-mask=False would give 6/(6+4)/1 ≠ 0.5.
    assert weights == pytest.approx([0.5, 0.5])


def test_apply_prompt_average_sequence_weights_excludes_truncated_tokens():
    """seq_len truncation drops tail tokens — they must not count toward the weight."""
    # prompt_len=3, completion_len=20. At seq_len=10 only 10 - 3 = 7 completion tokens
    # survive prepare_sample's truncation.
    sample = _make_sample(example_id="A", completion_len=20)
    weights = apply_prompt_average_sequence_weights([sample], seq_len=10)
    # 7 trainable tokens / (7 * 1 prompt) = 1.0
    assert weights == pytest.approx([1.0])

    # If prompt itself is at-or-over seq_len, no completion tokens are trainable; guard.
    sample_no_completion = _make_sample(example_id="A", completion_len=5)
    weights = apply_prompt_average_sequence_weights([sample_no_completion], seq_len=2)
    # 0 trainable tokens for the only sample; total guarded to 1 → weight is 0.
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


def test_setup_loss_fn_custom_loss_attaches_loss_scale_mode():
    cfg = CustomLossConfig(
        import_path="tests.unit.train.rl.test_loss_weighting._identity_custom_loss",
        loss_scale_mode="none",
    )
    loss_fn = setup_loss_fn(cfg)
    assert getattr(loss_fn, "loss_scale_mode") == "none"
