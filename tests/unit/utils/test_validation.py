"""Pin `validate_shared_weight_broadcast`'s contract.

The chained `a != b != c` form Python supports as `(a != b) and (b != c)`
silently lets a triplet like (filesystem, filesystem, nccl) pass — the FIRST
inequality is False, so the conjunction short-circuits and never evaluates
the second. The result is a configuration where the trainer writes weights
to disk while inference connects via NCCL and the orchestrator hangs at the
first weight broadcast (or vLLM crashes on init). The fix uses an explicit
3-element set comparison; pin both the rejection of every disagreement
permutation and the acceptance of all-equal cases.
"""

import pytest

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.configs.inference import InferenceConfig
from prime_rl.utils.validation import validate_shared_weight_broadcast


def _trainer_with(broadcast_type: str) -> TrainerConfig:
    return TrainerConfig.model_validate({"weight_broadcast": {"type": broadcast_type}})


def _orchestrator_with(broadcast_type: str) -> OrchestratorConfig:
    return OrchestratorConfig.model_validate({"weight_broadcast": {"type": broadcast_type}})


def _inference_with(broadcast_type: str) -> InferenceConfig:
    return InferenceConfig.model_validate({"weight_broadcast": {"type": broadcast_type}})


@pytest.mark.parametrize(
    "trainer_t,orch_t,infer_t",
    [
        # The exact case the chained-`!=` bug let through:
        #   (a != b) and (b != c) = False and True = False
        ("filesystem", "filesystem", "nccl"),
        # Mirror image — first inequality holds, second doesn't:
        #   (a != b) and (b != c) = True and False = False
        ("nccl", "filesystem", "filesystem"),
        # All three different — short-circuits but should also reject.
        ("filesystem", "nccl", "filesystem"),
    ],
)
def test_three_way_disagreement_is_rejected(trainer_t, orch_t, infer_t):
    """Any disagreement among the three components must raise — including
    the (X, Y, X) and (X, X, Y) patterns that the original chained-`!=`
    silently accepted."""
    with pytest.raises(ValueError, match="disagree across components"):
        validate_shared_weight_broadcast(
            _trainer_with(trainer_t),
            _orchestrator_with(orch_t),
            _inference_with(infer_t),
        )


@pytest.mark.parametrize("broadcast_type", ["filesystem", "nccl"])
def test_all_equal_is_accepted(broadcast_type):
    """All-equal: no raise (orchestrator/trainer/inference all use the same path)."""
    validate_shared_weight_broadcast(
        _trainer_with(broadcast_type),
        _orchestrator_with(broadcast_type),
        _inference_with(broadcast_type),
    )


def test_two_way_disagreement_without_inference_is_rejected():
    """Trainer != orchestrator without inference: also raises (legacy path)."""
    with pytest.raises(ValueError, match="not the same"):
        validate_shared_weight_broadcast(
            _trainer_with("filesystem"),
            _orchestrator_with("nccl"),
            inference=None,
        )


def test_two_way_agreement_without_inference_is_accepted():
    """No-op when only trainer + orchestrator are present and they agree."""
    validate_shared_weight_broadcast(
        _trainer_with("filesystem"),
        _orchestrator_with("filesystem"),
        inference=None,
    )
