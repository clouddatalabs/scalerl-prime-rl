"""Pin _validate_orch_lora_against_trainer's contract.

Multi-run mode launches each orchestrator independently — the trainer's
master rank then re-parses the orchestrator TOMLs via
`OrchestratorConfig(**dict)`, which does NOT run `RLConfig.auto_setup_lora`.
So `model.lora` can legitimately be None on the parsed config, and the
multi-run config-validation hook must surface that as a clean
(False, message) tuple rather than crash with AttributeError on the
master rank.
"""

import pytest

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.trainer import LoRAConfig
from prime_rl.trainer.runs import _validate_orch_lora_against_trainer


def _trainer_lora(rank=8, alpha=16) -> LoRAConfig:
    return LoRAConfig(rank=rank, alpha=alpha)


def _orch_config(lora_kwargs=None):
    payload = {"model": {}}
    if lora_kwargs is not None:
        payload["model"]["lora"] = lora_kwargs
    return OrchestratorConfig.model_validate(payload)


def test_missing_lora_section_returns_clean_validation_failure():
    """The original bug: orchestrator config without [model.lora] raised
    AttributeError on the trainer's master rank. The validator must now
    surface that as the standard (False, message) contract."""
    cfg = _orch_config(lora_kwargs=None)
    assert cfg.model.lora is None, "fixture must reproduce the failure mode"
    ok, msg = _validate_orch_lora_against_trainer(cfg, _trainer_lora())
    assert ok is False
    assert "[model.lora]" in msg


def test_lora_rank_alpha_inherited_from_trainer_when_omitted():
    """Orchestrator TOMLs may omit rank/alpha to inherit the trainer's values."""
    cfg = _orch_config(lora_kwargs={})
    assert cfg.model.lora is not None
    assert cfg.model.lora.rank is None
    assert cfg.model.lora.alpha is None
    ok, msg = _validate_orch_lora_against_trainer(cfg, _trainer_lora(rank=8, alpha=16))
    assert ok, msg
    # Validator mutates in place to fill in the inherited values.
    assert cfg.model.lora.rank == 8
    assert cfg.model.lora.alpha == 16


def test_lora_rank_exceeding_trainer_max_rejected():
    """rank > trainer's rank is rejected — LoRA matrices are sized at trainer's
    max rank; an orchestrator requesting more couldn't be served."""
    cfg = _orch_config(lora_kwargs={"rank": 16, "alpha": 32})
    ok, msg = _validate_orch_lora_against_trainer(cfg, _trainer_lora(rank=8, alpha=16))
    assert ok is False
    assert "exceeds trainer max rank" in msg


def test_lora_rank_equal_or_lower_than_trainer_accepted():
    cfg = _orch_config(lora_kwargs={"rank": 4, "alpha": 8})
    ok, msg = _validate_orch_lora_against_trainer(cfg, _trainer_lora(rank=8, alpha=16))
    assert ok, msg
    assert cfg.model.lora.rank == 4
    assert cfg.model.lora.alpha == 8


def test_validator_does_not_crash_on_none_lora():
    """Regression guard: pre-fix, the validator dereferenced .rank without a
    None check and raised AttributeError. The new code must return
    (False, message) cleanly — not raise.
    """
    cfg = _orch_config(lora_kwargs=None)
    # Should not raise.
    result = _validate_orch_lora_against_trainer(cfg, _trainer_lora())
    assert isinstance(result, tuple) and len(result) == 2
    assert result[0] is False
