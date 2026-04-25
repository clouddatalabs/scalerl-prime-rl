from pathlib import Path
from typing import Annotated, Literal

import pytest
import tomli_w
from pydantic import BaseModel, Field, ValidationError
from pydantic_config import ConfigFileError

from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.configs.sft import SFTConfig
from prime_rl.configs.trainer import ModelConfig as TrainerModelConfig
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.utils.config import BaseConfig, cli

# All config config classes
CONFIG_CLASSES = [
    RLConfig,
    TrainerConfig,
    SFTConfig,
    OrchestratorConfig,
    InferenceConfig,
]


_REPO_ROOT = Path(__file__).resolve().parents[2]


def get_config_files() -> list[Path]:
    """Any TOML file inside `configs/` or `examples/`.

    Anchor at the repo root (NOT pytest's cwd) so the parametrize list is
    non-empty regardless of where pytest runs from. A cwd-relative glob
    silently produced an empty list when run from any other directory,
    making this entire `test_load_configs` slice a no-op green pass.
    """
    config_files = list((_REPO_ROOT / "configs").rglob("*.toml"))
    example_files = list((_REPO_ROOT / "examples").rglob("*.toml"))

    files = config_files + example_files
    if not files:
        raise RuntimeError(
            f"No TOML configs found under {_REPO_ROOT}/configs or "
            f"{_REPO_ROOT}/examples — refusing to run a parametrize=0 'green'."
        )
    return files


def _expected_class_for(config_file: Path):
    """Best-fit expected schema by directory name. Falls back to the loose
    any-of contract for paths we don't recognize, but the canonical ScaleRL
    configs are pinned strictly so a typo that breaks RLConfig but parses
    as OrchestratorConfig (a subset) cannot slip through."""
    rel = config_file.relative_to(_REPO_ROOT)
    parts = rel.parts
    if "scalerl_math" in parts or "scalerl_terminal_bench" in parts:
        return RLConfig
    # Upstream debug/example configs are heterogeneous — any of the five.
    return None


# Configs that ship placeholder values for the operator to fill in
# (e.g. `[slurm] partition = "your-gpu-partition"`). The
# `SlurmConfig.reject_placeholder_values` validator rejects these by
# design — that's the load-time fail-loud the validator exists for. The
# test harness must NOT treat that rejection as a regression; instead,
# it asserts the expected ValidationError fires AND that filling in the
# placeholders produces a loadable config.
_TEMPLATE_CONFIGS: dict[Path, dict] = {
    _REPO_ROOT / "configs/scalerl_terminal_bench/rl_multinode.toml": {
        "slurm": {"partition": "compute", "account": "myaccount"}
    },
}


@pytest.mark.parametrize("config_file", get_config_files(), ids=lambda x: x.as_posix())
def test_load_configs(config_file: Path):
    """Tests that all config files can be loaded by their expected class
    (when known) or at least one of the five (for upstream debug configs).
    """
    expected = _expected_class_for(config_file)
    if expected is not None:
        if config_file in _TEMPLATE_CONFIGS:
            # Template configs ship placeholder values that
            # `SlurmConfig.reject_placeholder_values` REJECTS BY DESIGN —
            # that's the load-time fail-loud the validator exists for.
            # Verify (a) the rejection actually fires, and (b) filling in
            # the placeholders produces a loadable config. Catches both
            # "operator forgot to edit" and "validator regression that
            # silently lets placeholders through".
            import tomli as _tomli

            with open(config_file, "rb") as f:
                data = _tomli.load(f)
            with pytest.raises(ValidationError, match="placeholder"):
                expected.model_validate(data)
            # Patch placeholders with realistic values; the patched
            # config must validate cleanly. Mutate-merge: only override
            # the keys we list.
            patched = data
            for section, fields in _TEMPLATE_CONFIGS[config_file].items():
                patched.setdefault(section, {}).update(fields)
            expected.model_validate(patched)
            return
        # Strict pin: typo in a ScaleRL config that shifts the parse from
        # RLConfig to a subset class would silently pass under any-of-five.
        cli(expected, args=["@", config_file.as_posix()])
        return

    could_parse = []
    for config_cls in CONFIG_CLASSES:
        try:
            cli(config_cls, args=["@", config_file.as_posix()])
            could_parse.append(True)
        except (ValidationError, ConfigFileError, SystemExit):
            could_parse.append(False)
    assert any(could_parse), f"No config class could be parsed from {config_file}"


class NestedConfig(BaseConfig):
    lr: float = 1e-4
    weight_decay: float = 0.01
    name: str = "default"


class VariantA(BaseModel):
    type: Literal["a"] = "a"
    alpha: float = 0.1
    shared: int = 1


class VariantB(BaseModel):
    type: Literal["b"] = "b"
    beta: float = 0.2
    shared: int = 1


VariantType = Annotated[VariantA | VariantB, Field(discriminator="type")]


class DummyConfig(BaseConfig):
    name: str = "experiment"
    seed: int = 42
    nested: NestedConfig = NestedConfig()
    variant: VariantType = VariantA()


def write_toml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def test_defaults():
    """All defaults are applied when no TOML or CLI args are given."""
    config = cli(DummyConfig, args=[])
    assert config.name == "experiment"
    assert config.seed == 42
    assert config.nested.lr == 1e-4
    assert config.nested.weight_decay == 0.01
    assert config.variant.type == "a"
    assert config.variant.alpha == 0.1


def test_toml_partial_nested_override(tmp_path):
    """Partially overriding a nested model preserves unset field defaults."""
    write_toml(tmp_path / "cfg.toml", {"nested": {"lr": 3e-4}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml")])
    assert config.nested.lr == 3e-4
    assert config.nested.weight_decay == 0.01
    assert config.nested.name == "default"


def test_toml_discriminated_union_default_type(tmp_path):
    """Overriding a discriminated union field without 'type' uses the default variant."""
    write_toml(tmp_path / "cfg.toml", {"variant": {"alpha": 0.9}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml")])
    assert config.variant.type == "a"
    assert config.variant.alpha == 0.9
    assert config.variant.shared == 1


def test_toml_discriminated_union_switch_variant(tmp_path):
    """Providing an explicit 'type' switches to that variant."""
    write_toml(tmp_path / "cfg.toml", {"variant": {"type": "b"}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml")])
    assert config.variant.type == "b"
    assert config.variant.beta == 0.2


def test_toml_discriminated_union_override_switch_variant(tmp_path):
    """Providing an explicit 'type' overrides the default variant."""
    write_toml(tmp_path / "cfg.toml", {"variant": {"type": "b", "beta": 0.5}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml")])
    assert config.variant.type == "b"
    assert config.variant.beta == 0.5


def test_cli_overrides_defaults():
    """CLI args override defaults."""
    config = cli(DummyConfig, args=["--name", "my-run", "--seed", "7"])
    assert config.name == "my-run"
    assert config.seed == 7
    assert config.nested.lr == 1e-4


def test_toml_overrides_defaults(tmp_path):
    """TOML overrides defaults."""
    write_toml(tmp_path / "cfg.toml", {"name": "my-run", "seed": 7, "nested": {"lr": 3e-4}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml")])
    assert config.name == "my-run"
    assert config.seed == 7
    assert config.nested.lr == 3e-4


def test_cli_overrides_toml(tmp_path):
    """CLI args override TOML."""
    write_toml(tmp_path / "cfg.toml", {"seed": 1, "nested": {"lr": 3e-4}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml"), "--seed", "99", "--nested.lr", "5e-5"])
    assert config.seed == 99
    assert config.nested.lr == 5e-5
    # TOML value not overridden by CLI should still be applied (not reverted to class default)
    assert config.nested.weight_decay == 0.01


def test_removed_fused_lm_head_chunk_size_field_is_rejected():
    with pytest.raises(ValidationError, match="fused_lm_head_chunk_size"):
        TrainerModelConfig.model_validate({"fused_lm_head_chunk_size": "auto"})


def test_selective_activation_checkpointing_requires_custom_impl():
    with pytest.raises(ValidationError, match="Selective activation checkpointing requires model.impl='custom'"):
        TrainerModelConfig.model_validate({"impl": "hf", "ac": {"mode": "selective"}})


def test_nccl_async_level_requires_explicit_opt_in():
    """NCCL + max_async_level != 1 must raise unless the experimental override is set."""
    with pytest.raises(ValidationError, match="allow_nccl_async_level_override"):
        OrchestratorConfig.model_validate(
            {"weight_broadcast": {"type": "nccl"}, "max_async_level": 8}
        )


def test_nccl_async_level_opt_in_allows_override():
    """With experimental.allow_nccl_async_level_override = true, NCCL + async > 1 validates."""
    config = OrchestratorConfig.model_validate(
        {
            "weight_broadcast": {"type": "nccl"},
            "max_async_level": 8,
            "experimental": {"allow_nccl_async_level_override": True},
        }
    )
    assert config.max_async_level == 8
    assert config.experimental.allow_nccl_async_level_override is True


def test_nccl_async_level_default_one_still_validates():
    """The default NCCL + max_async_level=1 path still works without the override."""
    config = OrchestratorConfig.model_validate(
        {"weight_broadcast": {"type": "nccl"}, "max_async_level": 1}
    )
    assert config.experimental.allow_nccl_async_level_override is False


def test_paper_faithful_empty_batch_defaults_off():
    """Default keeps the engineering retry/crash guardrail."""
    config = OrchestratorConfig.model_validate({})
    assert config.experimental.paper_faithful_empty_batch is False


def test_paper_faithful_empty_batch_opt_in():
    config = OrchestratorConfig.model_validate(
        {"experimental": {"paper_faithful_empty_batch": True}}
    )
    assert config.experimental.paper_faithful_empty_batch is True


_RL_BASE = {
    # `model.name` must be explicitly set so the new "no implicit
    # Qwen/Qwen3-0.6B fallback" validator (auto_setup_model in rl.py)
    # accepts these test fixtures. Use the same default model the
    # individual ModelConfigs default to so the test contract remains
    # equivalent to "construct with all defaults" — only the explicit-set
    # bit changes.
    "model": {"name": "Qwen/Qwen3-0.6B"},
    "trainer": {},
    "orchestrator": {},
    "inference": {},
}


def test_rl_config_propagates_nccl_async_override_from_orchestrator_to_trainer():
    """Setting the override on the orchestrator side at the RL level propagates to the trainer.

    Without this propagation, the RLConfig-level max_async_level=8 + NCCL combo would still
    be rejected by TrainerConfig's own validator.
    """
    config = RLConfig.model_validate(
        {
            **_RL_BASE,
            "max_async_level": 8,
            "weight_broadcast": {"type": "nccl"},
            "orchestrator": {"experimental": {"allow_nccl_async_level_override": True}},
        }
    )
    assert config.trainer.experimental.allow_nccl_async_level_override is True
    assert config.orchestrator.experimental.allow_nccl_async_level_override is True
    assert config.trainer.max_async_level == 8
    assert config.orchestrator.max_async_level == 8


def test_rl_config_rejects_mismatched_fp32_lm_head():
    """Trainer fp32=True + inference fp32=False raises at config load.

    Mismatch defeats the train/inference logprob parity that fp32_lm_head exists for
    (ScaleRL §3.2).
    """
    with pytest.raises(ValidationError, match="fp32_lm_head must match"):
        RLConfig.model_validate(
            {
                **_RL_BASE,
                "trainer": {"model": {"fp32_lm_head": True}, "matmul_precision": "highest"},
                "inference": {"model": {"fp32_lm_head": False}},
            }
        )


def test_rl_config_accepts_matched_fp32_lm_head():
    """fp32_lm_head=True on both sides validates (with matmul_precision=highest)."""
    config = RLConfig.model_validate(
        {
            **_RL_BASE,
            "trainer": {"model": {"fp32_lm_head": True}, "matmul_precision": "highest"},
            "inference": {"model": {"fp32_lm_head": True}},
        }
    )
    assert config.trainer.model.fp32_lm_head is True
    assert config.inference.model.fp32_lm_head is True


def test_rl_config_rejects_fp32_lm_head_with_high_matmul_precision():
    """fp32_lm_head=True silently runs FP32 matmuls in TF32 unless
    matmul_precision='highest'. The default 'high' setting defeats the §3.2
    train/inference logprob parity that fp32_lm_head exists for."""
    with pytest.raises(ValidationError, match="matmul_precision='highest'"):
        RLConfig.model_validate(
            {
                **_RL_BASE,
                "trainer": {"model": {"fp32_lm_head": True}, "matmul_precision": "high"},
                "inference": {"model": {"fp32_lm_head": True}},
            }
        )


def test_rl_config_rejects_fp32_lm_head_with_default_matmul_precision():
    """The trainer default for matmul_precision is 'high'; fp32_lm_head=True
    without explicit override is the most likely footgun and must fail loudly."""
    with pytest.raises(ValidationError, match="matmul_precision='highest'"):
        RLConfig.model_validate(
            {
                **_RL_BASE,
                "trainer": {"model": {"fp32_lm_head": True}},
                "inference": {"model": {"fp32_lm_head": True}},
            }
        )


def test_trainer_config_rejects_fp32_lm_head_with_default_matmul_precision():
    """TrainerConfig validates standalone for direct cli(TrainerConfig) callers
    (the RL trainer subprocess, custom test fixtures). Belt-and-suspenders to
    the RLConfig-level cross-validator. (SFT goes through SFTConfig, which
    has its own validator below.)"""
    with pytest.raises(ValidationError, match="matmul_precision='highest'"):
        TrainerConfig.model_validate(
            {"model": {"fp32_lm_head": True}}
        )


def test_trainer_config_accepts_fp32_lm_head_with_highest_matmul_precision():
    config = TrainerConfig.model_validate(
        {"model": {"fp32_lm_head": True}, "matmul_precision": "highest"}
    )
    assert config.model.fp32_lm_head is True
    assert config.matmul_precision == "highest"


def test_trainer_config_no_op_when_fp32_lm_head_off():
    """fp32_lm_head=False with default matmul_precision='high' must validate."""
    config = TrainerConfig.model_validate({"model": {"fp32_lm_head": False}})
    assert config.matmul_precision == "high"


# SFTConfig is a SIBLING to TrainerConfig (it does NOT inherit), so the
# validator must be attached to both classes independently. The SFT entry
# point is `cli(SFTConfig)` in `prime_rl/trainer/sft/train.py`.

def test_sft_config_rejects_fp32_lm_head_with_default_matmul_precision():
    """SFT path: same TF32 footgun as RL/Trainer. The SFTConfig validator
    must fire on `[model] fp32_lm_head = true` with the default
    `matmul_precision = "high"`.
    """
    with pytest.raises(ValidationError, match="matmul_precision='highest'"):
        SFTConfig.model_validate({"model": {"fp32_lm_head": True}})


def test_sft_config_accepts_fp32_lm_head_with_highest_matmul_precision():
    config = SFTConfig.model_validate(
        {"model": {"fp32_lm_head": True}, "matmul_precision": "highest"}
    )
    assert config.model.fp32_lm_head is True
    assert config.matmul_precision == "highest"


def test_sft_config_no_op_when_fp32_lm_head_off():
    config = SFTConfig.model_validate({"model": {"fp32_lm_head": False}})
    assert config.matmul_precision == "high"


def test_rl_config_skips_fp32_lm_head_check_when_inference_omitted():
    """RLConfig.inference may legitimately be None (externally managed inference pools or
    num_infer_nodes=0 fake-data runs). The fp32 consistency check must skip rather than
    AttributeError, but it MUST also warn loudly when fp32_lm_head=True since
    setup_vllm_env does not run on an externally launched vLLM.
    """
    import warnings

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        config = RLConfig.model_validate(
            {
                "model": {"name": "Qwen/Qwen3-0.6B"},
                "trainer": {"model": {"fp32_lm_head": True}, "matmul_precision": "highest"},
                "orchestrator": {},
                "inference": None,
            }
        )
        assert config.inference is None
        assert config.trainer.model.fp32_lm_head is True
        # Warn loudly: external vLLM won't pick up fp32_lm_head from the trainer config.
        msgs = [str(w.message) for w in captured]
        assert any("PRIME_RL_VLLM_FP32_LM_HEAD" in m for m in msgs), msgs


def test_rl_config_no_warning_when_inference_omitted_and_fp32_off():
    """The mismatch warning must not fire when trainer.fp32_lm_head=False — there
    is no parity hazard if neither side wants the fp32 path.
    """
    import warnings

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        RLConfig.model_validate(
            {
                "model": {"name": "Qwen/Qwen3-0.6B"},
                "trainer": {"model": {"fp32_lm_head": False}},
                "orchestrator": {},
                "inference": None,
            }
        )
        msgs = [str(w.message) for w in captured]
        assert not any("PRIME_RL_VLLM_FP32_LM_HEAD" in m for m in msgs), msgs


def test_rl_config_rejects_prompt_average_loss_with_token_scale_mode():
    """orchestrator.prompt_average_loss=True without sequence/none scale mode would
    silently train under token-mean reduction — the packer's per-sequence weights are
    discarded by compute_loss when loss_scale_mode='token'."""
    with pytest.raises(ValidationError, match="prompt_average_loss=true requires"):
        RLConfig.model_validate(
            {
                **_RL_BASE,
                "orchestrator": {"prompt_average_loss": True},
                "trainer": {"loss": {"type": "default", "loss_scale_mode": "token"}},
            }
        )


def test_rl_config_accepts_prompt_average_loss_with_sequence_scale_mode():
    """The same combo with sequence-mode loss validates."""
    config = RLConfig.model_validate(
        {
            **_RL_BASE,
            "orchestrator": {"prompt_average_loss": True},
            "trainer": {"loss": {"type": "cispo", "loss_scale_mode": "sequence"}},
        }
    )
    assert config.orchestrator.prompt_average_loss is True
    assert config.trainer.loss.loss_scale_mode == "sequence"


def test_rl_config_rejects_cispo_with_teacher_model():
    """CISPO does not consume teacher logprobs — configuring a teacher with
    `loss.type='cispo'` would silently pay the teacher-prefill compute cost."""
    with pytest.raises(ValidationError, match="cispo.*teacher_model"):
        RLConfig.model_validate(
            {
                **_RL_BASE,
                "orchestrator": {
                    "prompt_average_loss": True,
                    "teacher_model": {"client": {}, "model": {}},
                },
                "trainer": {"loss": {"type": "cispo", "loss_scale_mode": "sequence"}},
            }
        )


def test_rl_config_rejects_sft_with_teacher_model():
    """SFT does not consume teacher logprobs either — same footgun as CISPO."""
    with pytest.raises(ValidationError, match="sft.*teacher_model"):
        RLConfig.model_validate(
            {
                **_RL_BASE,
                "orchestrator": {
                    "teacher_model": {"client": {}, "model": {}},
                },
                "trainer": {"loss": {"type": "sft", "loss_scale_mode": "token"}},
            }
        )


def test_rl_config_rejects_cispo_with_external_rollout_model():
    """`validate_external_rollout_mode` rejects any non-SFT loss with
    `teacher_rollout_model` set. CISPO + external rollout would also hit
    the silent-zero-logprobs path; this test pins the rejection.
    """
    with pytest.raises(ValidationError, match="teacher_rollout_model.*sft"):
        RLConfig.model_validate(
            {
                "orchestrator": {
                    "prompt_average_loss": True,
                    "use_token_client": False,
                    "teacher_rollout_model": {"client": {}, "model": {}},
                },
                "trainer": {"loss": {"type": "cispo", "loss_scale_mode": "sequence"}},
                "inference": None,
            }
        )


def test_rl_config_accepts_sft_with_external_rollout_string_client():
    """SFT does not consume inference_logprobs, so it's safe even with the
    string-client + external-rollout reconstruction path."""
    config = RLConfig.model_validate(
        {
            "model": {"name": "Qwen/Qwen3-0.6B"},
            "orchestrator": {
                "use_token_client": False,
                "teacher_rollout_model": {"client": {}, "model": {}},
            },
            "trainer": {"loss": {"type": "sft", "loss_scale_mode": "token"}},
            "inference": None,
        }
    )
    assert config.trainer.loss.type == "sft"


def test_rl_config_rejects_fp32_lm_head_with_fp8_weight_transfer():
    """fp32_lm_head + NCCL FP8 weight transfer is self-defeating: the
    LM-head weights would be quantized to FP8 before vLLM's promote-to-fp32
    path runs, baking in the noise the fp32 LM-head exists to prevent.
    """
    with pytest.raises(ValidationError, match="fp32_lm_head"):
        RLConfig.model_validate(
            {
                **_RL_BASE,
                "weight_broadcast": {
                    "type": "nccl",
                    "quantize_in_weight_transfer": True,
                },
                "trainer": {
                    "model": {"fp32_lm_head": True, "impl": "custom"},
                    "matmul_precision": "highest",
                },
                "inference": {"model": {"fp32_lm_head": True}},
            }
        )


def test_adamw_config_eps_bounds():
    """Tightened from `gt=0` to `Field(ge=1e-30, le=1e-3)` to reject
    sub-fp32-representable values that would silently train at effectively
    zero epsilon. Pin every boundary so a regression to `gt=0` fails."""
    from prime_rl.configs.trainer import AdamWConfig

    AdamWConfig(eps=1e-30)  # boundary in
    AdamWConfig(eps=1e-3)  # boundary in
    AdamWConfig(eps=1e-15)  # ScaleRL paper value
    with pytest.raises(ValidationError):
        AdamWConfig(eps=1e-50)  # below ge=1e-30
    with pytest.raises(ValidationError):
        AdamWConfig(eps=1e-2)  # above le=1e-3


def test_cispo_adv_tau_rejects_zero():
    """`adv_tau=0` zeros every per-token contribution to the CISPO loss
    (-sg(min(rho,eps_max)) * adv_tau * adv * log_pi). With no teacher / KL
    fallback path in CISPO, the loss is identically 0 and gradients vanish
    silently. Reject at config load."""
    from prime_rl.configs.trainer import CISPOLossConfig

    CISPOLossConfig(adv_tau=1e-6)  # in range
    CISPOLossConfig(adv_tau=1.0)  # default
    with pytest.raises(ValidationError):
        CISPOLossConfig(adv_tau=0.0)


def test_cispo_eps_max_bounds():
    """`eps_max < 1` would clamp the on-policy mode (rho≈1) to a sub-1
    coefficient and silently zero the gradient. Reject at config load."""
    from prime_rl.configs.trainer import CISPOLossConfig

    CISPOLossConfig(eps_max=1.0)  # boundary in
    CISPOLossConfig(eps_max=4.0)  # paper default
    CISPOLossConfig(eps_max=64.0)  # boundary in
    with pytest.raises(ValidationError):
        CISPOLossConfig(eps_max=0.5)  # below ge=1.0 — silent gradient zero
    with pytest.raises(ValidationError):
        CISPOLossConfig(eps_max=128.0)  # above le=64.0


def test_shipped_poc_configs_load_cleanly():
    """Both shipped POC configs must validate AND pin every paper-faithful
    knob the recipe depends on. A typo (`use_token_clinet = false`) or a
    deleted DEVIATION override (`max_off_policy_steps`, `prompt_average_loss`,
    etc.) would otherwise only surface at launch time, hours into a SLURM
    allocation. Each assertion below corresponds to one of the seven
    ScaleRL ingredients (arXiv:2510.13786) — when the recipe is the
    contract, the paper's settings must be the test contract too.
    """
    import tomli
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    for relpath in ["configs/scalerl_math/rl.toml", "configs/scalerl_terminal_bench/rl.toml"]:
        config_path = repo_root / relpath
        with open(config_path, "rb") as f:
            data = tomli.loads(f.read().decode())
        cfg = RLConfig.model_validate(data)

        # Optimizer / scheduler — paper §3.3.
        assert cfg.trainer.optim.eps == 1e-15, f"{relpath}: AdamW eps"
        assert cfg.trainer.optim.lr == 5e-7, f"{relpath}: paper LR §3.3"
        assert cfg.trainer.optim.weight_decay == 0.01, f"{relpath}: paper weight_decay §3.3"
        assert cfg.trainer.scheduler.type == "linear"
        assert cfg.trainer.scheduler.warmup_steps == 100

        # CISPO loss — paper §3.3.
        assert cfg.trainer.loss.type == "cispo", f"{relpath}: ScaleRL is CISPO"
        assert cfg.trainer.loss.eps_max == 4.0, f"{relpath}: CISPO eps_max"
        assert cfg.trainer.loss.loss_scale_mode == "sequence", f"{relpath}: sequence-mode reduction"

        # Prompt-level averaging — paper §3.3.
        assert cfg.orchestrator.prompt_average_loss is True, f"{relpath}: prompt-avg ON"

        # Batch-level advantage normalization — paper §3.4.
        assert cfg.orchestrator.advantage.normalization == "batch", (
            f"{relpath}: batch-level adv-norm (paper §3.4); default is 'none' (Dr. GRPO)"
        )

        # FP32 LM-head — paper §3.2.
        assert cfg.trainer.model.fp32_lm_head is True, f"{relpath}: FP32 LM-head trainer"
        assert cfg.inference.model.fp32_lm_head is True, f"{relpath}: FP32 LM-head inference"
        assert cfg.trainer.matmul_precision == "highest", (
            f"{relpath}: matmul_precision='highest' is required when fp32_lm_head=True"
        )

        # NPR — paper §3.6.
        assert cfg.orchestrator.buffer.no_positive_resampling is True, f"{relpath}: NPR ON"
        assert cfg.orchestrator.buffer.no_positive_resampling_threshold == 0.9, (
            f"{relpath}: NPR threshold from §3.6"
        )

        # Async pipeline-RL — paper §3.1.
        assert cfg.trainer.max_async_level == 8, f"{relpath}: paper max_async_level=8"
        assert cfg.orchestrator.max_async_level == 8, f"{relpath}: paper max_async_level=8 (orch)"
        assert cfg.orchestrator.max_off_policy_steps == 1_000_000_000, (
            f"{relpath}: never-cancel (§3.1); default 8 silently re-enables cancellation"
        )
        assert cfg.weight_broadcast.type == "nccl", f"{relpath}: NCCL weight broadcast"
        # The NCCL+max_async_level>1 combo requires the experimental override.
        assert cfg.trainer.experimental.allow_nccl_async_level_override is True, (
            f"{relpath}: NCCL async-level override is required for max_async_level>1"
        )

        # `extra_env_kwargs["max_seq_len"]` must match `orchestrator.seq_len`
        # on every train AND eval env. `OrchestratorConfig.resolve_env_config`
        # historically froze max_seq_len at the orchestrator's schema default
        # (2048) when top-level `seq_len` propagated AFTER resolve_env_config
        # had run, silently truncating rollouts at 2K while the trainer-side
        # packer used 8K. Re-propagation lives in `auto_setup_seq_len` now;
        # pin it.
        for env in cfg.orchestrator.train.env or []:
            assert env.extra_env_kwargs.get("max_seq_len") == cfg.orchestrator.seq_len, (
                f"{relpath}: train env {env.id!r} max_seq_len="
                f"{env.extra_env_kwargs.get('max_seq_len')} disagrees with "
                f"orchestrator.seq_len={cfg.orchestrator.seq_len} — "
                "rollouts will be silently truncated at the smaller value."
            )
        if cfg.orchestrator.eval is not None:
            for env in cfg.orchestrator.eval.env or []:
                assert env.extra_env_kwargs.get("max_seq_len") == cfg.orchestrator.seq_len, (
                    f"{relpath}: eval env {env.id!r} max_seq_len="
                    f"{env.extra_env_kwargs.get('max_seq_len')} disagrees with "
                    f"orchestrator.seq_len={cfg.orchestrator.seq_len}"
                )


def test_adamw_config_carries_eps_and_rejects_typos():
    """AdamW `eps` field must round-trip into the optimizer constructor.

    Regression: pre-fix `AdamWConfig` had no `eps` field; the smoke configs'
    `eps = 1e-15` was silently dropped (BaseModel default `extra="ignore"`),
    and torch's default 1e-8 was used. With ScaleRL gradient magnitudes
    that's an underflow risk the paper specifically warns about.
    """
    from prime_rl.configs.trainer import AdamWConfig

    cfg = AdamWConfig(eps=1e-15)
    assert cfg.eps == 1e-15
    # extra=forbid on BaseOptimizerConfig — typos must raise.
    with pytest.raises(ValidationError):
        AdamWConfig(epislon=1e-15)  # typo
    # `eps = 0.0` divides by zero in Adam's denominator — reject.
    with pytest.raises(ValidationError):
        AdamWConfig(eps=0.0)


def test_sgd_config_rejects_eps_typo():
    """`extra=forbid` should fire on every BaseOptimizerConfig subclass, not just
    AdamW — guards against a future config class forgetting to inherit the strict
    behavior."""
    from prime_rl.configs.trainer import SGDConfig

    with pytest.raises(ValidationError):
        SGDConfig(eps=1e-15)  # SGD has no eps field
    with pytest.raises(ValidationError):
        SGDConfig(nestrov=True)  # typo of `nesterov`


# `extra=forbid` on every ScaleRL-relevant Pydantic schema. A typo in any of
# these silently kept the default before; for the canonical paper knobs
# (`eps_max`, `adv_tau`, `loss_scale_mode`, scheduler `warmup_steps`) that
# was a paper-fidelity regression invisible to the test suite.


@pytest.mark.parametrize(
    "config_cls_path,base_payload,typo_field",
    [
        ("prime_rl.configs.trainer:CISPOLossConfig", dict(type="cispo"), "epsmax"),
        ("prime_rl.configs.trainer:CISPOLossConfig", dict(type="cispo"), "adv_taul"),
        ("prime_rl.configs.trainer:DefaultLossConfig", dict(type="default"), "klau"),
        ("prime_rl.configs.trainer:SFTLossConfig", dict(type="sft"), "looscale"),
        ("prime_rl.configs.trainer:CustomLossConfig", dict(type="custom", import_path="x.y"), "imp_path"),
        ("prime_rl.configs.trainer:ConstantSchedulerConfig", dict(type="constant"), "warmup"),
        ("prime_rl.configs.trainer:LinearSchedulerConfig", dict(type="linear"), "warmpu_steps"),
        ("prime_rl.configs.trainer:CosineSchedulerConfig", dict(type="cosine"), "warm_steps"),
        ("prime_rl.configs.trainer:FileSystemWeightBroadcastConfig", dict(type="filesystem"), "savesharded"),
        ("prime_rl.configs.trainer:NCCLWeightBroadcastConfig", dict(type="nccl"), "hosts"),
        ("prime_rl.configs.orchestrator:CustomAdvantageConfig", dict(type="custom", import_path="x.y"), "imp_path"),
        ("prime_rl.configs.orchestrator:FileSystemWeightBroadcastConfig", dict(type="filesystem"), "savesharded"),
        ("prime_rl.configs.orchestrator:NCCLWeightBroadcastConfig", dict(type="nccl"), "hosts"),
    ],
    ids=lambda x: str(x) if not isinstance(x, dict) else "",
)
def test_scalerl_schemas_reject_unknown_fields(config_cls_path, base_payload, typo_field):
    """Each ScaleRL-relevant schema must reject unknown fields via `extra="forbid"`.

    A silent default for `epsmax`/`adv_tau`/`loss_scale_mode`/`warmup_steps` is
    a paper-fidelity regression invisible to functional tests — pin the
    typo-rejection contract here.
    """
    import importlib

    module_name, class_name = config_cls_path.split(":")
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)

    cls.model_validate(base_payload)
    with pytest.raises(ValidationError):
        cls.model_validate({**base_payload, typo_field: 99.0})


def test_adamw_eps_round_trips_into_torch_optim():
    """The whole point of fixing the eps drop is that the value reaches
    `torch.optim.AdamW(eps=...)`. A regression that re-removes `eps=config.eps`
    from the AdamW constructor in `_create_optimizer` would silently put us
    back on torch's default 1e-8 — the original bug. Pin against that."""
    import torch
    from torch import nn

    from prime_rl.configs.trainer import AdamWConfig
    from prime_rl.trainer.optim import _create_optimizer

    config = AdamWConfig(eps=1e-15, lr=5e-7, weight_decay=0.01)
    params = [("dummy", nn.Parameter(torch.zeros(2)))]
    opt = _create_optimizer(config, params, parallel_dims=None)
    assert opt.param_groups[0]["eps"] == 1e-15


def test_linear_scheduler_config_rejects_zero_phases():
    """Both `warmup_steps=0` and `decay_steps=0` would crash inside
    `setup_linear_scheduler` after the SLURM container booted. Reject at config
    load."""
    from prime_rl.configs.trainer import LinearSchedulerConfig

    with pytest.raises(ValidationError, match="warmup_steps > 0 or"):
        LinearSchedulerConfig(warmup_steps=0, decay_steps=0)
    # Either alone is fine.
    LinearSchedulerConfig(warmup_steps=10, decay_steps=0)
    LinearSchedulerConfig(warmup_steps=0, decay_steps=10)


def test_rl_config_rejects_sequence_mode_without_prompt_average_loss():
    """`loss_scale_mode in {sequence, none}` without `prompt_average_loss=True`
    leaves per-sequence weights at the neutral 1.0 default. compute_loss
    multiplies by `fsdp_gradient_divide_factor`, producing a raw global sum scaled by DP
    — effective LR scales linearly with batch size and DP size. CISPO defaults
    to sequence so an out-of-the-box CISPO config without prompt-avg silently
    lacks normalization."""
    with pytest.raises(ValidationError, match="relies on the packer"):
        RLConfig.model_validate(
            {
                **_RL_BASE,
                "trainer": {"loss": {"type": "cispo", "loss_scale_mode": "sequence"}},
                # prompt_average_loss not set — defaults to False
            }
        )
    # Even CustomLossConfig needs explicit prompt_avg or token mode (the
    # CustomLossConfig exemption was dropped — too footgun-y).
    with pytest.raises(ValidationError, match="relies on the packer"):
        RLConfig.model_validate(
            {
                **_RL_BASE,
                "trainer": {
                    "loss": {
                        "type": "custom",
                        "import_path": "prime_rl.trainer.rl.loss.sft_loss_fn",
                        "loss_scale_mode": "sequence",
                    }
                },
            }
        )


def test_rl_config_rejects_conflicting_explicit_nccl_async_override():
    """Setting `allow_nccl_async_level_override` explicitly on BOTH sides with
    different values is an explicit user disagreement — propagation can't
    silently pick one. Reject."""
    with pytest.raises(ValidationError, match="conflict"):
        RLConfig.model_validate(
            {
                **_RL_BASE,
                "max_async_level": 8,
                "weight_broadcast": {"type": "nccl"},
                "orchestrator": {"experimental": {"allow_nccl_async_level_override": True}},
                "trainer": {"experimental": {"allow_nccl_async_level_override": False}},
            }
        )


def test_rl_config_propagates_nccl_async_override_from_trainer_to_orchestrator():
    """Symmetric to the existing orchestrator→trainer test — pin both directions."""
    config = RLConfig.model_validate(
        {
            **_RL_BASE,
            "max_async_level": 8,
            "weight_broadcast": {"type": "nccl"},
            "trainer": {"experimental": {"allow_nccl_async_level_override": True}},
        }
    )
    assert config.orchestrator.experimental.allow_nccl_async_level_override is True
    assert config.trainer.experimental.allow_nccl_async_level_override is True


def test_rl_config_root_rejects_nccl_async_without_override():
    """Setting NCCL + max_async_level > 1 at the RL root with no override on either
    sub-config must still be rejected. The auto_setup_weight_broadcast validator
    reassigns trainer/orchestrator weight_broadcast after their validators have run,
    so without an explicit root-level guardrail this combo would slip through.
    """
    with pytest.raises(ValidationError, match="NCCL weight broadcast with max_async_level"):
        RLConfig.model_validate(
            {
                **_RL_BASE,
                "max_async_level": 8,
                "weight_broadcast": {"type": "nccl"},
            }
        )
