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


def get_config_files() -> list[Path]:
    """Any TOML file inside `configs/` or `examples/`"""
    config_files = list(Path("configs").rglob("*.toml"))
    example_files = list(Path("examples").rglob("*.toml"))

    return config_files + example_files


@pytest.mark.parametrize("config_file", get_config_files(), ids=lambda x: x.as_posix())
def test_load_configs(config_file: Path):
    """Tests that all config files can be loaded by at least one config class."""
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


_RL_BASE = {
    "trainer": {},
    "orchestrator": {},
    "inference": {},
}


def test_rl_config_propagates_nccl_async_override_from_orchestrator_to_trainer():
    """Setting the override on the orchestrator side at the RL level propagates to the trainer.

    Without this propagation, the RLConfig-level max_async_level=8 + NCCL combo would still
    be rejected by TrainerConfig's own validator. See snowflake_poc_critique.md §1.
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
    (ScaleRL §3.2). See snowflake_poc_critique.md §3.
    """
    with pytest.raises(ValidationError, match="fp32_lm_head must match"):
        RLConfig.model_validate(
            {
                **_RL_BASE,
                "trainer": {"model": {"fp32_lm_head": True}},
                "inference": {"model": {"fp32_lm_head": False}},
            }
        )


def test_rl_config_accepts_matched_fp32_lm_head():
    """fp32_lm_head=True on both sides validates."""
    config = RLConfig.model_validate(
        {
            **_RL_BASE,
            "trainer": {"model": {"fp32_lm_head": True}},
            "inference": {"model": {"fp32_lm_head": True}},
        }
    )
    assert config.trainer.model.fp32_lm_head is True
    assert config.inference.model.fp32_lm_head is True


def test_rl_config_skips_fp32_lm_head_check_when_inference_omitted():
    """RLConfig.inference may legitimately be None (externally managed inference pools or
    num_infer_nodes=0 fake-data runs). The fp32 consistency check must skip rather than
    AttributeError. See snowflake_poc_critique.md §2.
    """
    config = RLConfig.model_validate(
        {
            "trainer": {"model": {"fp32_lm_head": True}},
            "orchestrator": {},
            "inference": None,
        }
    )
    assert config.inference is None
    assert config.trainer.model.fp32_lm_head is True


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


def test_rl_config_root_rejects_nccl_async_without_override():
    """Setting NCCL + max_async_level > 1 at the RL root with no override on either
    sub-config must still be rejected. The auto_setup_weight_broadcast validator
    reassigns trainer/orchestrator weight_broadcast after their validators have run,
    so without an explicit root-level guardrail this combo would slip through.
    See snowflake_poc_critique.md §5.
    """
    with pytest.raises(ValidationError, match="NCCL weight broadcast with max_async_level"):
        RLConfig.model_validate(
            {
                **_RL_BASE,
                "max_async_level": 8,
                "weight_broadcast": {"type": "nccl"},
            }
        )
