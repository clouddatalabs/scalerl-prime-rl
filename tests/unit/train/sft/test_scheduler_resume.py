"""Pin _resolve_sft_scheduler_steps's handling of resume_step=-1 sentinel.

The R32 fix swapped `config.ckpt.resume_step` (which retains the literal -1
sentinel even after `resolve_latest_ckpt_step` runs) for the resolved
`checkpoint_step` value. Without that fix, a user resuming with
`--ckpt.resume_step=-1 --ckpt.skip_scheduler=true` would silently get
`max_steps - (-1) = max_steps + 1` steps of slower LR decay across the
entire post-resume run.
"""

from prime_rl.configs.sft import SFTConfig
from prime_rl.trainer.sft.train import _resolve_sft_scheduler_steps


def _config(*, max_steps, ckpt_kwargs=None):
    payload = {"max_steps": max_steps}
    if ckpt_kwargs is not None:
        payload["ckpt"] = {"interval": 100, "resume_step": None, "skip_scheduler": False, **ckpt_kwargs}
    return SFTConfig.model_validate(payload)


def test_no_resume_returns_max_steps_unchanged():
    cfg = _config(max_steps=1000)
    assert _resolve_sft_scheduler_steps(cfg, checkpoint_step=None) == 1000


def test_resume_without_skip_scheduler_returns_max_steps():
    cfg = _config(max_steps=1000, ckpt_kwargs={"resume_step": 200, "skip_scheduler": False})
    assert _resolve_sft_scheduler_steps(cfg, checkpoint_step=200) == 1000


def test_resume_with_skip_scheduler_subtracts_resolved_checkpoint_step():
    """Explicit numeric resume_step still uses the resolved checkpoint_step
    (the function takes the resolved value as argument and ignores resume_step)."""
    cfg = _config(max_steps=1000, ckpt_kwargs={"resume_step": 200, "skip_scheduler": True})
    assert _resolve_sft_scheduler_steps(cfg, checkpoint_step=200) == 800


def test_resume_step_minus_one_sentinel_uses_resolved_checkpoint_not_negative():
    """Critical regression: resume_step=-1 + skip_scheduler=True must NOT
    add 1 to scheduler_steps. The resolved checkpoint_step (e.g. 350) is what
    counts, not the literal -1 sentinel.
    """
    cfg = _config(max_steps=1000, ckpt_kwargs={"resume_step": -1, "skip_scheduler": True})
    assert _resolve_sft_scheduler_steps(cfg, checkpoint_step=350) == 650
    # The pre-fix arithmetic `max_steps - resume_step` would have given 1001.
    assert _resolve_sft_scheduler_steps(cfg, checkpoint_step=350) != 1001


def test_no_max_steps_returns_none():
    """`max_steps=None` (run indefinitely) is preserved through both branches."""
    cfg = _config(max_steps=None)
    assert _resolve_sft_scheduler_steps(cfg, checkpoint_step=None) is None
    cfg2 = _config(max_steps=None, ckpt_kwargs={"resume_step": 100, "skip_scheduler": True})
    assert _resolve_sft_scheduler_steps(cfg2, checkpoint_step=100) is None
