# ScaleRL recipe — what this fork adds

This is a public fork of [`PrimeIntellect-ai/prime-rl`](https://github.com/PrimeIntellect-ai/prime-rl)
that adds the seven training-recipe ingredients from
[**The Art of Scaling Reinforcement Learning Compute for LLMs**](https://arxiv.org/abs/2510.13786)
(Khatri et al., Meta / UT Austin / UCL / Berkeley / Harvard / Periodic Labs, Oct 2025).

The diff against upstream is intentionally minimal — every ScaleRL feature is opt-in via config; the
upstream defaults are preserved.

## Features

| # | Feature | Where it lives | Default |
|---|---------|----------------|---------|
| 1 | Async **Pipeline-RL** (`max_async_level=k`, NCCL broadcast at `k>1`) | `configs/orchestrator.py` (`OrchestratorExperimentalConfig.allow_nccl_async_level_override`) — upstream already has the `max_async_level` knob; we add an opt-in override of the validator that hard-rejected NCCL + async-level > 1. | off |
| 2 | Canonical **Minimax-CISPO** loss (stop-gradient, upper-truncated IS) | `configs/trainer.py` (`CISPOLossConfig`), `trainer/rl/loss.py` (`cispo_loss_fn`). Loss form: `-sg(min(ρ, ε_max)) · Â · log π_θ`. | off (default loss unchanged) |
| 3 | **Prompt-level loss averaging** | `configs/orchestrator.py` (`OrchestratorConfig.prompt_average_loss`), `trainer/batch.py` (`apply_prompt_average_sequence_weights`), `trainer/rl/loss.py` (`compute_loss(sequence_loss_weights=...)`). | off |
| 4 | **Batch-level advantage normalization** | `configs/orchestrator.py` (`DefaultAdvantageConfig.normalization = "none" \| "group" \| "batch"`), `orchestrator/advantage.py`. | `"none"` (upstream behavior) |
| 5 | **FP32 LM-head** matmul on both trainer and inference | `configs/trainer.py` + `configs/inference.py` (`ModelConfig.fp32_lm_head`), `trainer/models/layers/lm_head.py`, `inference/patches.py` (vLLM monkey-patch via `vllm.general_plugins` entry point). | off |
| 6 | **Zero-variance filtering** | Upstream's existing `ZeroAdvantageFilter` (#2192) ships in the default filter list. Nothing to enable. | on |
| 7 | **No-Positive-Resampling** (Polaris-style permanent prompt removal at `pass_rate ≥ τ`) | `configs/orchestrator.py` (`BufferConfig.no_positive_resampling`, `no_positive_resampling_threshold`), `orchestrator/buffer.py`. | off |

**Not in this fork:** forced-length interruptions (ScaleRL §2.3 / `sec:length_control`). Lowest-priority
ingredient per the paper's leave-one-out ablation; monitor truncation rate first and add only if
rollouts exceed budget more than ~5% of the time.

## Configs

Two reference configs ship with the recipe wired up:

- **`configs/scalerl_math/rl.toml`** — internal smoke. Qwen3-8B on the `math-env` (verifiers'
  built-in `PrimeIntellect/Hendrycks-Math` env). Submit with
  `sbatch scripts/scalerl_smoke.sbatch`.
- **`configs/scalerl_terminal_bench/rl.toml`** — Snowflake POC handoff config. Qwen3-8B on
  the in-tree Terminal-Bench Harbor tasks (`environments/terminal_bench/`). 30 tasks
  shipped, split 20 train / 10 test. Requires a Docker daemon reachable from the rollout
  process (host or `/var/run/docker.sock` bind-mounted into the container).

Both configs exercise the full ScaleRL knob set (CISPO, prompt-level averaging, batch-level
advantage normalization, FP32 LM-head on both sides, NPR @ 0.9, `max_async_level=8` with the
NCCL experimental override).

## Recipe references

Each ingredient cites its origin in the config docstrings. Primary sources:

- ScaleRL: Khatri et al. 2026, *The Art of Scaling Reinforcement Learning Compute for LLMs*, arXiv:2510.13786.
- CISPO origin: Minimax-M1 §3.1, *MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention*, arXiv:2506.13585.
- Prompt-level loss averaging: DAPO §3.3, arXiv:2503.14476.
- Batch-level advantage normalization: Hu et al., *Reinforce++*, also Magistral.
- No-Positive-Resampling: Polaris (An et al., 2025), 0.9 threshold.
- FP32 LM-head: Minimax-M1 §3.2.

For a deeper write-up of which features are server-side vs client-side and the API contract this
fork assumes, see the upstream POC plan doc (kept in the parent research repo, not shipped here).
