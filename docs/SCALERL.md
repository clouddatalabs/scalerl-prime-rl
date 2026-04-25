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
| 5 | **FP32 LM-head** matmul on both trainer and inference | `configs/trainer.py` + `configs/inference.py` (`ModelConfig.fp32_lm_head`), `trainer/models/layers/lm_head.py`, `inference/patches.py` (vLLM monkey-patch via `vllm.general_plugins` entry point). The inference patch reads the env var `PRIME_RL_VLLM_FP32_LM_HEAD`, which `prime_rl.inference.server` sets from `inference.model.fp32_lm_head`. **If you launch vLLM directly (e.g. `vllm serve …`), set `PRIME_RL_VLLM_FP32_LM_HEAD=1` yourself; the config-side flag has no effect outside the prime-rl entrypoint.** | off |
| 6 | **Zero-variance filtering** | Upstream's existing `ZeroAdvantageFilter` (#2192) ships in the default filter list. Nothing to enable. | on |
| 7 | **No-Positive-Resampling** (Polaris-style permanent prompt removal at `pass_rate ≥ τ`) | `configs/orchestrator.py` (`BufferConfig.no_positive_resampling`, `no_positive_resampling_threshold`), `orchestrator/buffer.py`. | off |

**Not in this fork:** forced-length interruptions (ScaleRL §2.3 / `sec:length_control`). Lowest-priority
ingredient per the paper's leave-one-out ablation; monitor truncation rate first and add only if
rollouts exceed budget more than ~5% of the time.

## Quickstart

```bash
git clone <this-repo> ~/scalerl-prime-rl
cd ~/scalerl-prime-rl

# 1. Install. Pinned to vllm>=0.19, torch+cu128, transformers @ a HEAD commit,
#    flash-attn-cute (FA4) at rev abd9943b. Takes ~10 minutes on a fresh cache.
uv sync --all-extras

# 2. Repair flash-attn-cute namespace if `uv sync` happened to install
#    `flash-attn` (FA2) after `flash-attn-cute` — both ship a `flash_attn/cute/`
#    sub-package and the FA2 stub silently shadows the real FA4. Idempotent;
#    this also runs from the sbatch script as a defensive double-check.
bash scripts/fix-flash-attn-cute.sh

# 3. Auth (only the bits the config needs):
#    - HF token if you use a gated model (Qwen3-8B is open, so this is optional).
#    - WANDB_API_KEY if you uncomment [wandb] in either config.
#      Either `wandb login` once, or put `WANDB_API_KEY=...` in `.env`.
#    The smoke configs ship with `[wandb]` commented out.

# 4. Pre-download the model so the first sbatch doesn't block on a multi-GB pull:
huggingface-cli download Qwen/Qwen3-8B

# 5. Submit. Override the partition / log dir if your cluster differs:
sbatch -p <partition> --export=ALL,REPO_ROOT="$PWD",OUTPUT_ROOT=/your/path \
       scripts/scalerl_smoke.sbatch

# 6. Tail the log (smoke configs train for 500 steps; first step is ~3 min cold,
#    steady-state ~30-60s/step on 4xB200 with FA4 + compiled vLLM):
tail -f /your/path/<jobid>.log
```

**Cluster preconditions:**
- 8 GPUs visible to one node (config splits 4 train / 4 inference).
- SLURM with a partition you pass via `sbatch -p <name>` (the sbatch defaults
  to `gpu`).
- Network reachable from the compute node, OR run `huggingface-cli download`
  and `bash scripts/fix-flash-attn-cute.sh` on the login node before submit.
- For Terminal-Bench: a Docker daemon reachable from the rollout process
  (host or `/var/run/docker.sock` bind-mounted into the container).

**Resume.** Pass `--ckpt.resume_step=-1` to load the latest checkpoint from
`outputs/run_default/checkpoints`:
```bash
.venv/bin/rl @ configs/scalerl_math/rl.toml --ckpt.resume_step=-1
```
The shipped configs cap retention at `keep_last=3` plus every 500-step
milestone (`keep_interval=500`); raise `keep_last` if your cluster has the
storage and you want fast random-step resume.

**Multi-node.** The shipped configs target a single 8-GPU node. To scale
horizontally, multiply `num_train_gpus` in `[deployment]` and add
`#SBATCH --nodes=N`. Multi-node NCCL weight broadcast still requires the
experimental override (already on); on hardware without EFA, switch
`[weight_broadcast] type = "filesystem"` and drop the override.

## Configs

Two reference configs ship with the recipe wired up:

- **`configs/scalerl_math/rl.toml`** — internal smoke. Qwen3-8B on the `math-env` (verifiers'
  built-in `PrimeIntellect/Hendrycks-Math` env). Submit with
  `sbatch scripts/scalerl_smoke.sbatch`.
- **`configs/scalerl_terminal_bench/rl.toml`** — Snowflake POC handoff config. Qwen3-8B on
  the in-tree Terminal-Bench Harbor tasks (`environments/terminal_bench/`). 28 usable
  committed tasks split 18 train / 10 test. (Two task names — `fix-code-vulnerability`
  and `sqlite-with-gcov` — are committed as symlinks pointing outside the repo and are
  excluded from the runtime config.) Requires a Docker daemon reachable from the rollout
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
