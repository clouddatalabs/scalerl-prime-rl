# ScaleRL recipe — what this fork adds

This is a public fork of [`PrimeIntellect-ai/prime-rl`](https://github.com/PrimeIntellect-ai/prime-rl)
that adds the seven training-recipe ingredients from
[**The Art of Scaling Reinforcement Learning Compute for LLMs**](https://arxiv.org/abs/2510.13786)
(Khatri et al., Meta / UT Austin / UCL / Berkeley / Harvard / Periodic Labs, Oct 2025).

The diff against upstream is intentionally minimal — every ScaleRL feature is opt-in via config; the
upstream defaults are preserved.

## Features

The "Schema default" column is what you get with a bare upstream config that
sets none of the fork's knobs (preserves upstream behavior). The "POC config"
column is what the shipped `configs/scalerl_math/rl.toml` and
`configs/scalerl_terminal_bench/rl.toml` actually set — this is the recipe
end-state the handoff exercises.

| # | Feature | Where it lives | Schema default | POC config |
|---|---------|----------------|----------------|------------|
| 1 | Async **Pipeline-RL** (`max_async_level=k`, NCCL broadcast at `k>1`) | `configs/orchestrator.py` (`OrchestratorExperimentalConfig.allow_nccl_async_level_override`) — upstream already has the `max_async_level` knob; we add an opt-in override of the validator that hard-rejected NCCL + async-level > 1. | off | on (`max_async_level=8`, NCCL) |
| 2 | Canonical **Minimax-CISPO** loss (stop-gradient, upper-truncated IS) | `configs/trainer.py` (`CISPOLossConfig`), `trainer/rl/loss.py` (`cispo_loss_fn`). Loss form: `-sg(min(ρ, ε_max)) · Â · log π_θ`. | off (default loss unchanged) | on (`type="cispo"`, `eps_max=4.0`) |
| 3 | **Prompt-level loss averaging** | `configs/orchestrator.py` (`OrchestratorConfig.prompt_average_loss`), `trainer/batch.py` (`apply_prompt_average_sequence_weights`), `trainer/rl/loss.py` (`compute_loss(sequence_loss_weights=...)`). | off | on |
| 4 | **Batch-level advantage normalization** | `configs/orchestrator.py` (`DefaultAdvantageConfig.normalization = "none" \| "group" \| "batch"`), `orchestrator/advantage.py`. | `"none"` (upstream behavior) | `"batch"` |
| 5 | **FP32 LM-head** matmul on both trainer and inference | `configs/trainer.py` + `configs/inference.py` (`ModelConfig.fp32_lm_head`), `trainer/models/layers/lm_head.py`, `inference/patches.py` (vLLM monkey-patch via `vllm.general_plugins` entry point). The inference patch reads the env var `PRIME_RL_VLLM_FP32_LM_HEAD`, which `prime_rl.inference.server` sets from `inference.model.fp32_lm_head`. **If you launch vLLM directly (e.g. `vllm serve …`), set `PRIME_RL_VLLM_FP32_LM_HEAD=1` yourself; the config-side flag has no effect outside the prime-rl entrypoint.** | off | on (both sides) |
| 6 | **Zero-variance filtering** | Upstream's existing `ZeroAdvantageFilter` (#2192) ships in the default filter list. Nothing to enable. | on | on |
| 7 | **No-Positive-Resampling** (Polaris-style permanent prompt removal at `pass_rate ≥ τ`) | `configs/orchestrator.py` (`BufferConfig.no_positive_resampling`, `no_positive_resampling_threshold`, `no_positive_resampling_min_groups`), `orchestrator/buffer.py`. | off | on (`τ=0.9`, `min_groups=4`) |

**Not in this fork:** forced-length interruptions (ScaleRL §2.3 / `sec:length_control`). Lowest-priority
ingredient per the paper's leave-one-out ablation; monitor truncation rate first and add only if
rollouts exceed budget more than ~5% of the time.

## Quickstart

```bash
git clone https://github.com/clouddatalabs/scalerl-prime-rl.git ~/scalerl-prime-rl
cd ~/scalerl-prime-rl
REPO_ROOT="$PWD"

# Pin a HuggingFace cache location for the rest of this shell so step 4's
# pre-download lands where the sbatch will look. Use the absolute repo path
# (NOT $PWD) so a `cd` between steps 4 and 5 doesn't orphan the cache —
# the sbatch defaults `HF_HOME` to `$REPO_ROOT/hf-cache`.
export HF_HOME="$REPO_ROOT/hf-cache"

# 1. Install. Pinned to vllm>=0.19, torch+cu128, transformers @ a HEAD commit,
#    flash-attn-cute (FA4) at rev abd9943b. Takes ~10 minutes on a fresh cache.
#    NOTE: use --extra all (the aggregate extra), NOT --all-extras. The latter
#    enumerates every extra by name and pulls in [flash-attn-3], whose wheel
#    ships Hopper sm_90 kernels only and crashes on B200.
uv sync --extra all

# 2. Repair flash-attn-cute namespace if `uv sync` happened to install
#    `flash-attn` (FA2) after `flash-attn-cute` — both ship a `flash_attn/cute/`
#    sub-package and the FA2 stub silently shadows the real FA4. Idempotent;
#    this also runs from the sbatch script as a defensive double-check, but
#    running it on the login node first lets us catch network failures early
#    (compute nodes often can't reach github.com).
bash scripts/fix-flash-attn-cute.sh

# 3. Auth (only the bits the config needs):
#    - HF token if you use a gated model (Qwen3-8B is open, so this is optional).
#    - WANDB_API_KEY if you uncomment [wandb] in either config.
#      Either `wandb login` once, or put `WANDB_API_KEY=...` in `.env`.
#      Set `WANDB_MODE=offline` if your cluster has no outbound network.
#    The smoke configs ship with `[wandb]` commented out.

# 4. Pre-download the model so the first sbatch doesn't block on a multi-GB
#    pull through whatever NAT your compute nodes have. Uses HF_HOME set above.
#    The CLI is `hf` in modern huggingface_hub; older `huggingface-cli` was
#    removed. Invoke the venv binary directly so we don't clobber any
#    pre-existing conda/pyenv activation in your shell.
.venv/bin/hf download Qwen/Qwen3-8B

# 5. Submit. Override the partition / log dir / config-to-launch if your
#    cluster differs. OUTPUT_ROOT controls where the slurm stdout/stderr go;
#    the run's checkpoints + rollouts go under `[output_dir]` from the TOML
#    (default: outputs/scalerl_math or outputs/scalerl_terminal_bench).
#    SCALERL_CONFIG selects which config to launch — defaults to the math
#    smoke; switch to the Snowflake-handoff TB config explicitly.
#
#    NOTE: `export VAR=...; sbatch --export=ALL` is the safe pattern.
#    `sbatch --export=ALL,KEY=VAL,KEY2=VAL2` with embedded commas inside
#    a backslash-newline continuation tends to be miscopied or parsed
#    differently by various shells.
export REPO_ROOT="$PWD"
export OUTPUT_ROOT="$PWD/slurm-logs"
export SCALERL_CONFIG=configs/scalerl_terminal_bench/rl.toml
sbatch -p <partition> --export=ALL scripts/scalerl_smoke.sbatch
# (For the math smoke, unset SCALERL_CONFIG — sbatch defaults to scalerl_math.)

# 6. Tail the log. Smoke configs train for 500 steps; first step is ~3 min
#    cold, steady-state ~30-60s/step on 4xB200 with FA4 + compiled vLLM
#    for the math config. NOTE: the Terminal-Bench config spins one Docker
#    container per rollout (Harbor format), so steady-state per-step time is
#    dominated by container startup and test execution, not GPU — expect
#    several minutes/step until the per-task base images are warm in the
#    Docker layer cache. The training process is not hung.
tail -f "$PWD/slurm-logs/<jobid>.log"
```

**Cluster preconditions:**
- 8 GPUs visible to one node (config splits 4 train / 4 inference).
- SLURM with a partition you pass via `sbatch -p <name>` (the sbatch defaults
  to `gpu`). Run `sinfo` on the login node first to find the right name.
  The shipped sbatch requests `--gres=gpu:8 --exclusive`; clusters that type
  their gres (e.g. `gpu:b200:8`, `gpu:h100:8`) need an override at submit
  time: `sbatch --gres=gpu:b200:8 -p <partition> ... scripts/scalerl_smoke.sbatch`.
  Verify with `sinfo -o '%G'` (or your scheduler's gres listing) on the
  login node first.
- **NVIDIA Blackwell (B200) GPUs.** The configs are pinned to flash-attn-cute
  (FA4) which builds Blackwell sm_100 kernels; FA3 is deliberately excluded
  by the `[all]` extra (FA3 wheels ship Hopper sm_90 only and crash on B200
  with "no kernel image available"). For H100/Hopper or older, you must
  swap `attn = "fa4"` to `attn = "sdpa"` (or wire FA2/FA3 yourself) in the
  `[trainer.model]` block of the config you submit; the recipe is otherwise
  hardware-agnostic.
- **Build toolchain on the login node** (where `uv sync --extra all` runs):
  `build-essential` (gcc/g++), `git`, `curl`, **plus the CUDA toolkit**
  (`nvcc` matching torch's CUDA 12.8 — the `cuda-toolkit-12-8` package or
  equivalent). CUTLASS + flash-attn-cute build from source on first sync.
  On a minimal/distroless image, install via
  `INSTALL_BASE_PACKAGES=1 bash scripts/install.sh` or your cluster's
  package manager — `uv sync` will otherwise fail with an opaque CUTLASS
  or "nvcc not found" error halfway through.
- **Network from the COMPUTE node, OR pre-stage on the login node.** Many
  managed clusters firewall compute nodes off the package mirrors. The
  `uv sync` step needs egress to:
  - `github.com` (CUTLASS, FA4 sources)
  - `huggingface.co` (Qwen3-8B and dataset weights)
  - `hub.primeintellect.ai` (the `[envs]` extra resolves verifiers
    environments through Prime Intellect's index)
  - For Terminal-Bench only: `ghcr.io/laude-institute/...` (TB task base
    images pulled on first rollout per task).

  If your compute nodes can't reach any of these, pre-stage on the login
  node: `hf download Qwen/Qwen3-8B`, `bash scripts/fix-flash-attn-cute.sh`,
  and `docker pull` the TB base images you'll need.
- **Non-AWS clusters with `LD_PRELOAD` shimming `libcublas`.** Set
  `SCALERL_FORCE_CLEAR_LD=1` in `.env` (or before `sbatch`). The smoke
  script auto-clears `LD_LIBRARY_PATH` and `LD_PRELOAD` when it sees the
  AWS DLAMI marker (`/etc/dlami/dlami_version`). Other clusters with a
  pre-loaded system cuBLAS hit the same `CUBLAS_STATUS_INVALID_VALUE`
  failure in vLLM's compiled QKV path on B200 — `SCALERL_FORCE_CLEAR_LD=1`
  is the manual override.
- **For Terminal-Bench:** the rollout process must reach a Docker daemon.
  `scripts/scalerl_smoke.sbatch` runs `.venv/bin/rl` on the SLURM host
  (NOT inside a container), so the `prime-rl` user needs **either** read
  access to `/var/run/docker.sock` (typically via `docker` group
  membership) **or** passwordless `sudo docker` configured. If neither
  applies, the rollouts crash early with a clear "Docker daemon
  unreachable" error from `environments/terminal_bench/env.py`.

**Resume.** Resubmit the same sbatch with `SCALERL_RESUME=1`:
```bash
export REPO_ROOT="$PWD"
export OUTPUT_ROOT="$PWD/slurm-logs"
export SCALERL_CONFIG=configs/scalerl_math/rl.toml
export SCALERL_RESUME=1
sbatch -p <partition> --export=ALL scripts/scalerl_smoke.sbatch
```
Caveat: a run that crashes BEFORE the first checkpoint at `[ckpt] interval`
(default 100 steps) leaves no checkpoint to resume from — the resume run
will error out at startup. Lower `[ckpt] interval` if early crashes are
likely, or accept restarting from step 0.

**Restart from step 0 after an early crash.** `RLConfig` rejects re-using a
populated `output_dir` only when it contains checkpoints (see
`validate_output_dir` in `src/prime_rl/utils/pathing.py`); a populated
`output_dir` with rollouts/logs but no checkpoint is silently overwritten —
`clean_future_steps(resume_step=-1)` wipes it during the next launch. If
you crashed pre-first-checkpoint, the cleanest unblocking sequence is to
remove the directory yourself so the next run starts from a known empty
state:
```bash
rm -rf outputs/scalerl_math   # or outputs/scalerl_terminal_bench
# unset SCALERL_RESUME (or skip exporting it)
sbatch -p <partition> --export=ALL scripts/scalerl_smoke.sbatch
```
The shipped configs set `output_dir = "outputs/scalerl_math"` (and
`outputs/scalerl_terminal_bench`) so resume is deterministic and concurrent
submissions don't collide. Checkpoints land under
`<output_dir>/checkpoints/` for the trainer and
`<output_dir>/run_default/checkpoints/` for the orchestrator. Retention is
capped at `keep_last = 3` plus every 500-step milestone (`keep_interval = 500`);
raise `keep_last` if you want fast random-step resume and have the storage.

**Reproducibility.** Two consecutive runs of either shipped config will NOT
be bit-equivalent. Sources of non-determinism:
- Sampling seeds: `[orchestrator] seed` defaults to 42 (fixed). But
  `[orchestrator.train.sampling] seed`, `[orchestrator.eval.sampling] seed`,
  and `[inference] seed` are all unset by default — vLLM samples differently
  each run. Pin these to integers if you need deterministic rollouts.
- NCCL all-reduce ordering across ranks produces ~1e-6 numerical drift even
  with seeds pinned.
- FA4 (flash_attn.cute) is non-deterministic by design (atomic adds in the
  backward).

In practice, expect step-level metrics to track within ~1% across runs but
not match exactly. If you need to bisect a regression, set every seed
above to the same integer and disable FA4 (`attn = "sdpa"`) for the
comparison run.

**Postmortem when wandb is off.** Both shipped configs ship with `[wandb]`
commented out. Run state persists locally under `[output_dir]` from the
TOML — checkpoints in `<output_dir>/checkpoints/`, rollout JSONLs in
`<output_dir>/run_default/rollouts/`, orchestrator + trainer stdout in the
slurm log file (path printed by the sbatch). Uncomment `[wandb]` and set
`WANDB_API_KEY` in `.env` for cloud logging.

**Multi-node.** The shipped configs target a single 8-GPU node via
`SingleNodeDeploymentConfig` (`num_train_gpus`/`num_infer_gpus`). To scale
out, **copy the config to a new file** so the single-node smoke target
stays intact (e.g. `cp configs/scalerl_terminal_bench/rl.toml
configs/scalerl_terminal_bench/rl_multinode.toml`), then in the copy
switch `[deployment]` to the multi-node schema AND add a `[slurm]` block.
The validator (`src/prime_rl/configs/rl.py: validate_deployment`)
hard-rejects multi-node without `[slurm]`:
```toml
[deployment]
type = "multi_node"
num_train_nodes = 4
num_infer_nodes = 4
gpus_per_node = 8

[slurm]
partition = "<your-gpu-partition>"
account = "<your-slurm-account>"   # omit if your cluster doesn't require it
time = "24:00:00"                  # field name is `time`, NOT `time_limit`
```
For multi-node runs you do NOT use `sbatch scripts/scalerl_smoke.sbatch`
(it's a single-node `--nodes=1` template). The multi-node entrypoint
submits its own sbatch internally — invoke it directly as:
```bash
uv run rl @ configs/scalerl_terminal_bench/rl_multinode.toml
```
See `docs/slurm.md` for the full `[slurm]` knob list (rendezvous host/port,
`exclude_nodes`, `qos`, etc.). Multi-node NCCL weight broadcast still
requires the experimental override (already on in the shipped configs);
on hardware without EFA set `SCALERL_NO_EFA=1` in your `.env` and switch
`[weight_broadcast] type = "filesystem"` (then drop the override).

## Configs

Two reference configs ship with the recipe wired up:

- **`configs/scalerl_math/rl.toml`** — internal smoke. Qwen3-8B on the `math-env` (verifiers'
  built-in `PrimeIntellect/Hendrycks-Math` env). Submit with
  `sbatch scripts/scalerl_smoke.sbatch`.
- **`configs/scalerl_terminal_bench/rl.toml`** — Snowflake POC handoff config. Qwen3-8B on
  the in-tree Terminal-Bench Harbor tasks (`environments/terminal_bench/`). 28 usable
  committed tasks split 18 train / 10 test. Requires a Docker daemon reachable from the
  rollout process (host or `/var/run/docker.sock` bind-mounted into the container).

Both configs exercise the full ScaleRL knob set (CISPO, prompt-level averaging, batch-level
advantage normalization, FP32 LM-head on both sides, NPR @ 0.9, `max_async_level=8` with the
NCCL experimental override).

## CI on this fork

Only the `style.yaml` workflow runs on push/PR to this fork. Every other workflow
(`cpu_tests`, `gpu_tests`, `nightly_tests`, `release`, `sync-docs`, `devx_tag`) is gated on
`github.repository == 'PrimeIntellect-ai/prime-rl'` so the fork doesn't (a) try to invoke
upstream secrets it doesn't have or (b) push tags / docs into upstream's namespace. To run
the test suite locally before pushing:

```bash
uv run pytest -m 'not gpu' tests/unit -q
```

The `style.yaml` ruff job will still lint every push.

## Recipe references

Each ingredient cites its origin in the config docstrings. Primary sources:

- ScaleRL: Khatri et al. 2025, *The Art of Scaling Reinforcement Learning Compute for LLMs*, arXiv:2510.13786.
- CISPO origin: Minimax-M1 §3.1, *MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention*, arXiv:2506.13585.
- Prompt-level loss averaging: DAPO §3.3, arXiv:2503.14476.
- Batch-level advantage normalization: Hu et al., *Reinforce++*, also Magistral.
- No-Positive-Resampling: Polaris (An et al., 2025), 0.9 threshold.
- FP32 LM-head: Minimax-M1 §3.2.

