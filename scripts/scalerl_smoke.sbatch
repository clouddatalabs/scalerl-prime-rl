#!/bin/bash
# ScaleRL recipe smoke run on Hendrycks Math, 1 node.
#
# Usage:
#   sbatch scripts/scalerl_smoke.sbatch
#
# Environment overrides (all optional):
#   REPO_ROOT       — clone path (default: $HOME/scalerl-prime-rl)
#   OUTPUT_ROOT     — slurm stdout/stderr LOG dir only (default: $REPO_ROOT/outputs/smoke).
#                     Run outputs (checkpoints, rollouts, run_default/) go to
#                     `[output_dir]` from the TOML, NOT here.
#   HF_HOME         — HuggingFace cache (default: $REPO_ROOT/hf-cache)
#
# To run on a partition other than `gpu`, override at submit time:
#     sbatch -p <partition> scripts/scalerl_smoke.sbatch
# (#SBATCH directives are parsed before the script body, so an exported
# PARTITION env var cannot drive --partition; use `sbatch -p` instead.)
#
#   SCALERL_CONFIG  — TOML config to launch (default: configs/scalerl_math/rl.toml).
#                     Set to configs/scalerl_terminal_bench/rl.toml for the Snowflake handoff.
#   SCALERL_RESUME  — set "1" to add `--ckpt.resume_step=-1` to the rl invocation.
#   SCALERL_FORCE_CLEAR_LD — set "1" to unset LD_LIBRARY_PATH/LD_PRELOAD even on non-AWS-DLAMI hosts
#   SCALERL_NO_EFA  — set "1" to skip the EFA libfabric env vars when your cluster has no EFA
#
# Exercises:
#   - canonical Minimax-CISPO loss (eps_max=4)
#   - prompt-level loss averaging (loss_scale_mode="sequence")
#   - batch-level advantage normalization
#   - FP32 LM-head on both trainer and inference
#   - max_async_level=8 + experimental NCCL override
#   - No-Positive-Resampling @ 0.9
#   - upstream ZeroAdvantageFilter (default filter list)

#SBATCH --job-name=scalerl-math-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --partition=gpu
# Many shared SLURM clusters reject submissions with no time limit or default to
# a MaxTime policy of 1–4h. The math smoke needs ~30–60 min to reach the first
# checkpoint at [ckpt] interval = 100; TB rollouts need several minutes/step
# until task base images are warm in the Docker layer cache, so multiply
# accordingly. Override at submit time with `sbatch -t HH:MM:SS ...` if your
# cluster's QoS forbids 12h jobs.
#SBATCH --time=12:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/scalerl-prime-rl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/smoke}"
mkdir -p "$OUTPUT_ROOT"

# Redirect job stdout+stderr ourselves so #SBATCH --output (which is evaluated
# before any commands run) doesn't need a hardcoded path. Use `>` (overwrite)
# instead of `>>` so a slurm `--requeue` of the same SLURM_JOB_ID starts a
# fresh log rather than concatenating onto the prior run's tail. Tee preserves
# console visibility for `srun --pty` / `sbatch --wait` users.
exec > >(tee "$OUTPUT_ROOT/${SLURM_JOB_ID:-local}.log") 2>&1

cd "$REPO_ROOT"

# Make `environments.terminal_bench` (and any future repo-local env package)
# importable from the orchestrator's console-script entrypoint. The
# `.venv/bin/orchestrator` shim is a Python script whose `sys.path[0]` is its
# parent dir (`.venv/bin/`), NOT `$REPO_ROOT` — `cd "$REPO_ROOT"` does not
# change that. Without this PYTHONPATH prepend, `vf.load_environment("environments.terminal_bench")`
# fails with `ModuleNotFoundError: No module named 'environments'` on a fresh
# clone. (Math smoke is unaffected because `math-env` resolves via the
# `prime-rl[envs]` extra, which IS installed into site-packages.)
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Force HF online before sourcing .env so a parent shell's HF_HUB_OFFLINE=1
# can't suppress the override silently — but let .env opt back in if the
# operator explicitly wants offline (HF_HUB_OFFLINE=1 in .env wins because we
# source after).
export HF_HUB_OFFLINE=0
[ -f .env ] && source .env

# Make sure `uv` is on PATH. Slurm sets a minimal PATH; `command -v uv` is the
# canonical way to find it. Falling back to common install locations.
if ! command -v uv >/dev/null 2>&1; then
    for candidate in /usr/local/bin /opt/uv/bin "$HOME/.local/bin"; do
        if [ -x "$candidate/uv" ]; then
            export PATH="$candidate:$PATH"
            break
        fi
    done
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: \`uv\` not found on PATH; install per https://github.com/astral-sh/uv#installation" >&2
    exit 1
fi

# AWS DLAMI puts /usr/local/cuda-12.9/lib first in LD_LIBRARY_PATH (via
# /etc/profile.d/dlami.sh) — torch (cu128) bundles its own libcublas built
# against CUDA 12.8, and the dynamic linker resolves the wrong libcublas, which
# triggers CUBLAS_STATUS_INVALID_VALUE on B200 in vLLM's compiled QKV path.
# Same shell also sets LD_PRELOAD pointing at a venv that may not exist on
# the compute node. Strip both ONLY when one of these explicit triggers fires;
# do NOT pattern-match arbitrary substrings in LD_LIBRARY_PATH / LD_PRELOAD —
# external clusters rely on those for AWS-OFI-NCCL, system cuDNN, etc., and a
# false-positive substring match silently destroys their networking config.
#   1. `SCALERL_FORCE_CLEAR_LD=1` — explicit user opt-in (.env or parent shell)
#   2. `/etc/dlami/dlami_version` — AWS DLAMI marker file (canonical detection)
if [ "${SCALERL_FORCE_CLEAR_LD:-0}" = "1" ] || [ -f /etc/dlami/dlami_version ]; then
    echo "=== AWS DLAMI / SCALERL_FORCE_CLEAR_LD detected — stripping LD_LIBRARY_PATH and LD_PRELOAD ==="
    unset LD_PRELOAD
    unset LD_LIBRARY_PATH
fi

# Container-side runtime env mirrored on the slurm shell when not running
# under Docker. Each value is only set when not already provided so .env can
# override per-cluster.
#
# EFA-only knobs are gated on a probe: setting FI_PROVIDER=efa on a cluster
# without EFA makes libfabric fail provider lookup and NCCL hangs at init.
# Set SCALERL_NO_EFA=1 to skip them explicitly.
HAS_EFA=0
if [ "${SCALERL_NO_EFA:-0}" != "1" ] && {
    [ -d /sys/class/infiniband ] && ls /sys/class/infiniband/ 2>/dev/null | grep -q efa
} ; then
    HAS_EFA=1
fi
if [ "$HAS_EFA" = "1" ]; then
    export FI_PROVIDER="${FI_PROVIDER:-efa}"
    export FI_EFA_USE_DEVICE_RDMA="${FI_EFA_USE_DEVICE_RDMA:-1}"
    export FI_EFA_USE_HUGE_PAGE="${FI_EFA_USE_HUGE_PAGE:-0}"
fi
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
# CUDA_DEVICE_MAX_CONNECTIONS=1 constrains kernel concurrency on B200; raising
# it triggers an NCCL/comm-overlap fast-path that's flaky for prime-rl's
# weight-broadcast pattern.
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

# HF cache. Default to a repo-local cache so the cluster doesn't need a writable
# home dir; if the operator pre-set HF_HOME in .env or the parent shell, honor it.
export HF_HOME="${HF_HOME:-$REPO_ROOT/hf-cache}"

# HF Hub default timeouts (10s) are too tight for a 16 GB Qwen3-8B pull
# through corporate NAT; operators saw opaque ReadTimeoutError mid-download.
# Match what `Dockerfile.cuda` already exports. Honor any operator-set
# values from `.env` or the parent shell.
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-500}"

# WANDB is opt-in. Both shipped configs ship with `[wandb]` commented out;
# if an operator uncomments it but forgets WANDB_API_KEY, wandb's interactive
# `wandb login` prompt can hang `srun` (no TTY). Default to disabled when
# no key is present, online when it is. Operators who want offline-mode
# can set `WANDB_MODE=offline` explicitly.
if [ -z "${WANDB_MODE:-}" ]; then
    if [ -n "${WANDB_API_KEY:-}" ]; then
        export WANDB_MODE="online"
    else
        export WANDB_MODE="disabled"
    fi
fi

# Offline-mode persistence. `WANDB_MODE=offline` writes run artifacts to
# `$WANDB_DIR/wandb/`; with no override that is `$CWD/wandb/`, which on
# most clusters is the compute node's local scratch — lost when SLURM
# reaps the job. Pin to `$OUTPUT_ROOT/wandb/` (shared with the slurm
# stdout log dir, persistent across jobs) so a post-mortem `wandb sync
# $OUTPUT_ROOT/wandb/offline-run-*` actually finds the data.
if [ "${WANDB_MODE}" = "offline" ] && [ -z "${WANDB_DIR:-}" ]; then
    export WANDB_DIR="$OUTPUT_ROOT"
fi

# Use the venv we set up in this repo. The .venv was populated with `uv sync
# --extra all` outside slurm; we don't re-sync here to avoid network races
# between concurrent jobs.
if [ ! -d "$REPO_ROOT/.venv" ]; then
    echo "ERROR: $REPO_ROOT/.venv missing; run 'uv sync --extra all' on the login node first." >&2
    exit 1
fi
source .venv/bin/activate
# `source activate` doesn't always rewrite PATH cleanly under slurm; export
# the venv bin path explicitly so child processes (e.g. spawned `inference`)
# resolve from it.
export PATH="$REPO_ROOT/.venv/bin:$PATH"
PYBIN="$REPO_ROOT/.venv/bin/python"

# Preflight: GPUs visible, model weights cached, FA4 stub not shadowing FA4.
GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)"
# Read the actual GPU requirement from the resolved config so this works on
# custom configs that use different splits. Falls back to 8 if the parse
# fails (matches both shipped POC configs: 4 train + 4 infer = 8).
GPU_REQUIRED="$("$PYBIN" - <<EOF 2>/dev/null
import tomllib, sys
try:
    with open("$SCALERL_CONFIG", "rb") as f:
        cfg = tomllib.load(f)
    d = cfg.get("deployment", {})
    if d.get("type", "single_node") == "single_node":
        n = int(d.get("num_train_gpus", 4)) + int(d.get("num_infer_gpus", 4))
    else:
        # Multi-node uses gpus_per_node × (num_train_nodes + num_infer_nodes)
        n = int(d.get("gpus_per_node", 8)) * (
            int(d.get("num_train_nodes", 1)) + int(d.get("num_infer_nodes", 1))
        )
    print(n)
except Exception:
    print(8)
EOF
)"
GPU_REQUIRED="${GPU_REQUIRED:-8}"
if [ "$GPU_COUNT" -lt "$GPU_REQUIRED" ]; then
    echo "ERROR: nvidia-smi reports $GPU_COUNT GPUs but $SCALERL_CONFIG requires $GPU_REQUIRED." >&2
    echo "       Override [deployment] in the config or run on a node with enough GPUs." >&2
    exit 1
fi

# Repair flash-attn-cute (FA4) namespace collision if `uv sync --extra all`
# happened to install `flash-attn` after `flash-attn-cute`. Both ship a
# `flash_attn/cute/` sub-package; the one from `flash-attn` is a tiny stub
# that shadows the real FA4 module. The repair script reinstalls FA4 from
# git — it needs network. On clusters whose compute nodes can't reach
# github.com, run `bash scripts/fix-flash-attn-cute.sh` on the login node
# once after `uv sync --extra all` and the check below will short-circuit.
if ! "$PYBIN" -c "
import sys
import flash_attn.cute.interface as fa4
with open(fa4.__file__) as f:
    sys.exit(0 if sum(1 for _ in f) > 1000 else 1)
" 2>/dev/null; then
    echo "=== flash-attn-cute namespace repair (uv sync ordering) ==="
    if ! bash "$REPO_ROOT/scripts/fix-flash-attn-cute.sh"; then
        echo "ERROR: flash-attn-cute repair failed (likely network unreachable from compute node)." >&2
        echo "       Run 'bash scripts/fix-flash-attn-cute.sh' on the login node and resubmit." >&2
        exit 1
    fi
fi

# Config path. Set SCALERL_CONFIG to launch a different config (e.g. the
# Snowflake handoff): `sbatch --export=ALL,SCALERL_CONFIG=configs/scalerl_terminal_bench/rl.toml ...`
# Accepts both absolute paths and paths relative to $REPO_ROOT.
SCALERL_CONFIG="${SCALERL_CONFIG:-configs/scalerl_math/rl.toml}"
if [ -f "$SCALERL_CONFIG" ]; then
    : # absolute or already-resolved path
elif [ -f "$REPO_ROOT/$SCALERL_CONFIG" ]; then
    SCALERL_CONFIG="$REPO_ROOT/$SCALERL_CONFIG"
else
    echo "ERROR: SCALERL_CONFIG=$SCALERL_CONFIG not found (tried as-is and under \$REPO_ROOT=$REPO_ROOT)" >&2
    exit 1
fi

# Print effective config (helpful for postmortem) before launching.
echo "=== effective ScaleRL config ($SCALERL_CONFIG) ==="
"$PYBIN" -c "
import sys, tomli
from prime_rl.configs.rl import RLConfig
cfg = RLConfig.model_validate(tomli.loads(open(sys.argv[1],'rb').read().decode()))
print(' loss type:               ', cfg.trainer.loss.type)
print(' loss eps_max:            ', cfg.trainer.loss.eps_max)
print(' loss scale mode:         ', cfg.trainer.loss.loss_scale_mode)
print(' adv normalization:       ', cfg.orchestrator.advantage.normalization)
print(' prompt_average_loss:     ', cfg.orchestrator.prompt_average_loss)
print(' no_positive_resampling:  ', cfg.orchestrator.buffer.no_positive_resampling, '@', cfg.orchestrator.buffer.no_positive_resampling_threshold)
print(' fp32_lm_head trainer:    ', cfg.trainer.model.fp32_lm_head)
print(' fp32_lm_head inference:  ', cfg.inference.model.fp32_lm_head)
print(' max_async_level:         ', cfg.orchestrator.max_async_level)
print(' strict_async_level:      ', cfg.orchestrator.strict_async_level)
print(' max_off_policy_steps:    ', cfg.orchestrator.max_off_policy_steps)
print(' allow_nccl_async_override:', cfg.orchestrator.experimental.allow_nccl_async_level_override)
print(' attn:                    ', cfg.trainer.model.attn)
print(' optim_dtype:             ', cfg.trainer.model.optimization_dtype)
print(' optim eps:               ', cfg.trainer.optim.eps)
print(' scheduler:               ', cfg.trainer.scheduler.type, 'warmup=', getattr(cfg.trainer.scheduler, 'warmup_steps', '-'))
print(' batch_size:              ', cfg.orchestrator.batch_size)
print(' rollouts_per_example:    ', cfg.orchestrator.rollouts_per_example)
" "$SCALERL_CONFIG"
echo "================================="

# Use the venv's `rl` entry-point directly. `uv run` is unreliable here because
# `source .venv/bin/activate` does not always rewrite PATH cleanly under slurm.
RL_ARGS=()
if [ "${SCALERL_RESUME:-0}" = "1" ]; then
    # Fail-fast if there's no checkpoint to resume from. Otherwise prime-rl
    # logs a `WARNING: no checkpoint found` and silently restarts from
    # step 0 — wiping prior rollouts and broadcasts in the same output_dir
    # via `clean_future_steps(resume_step=-1)`. That's a footgun the user
    # almost never wants when they explicitly asked to resume.
    #
    # Resolve in Python: honor `[ckpt] output_dir` override and absolute
    # paths. Print three lines: ckpt_dir, interval, then a sentinel.
    PRELOAD="$("$PYBIN" - <<PYEOF "$SCALERL_CONFIG" "$REPO_ROOT"
import sys, tomli
from pathlib import Path
from prime_rl.configs.rl import RLConfig
cfg = RLConfig.model_validate(tomli.loads(open(sys.argv[1],'rb').read().decode()))
ckpt_root = cfg.ckpt.output_dir if (cfg.ckpt and cfg.ckpt.output_dir is not None) else cfg.output_dir
ckpt_root = Path(str(ckpt_root))
if not ckpt_root.is_absolute():
    ckpt_root = (Path(sys.argv[2]) / ckpt_root).resolve()
print(ckpt_root / "checkpoints")
print(cfg.ckpt.interval if cfg.ckpt else 100)
PYEOF
)"
    CKPT_DIR="$(echo "$PRELOAD" | sed -n 1p)"
    CKPT_INTERVAL="$(echo "$PRELOAD" | sed -n 2p)"
    if [ -z "$(find "$CKPT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'step_*' 2>/dev/null)" ]; then
        echo "ERROR: SCALERL_RESUME=1 but no checkpoints found under $CKPT_DIR." >&2
        echo "       The first checkpoint lands at [ckpt] interval (= $CKPT_INTERVAL) steps;" >&2
        echo "       a run that crashed before that has nothing to resume from." >&2
        echo "       Resubmit without SCALERL_RESUME=1 to start fresh, or rm -rf the output_dir if you" >&2
        echo "       want a clean restart in place." >&2
        exit 1
    fi
    RL_ARGS+=(--ckpt.resume_step=-1)
fi
"$REPO_ROOT/.venv/bin/rl" @ "$SCALERL_CONFIG" "${RL_ARGS[@]}"
