#!/bin/bash
# ScaleRL recipe smoke run on Hendrycks Math, 1 node.
#
# Usage:
#   sbatch scripts/scalerl_smoke.sbatch
#
# Environment overrides (all optional):
#   REPO_ROOT       — clone path (default: $HOME/scalerl-prime-rl)
#   OUTPUT_ROOT     — slurm log + run output dir (default: $REPO_ROOT/outputs/smoke)
#   HF_HOME         — HuggingFace cache (default: $REPO_ROOT/hf-cache)
#   PARTITION       — slurm partition (default: gpu); pass via `sbatch -p <name>` if you don't want to edit this file
#   SCALERL_FORCE_CLEAR_LD — set "1" to unset LD_LIBRARY_PATH/LD_PRELOAD even on non-AWS-DLAMI hosts
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

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/scalerl-prime-rl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/smoke}"
mkdir -p "$OUTPUT_ROOT"

# Redirect job stdout+stderr ourselves so #SBATCH --output (which is evaluated
# before any commands run) doesn't need a hardcoded path. Tee preserves
# console visibility for `srun --pty` / `sbatch --wait` users.
exec > >(tee -a "$OUTPUT_ROOT/${SLURM_JOB_ID:-local}.log") 2>&1

cd "$REPO_ROOT"
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
# Same shell also sets LD_PRELOAD pointing at a venv that may not exist on the
# compute node. Strip both ONLY when we detect the DLAMI marker (or when the
# user opts in via SCALERL_FORCE_CLEAR_LD=1) so we don't break clusters that
# legitimately rely on LD_LIBRARY_PATH for AWS-OFI-NCCL, system cuDNN, etc.
DLAMI_MARKER="dlami"
if [ "${SCALERL_FORCE_CLEAR_LD:-0}" = "1" ] \
    || [ -f /etc/dlami/dlami_version ] \
    || { [ -n "${LD_LIBRARY_PATH:-}" ] && [[ "$LD_LIBRARY_PATH" == *"/usr/local/cuda-12.9/"* ]]; } \
    || { [ -n "${LD_PRELOAD:-}" ] && [[ "$LD_PRELOAD" == *"venv_build"* ]]; }; then
    echo "=== AWS DLAMI detected — stripping LD_LIBRARY_PATH and LD_PRELOAD ==="
    unset LD_PRELOAD
    unset LD_LIBRARY_PATH
fi

# Mirror baker's container-side runtime env (baker/cli/docker_runtime.py
# EFA_RUNTIME_ENV). Baker runs prime-rl inside Docker and sets these on the
# container; without Docker we set them on the slurm shell. CUDA_DEVICE_MAX_CONNECTIONS=1
# in particular constrains kernel concurrency on B200 and is part of the working
# baker recipe. Override any of these via .env in the repo root if your cluster
# differs (e.g. NCCL_SOCKET_IFNAME for a different interface naming).
export FI_PROVIDER="${FI_PROVIDER:-efa}"
export FI_EFA_USE_DEVICE_RDMA="${FI_EFA_USE_DEVICE_RDMA:-1}"
export FI_EFA_USE_HUGE_PAGE="${FI_EFA_USE_HUGE_PAGE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

# Many cluster shells set HF_HUB_OFFLINE=1; that breaks first-time downloads.
# Override here for the smoke run; set to 1 once everything is cached.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/hf-cache}"

# Use the venv we set up in this repo. The .venv was populated with `uv sync
# --all-extras` outside slurm; we don't re-sync here to avoid network races
# between concurrent jobs.
if [ ! -d "$REPO_ROOT/.venv" ]; then
    echo "ERROR: $REPO_ROOT/.venv missing; run 'uv sync --all-extras' on the login node first." >&2
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
if [ "$GPU_COUNT" -lt 8 ]; then
    echo "WARNING: nvidia-smi reports $GPU_COUNT GPUs; the math smoke config wants 4 train + 4 infer = 8."
fi

# Repair flash-attn-cute (FA4) namespace collision if `uv sync --all-extras`
# happened to install `flash-attn` after `flash-attn-cute`. Both ship a
# `flash_attn/cute/` sub-package; the one from `flash-attn` is a tiny stub
# that shadows the real FA4 module. The repair script reinstalls FA4 from
# git — it needs network. On clusters whose compute nodes can't reach
# github.com, run `bash scripts/fix-flash-attn-cute.sh` on the login node
# once after `uv sync --all-extras` and the check below will short-circuit.
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

# Print effective config (helpful for postmortem) before launching.
echo "=== effective ScaleRL config ==="
"$PYBIN" -c "
import tomli
from prime_rl.configs.rl import RLConfig
cfg = RLConfig.model_validate(tomli.loads(open('configs/scalerl_math/rl.toml','rb').read().decode()))
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
"
echo "================================="

# Use the venv's `rl` entry-point directly. `uv run` is unreliable here because
# `source .venv/bin/activate` does not always rewrite PATH cleanly under slurm.
"$REPO_ROOT/.venv/bin/rl" @ configs/scalerl_math/rl.toml
