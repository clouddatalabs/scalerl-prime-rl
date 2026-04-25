#!/bin/bash
# ScaleRL recipe smoke run on Hendrycks Math, 1 node.
#
# Usage: sbatch scripts/scalerl_smoke.sbatch
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
#SBATCH --output=/mnt/shared/scalerl-prime-rl-outputs/smoke/%j.log
#SBATCH --error=/mnt/shared/scalerl-prime-rl-outputs/smoke/%j.log

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/scalerl-prime-rl}"
mkdir -p /mnt/shared/scalerl-prime-rl-outputs/smoke

cd "$REPO_ROOT"
[ -f .env ] && source .env

# Slurm sets a minimal PATH; uv lives in /usr/local/bin on this cluster.
export PATH="/usr/local/bin:$PATH"

# Clear inherited LD_PRELOAD and LD_LIBRARY_PATH — the AMI's
# /etc/profile.d/dlami.sh sets LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib:...
# FIRST, but torch (cu128) was built against CUDA 12.8 and bundles its own
# libcublas under .venv/.../nvidia/cublas/lib/. With the host setting in front,
# the dynamic linker resolves libcublas from CUDA 12.9 — ABI drift in cublasGemmEx
# → CUBLAS_STATUS_INVALID_VALUE on B200. Baker's Docker doesn't see profile.d so
# it has a clean env; we need to clear it ourselves.
# Same story for LD_PRELOAD (points at a libcublas under /opt/dlami/nvme/venv_build/
# that doesn't exist on this node — ld.so logs warnings and ignores it, but cleaner
# to just unset).
unset LD_PRELOAD
unset LD_LIBRARY_PATH

# Mirror baker's container-side runtime env (baker/cli/docker_runtime.py
# EFA_RUNTIME_ENV). Baker runs prime-rl inside Docker and sets these on the
# container; without Docker we set them on the slurm shell. CUDA_DEVICE_MAX_CONNECTIONS=1
# in particular constrains kernel concurrency on B200 and is part of the working
# baker recipe.
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_USE_HUGE_PAGE=0
export NCCL_SOCKET_IFNAME='^lo,docker'
export NCCL_NET_GDR_LEVEL=PHB
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1

# The cluster's user shell sets HF_HUB_OFFLINE=1, which breaks first-time
# dataset/model downloads. Override here for the smoke run; set to 1 once
# everything is cached.
export HF_HUB_OFFLINE=0
# Keep HF cache on the shared filesystem so multi-node jobs share downloads.
export HF_HOME="${HF_HOME:-/mnt/shared/hf-cache}"

# Use the venv we set up in this repo. The .venv was populated with `uv sync
# --all-extras` outside slurm; we don't re-sync here to avoid network races
# between concurrent jobs.
source .venv/bin/activate
# `source activate` doesn't always rewrite PATH cleanly under slurm; export
# the venv bin path explicitly so child processes (e.g. spawned `inference`)
# resolve from it.
export PATH="$REPO_ROOT/.venv/bin:$PATH"
PYBIN="$REPO_ROOT/.venv/bin/python"

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
