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

# Use the venv we set up in this repo. The .venv was populated with `uv sync
# --all-extras` outside slurm; we don't re-sync here to avoid network races
# between concurrent jobs.
source .venv/bin/activate

# Print effective config (helpful for postmortem) before launching.
echo "=== effective ScaleRL config ==="
python -c "
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

uv run rl @ configs/scalerl_math/rl.toml
