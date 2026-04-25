import os

from prime_rl.configs.inference import InferenceConfig
from prime_rl.utils.config import cli


def setup_vllm_env(config: InferenceConfig):
    """Set vLLM environment variables based on config. Must be called before importing vLLM."""

    # spawn is more robust in vLLM nightlies and Qwen3-VL (fork can deadlock
    # with multithreaded processes). Use plain assignment, NOT `setdefault` —
    # otherwise an operator who exported `VLLM_WORKER_MULTIPROC_METHOD=fork`
    # (e.g. for an unrelated debugging session) silently re-introduces the
    # deadlock class this guard exists to prevent. The asymmetry vs the
    # `PRIME_RL_VLLM_FP32_LM_HEAD` line below (which correctly uses plain
    # assignment) was the bug.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    if config.enable_lora:
        os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"

    # Read by prime_rl.inference.patches.monkey_patch_vllm_fp32_lm_head, registered via the
    # vllm.general_plugins entry point in pyproject.toml. Trainer side must match.
    os.environ["PRIME_RL_VLLM_FP32_LM_HEAD"] = "1" if config.model.fp32_lm_head else "0"


def main():
    config = cli(InferenceConfig)
    setup_vllm_env(config)

    # We import here to be able to set environment variables before importing vLLM
    from prime_rl.inference.vllm.server import server  # pyright: ignore

    server(config, vllm_extra=config.vllm_extra)


if __name__ == "__main__":
    main()
