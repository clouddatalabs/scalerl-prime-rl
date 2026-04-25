import logging
import os

from prime_rl.inference.patches import (
    monkey_patch_LRUCacheWorkerLoRAManager,
    monkey_patch_minimax_m2_for_lora,
    monkey_patch_no_moe_lora,
)

logger = logging.getLogger(__name__)

# Each patch reaches into vLLM internals (`vllm.lora.worker_manager`,
# `vllm.model_executor.models.minimax_m2`, etc.). A downstream consumer
# pinned to a different vLLM version may have renamed or removed those
# symbols — without isolation, an ImportError in any patch would crash
# the vLLM worker on boot, even when serving a model that doesn't need
# the patched code path (e.g. Qwen3-8B without LoRA, no MiniMax). Wrap
# each patch independently so a stale-pin failure surfaces as a logged
# warning rather than a hard worker-boot crash.
for patch_name, patch_fn, condition in [
    ("monkey_patch_LRUCacheWorkerLoRAManager", monkey_patch_LRUCacheWorkerLoRAManager, True),
    ("monkey_patch_minimax_m2_for_lora", monkey_patch_minimax_m2_for_lora, True),
    (
        "monkey_patch_no_moe_lora",
        monkey_patch_no_moe_lora,
        os.environ.get("PRIME_NO_MOE_LORA") == "1",
    ),
]:
    if not condition:
        if patch_name == "monkey_patch_no_moe_lora":
            logger.info("PRIME_NO_MOE_LORA=0: no patch applied")
        continue
    try:
        patch_fn()
        if patch_name == "monkey_patch_no_moe_lora":
            logger.info("PRIME_NO_MOE_LORA=1: disabling LoRA on MoE layers")
    except Exception as e:
        logger.warning(
            "prime-rl: vLLM worker-boot patch %s failed (%s: %s); "
            "the worker continues. If your model relies on this patch, "
            "fix the underlying error.",
            patch_name,
            type(e).__name__,
            e,
        )
