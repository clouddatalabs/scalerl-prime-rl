"""Pin the paper-faithful-vs-engineering branch on empty-after-filter batches.

The runtime loop in `orchestrator.orchestrate` reads
`_resolve_max_empty_batch_attempts(...)` to decide whether to retry generation
or fall straight through to the evicted.txt + RuntimeError path. Config-load
tests already pin the flag's default and accept paths; these tests pin the
mapping from the flag to the attempt budget.
"""

from prime_rl.orchestrator.orchestrator import (
    MAX_EMPTY_BATCH_ATTEMPTS,
    _resolve_max_empty_batch_attempts,
)


def test_default_uses_engineering_retry_budget():
    """Off-by-default keeps the upstream retry-then-crash guardrail."""
    assert _resolve_max_empty_batch_attempts(False) == MAX_EMPTY_BATCH_ATTEMPTS
    assert MAX_EMPTY_BATCH_ATTEMPTS > 1, "engineering budget must allow at least one retry"


def test_paper_faithful_runs_single_attempt():
    """ScaleRL §3.4: zero-variance is drop-don't-refill; opt-in disables retry."""
    assert _resolve_max_empty_batch_attempts(True) == 1
