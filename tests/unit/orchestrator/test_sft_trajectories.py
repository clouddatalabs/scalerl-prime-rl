from unittest.mock import MagicMock

import verifiers as vf

from prime_rl.orchestrator.trajectories import interleave_rollout, pretokenize_rollout_trajectory


class SimpleChatTokenizer:
    def __init__(self):
        self._tok2id: dict[str, int] = {}
        self._next_id = 1

    def _id(self, token: str) -> int:
        if token not in self._tok2id:
            self._tok2id[token] = self._next_id
            self._next_id += 1
        return self._tok2id[token]

    def apply_chat_template(self, messages, add_generation_prompt=False, return_dict=False):
        del return_dict
        ids = []
        for message in messages:
            role = message.get("role", "unknown")
            ids.append(self._id(f"<|{role}|>"))
            content = message.get("content", "")
            if isinstance(content, str):
                if content:
                    ids.append(self._id(content))
            else:
                ids.append(self._id(str(content)))
        if add_generation_prompt:
            ids.append(self._id("<|assistant|>"))
        return ids


def test_interleave_rollout_missing_tokens_returns_none():
    output = vf.RolloutOutput(
        example_id=1,
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "U1"}],
                completion=[{"role": "assistant", "content": "A1"}],
                response=MagicMock(),
                tokens=None,
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="1",
                extras={},
            )
        ],
        sampling_args={"temperature": 1.0},
        error=None,
    )

    assert interleave_rollout(output) is None


def test_pretokenize_rollout_trajectory_for_sft():
    tokenizer = SimpleChatTokenizer()
    output = vf.RolloutOutput(
        example_id=42,
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "U1"}],
                completion=[{"role": "assistant", "content": "A1"}],
                response=MagicMock(),
                tokens=None,
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="1",
                extras={},
            ),
            vf.TrajectoryStep(
                prompt=[
                    {"role": "user", "content": "U1"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "U2"},
                ],
                completion=[{"role": "assistant", "content": "A2"}],
                response=MagicMock(),
                tokens=None,
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="1",
                extras={},
            ),
        ],
        sampling_args={"temperature": 1.0},
        error=None,
    )

    pretokenize_rollout_trajectory(output, tokenizer)

    rollouts = interleave_rollout(output)
    assert rollouts is not None
    assert len(rollouts) == 1

    rollout = rollouts[0]
    step1_prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "U1"}],
        add_generation_prompt=True,
    )
    step1_full_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "U1"}, {"role": "assistant", "content": "A1"}],
        add_generation_prompt=False,
    )
    step2_prompt_ids = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "U2"},
        ],
        add_generation_prompt=True,
    )
    step2_full_ids = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "U2"},
            {"role": "assistant", "content": "A2"},
        ],
        add_generation_prompt=False,
    )

    prefix_len_1 = len(step1_prompt_ids)
    prefix_len_2 = len(step2_prompt_ids)
    step1_completion_ids = step1_full_ids[prefix_len_1:]
    step2_completion_ids = step2_full_ids[prefix_len_2:]
    step1_prefix = step1_prompt_ids + step1_completion_ids
    step2_new_prompt_ids = step2_prompt_ids[len(step1_prefix) :]

    assert rollout.prompt_ids == step1_prompt_ids
    assert rollout.completion_ids == step1_completion_ids + step2_new_prompt_ids + step2_completion_ids
    assert rollout.completion_mask == (
        [True] * len(step1_completion_ids) + [False] * len(step2_new_prompt_ids) + [True] * len(step2_completion_ids)
    )
    assert rollout.completion_logprobs == [0.0] * len(rollout.completion_ids)
    # Reconstructed steps must mark `inference_logprobs_synthesized=True` so
    # downstream code can refuse to run an importance-ratio loss against
    # zero-fill logprobs. Both contributing steps had `tokens=None` and ran
    # the reconstruction path; the sample-level flag must reflect that.
    assert rollout.inference_logprobs_synthesized is True


def test_pretokenize_does_not_mark_synthesized_when_tokens_present():
    """When the chat client preserves real token data (vLLM with
    return_token_ids=true), `pretokenize_rollout_trajectory` should NOT mark
    the step synthesized — the logprobs are real generator logprobs."""
    tokenizer = SimpleChatTokenizer()
    output = vf.RolloutOutput(
        example_id=99,
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "U1"}],
                completion=[{"role": "assistant", "content": "A1"}],
                response=MagicMock(),
                tokens={
                    "prompt_ids": [1, 2, 3],
                    "prompt_mask": [False, False, False],
                    "completion_ids": [10, 11, 12],
                    "completion_mask": [True, True, True],
                    "completion_logprobs": [-0.5, -0.7, -0.9],
                    "routed_experts": None,
                },
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="1",
                extras={},
            )
        ],
        sampling_args={"temperature": 1.0},
        error=None,
    )

    pretokenize_rollout_trajectory(output, tokenizer)
    rollouts = interleave_rollout(output)
    assert rollouts is not None
    assert len(rollouts) == 1
    rollout = rollouts[0]
    # Real logprobs preserved → synthesized flag must remain False.
    assert rollout.inference_logprobs_synthesized is False
    assert rollout.completion_logprobs == [-0.5, -0.7, -0.9]
