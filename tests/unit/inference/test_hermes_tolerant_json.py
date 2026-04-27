"""Unit tests for the Hermes-tool-parser tolerant-JSON rescue patch.

See ``prime_rl.inference.patches.monkey_patch_hermes_tool_parser_tolerant_json``
for background. The rescue is intended to recover tool calls whose JSON body
carries unescaped shell regex backslashes (``\\.``, ``\\+``, ``\\-``, …); it
must be a no-op on already-valid JSON.
"""

import json

import pytest

from prime_rl.inference.patches import (
    _fix_invalid_json_escapes,
    monkey_patch_hermes_tool_parser_tolerant_json,
)

# Real failure body captured from run
# ``20260422_205716__xuchao__rl-gpu-local`` step 17 rollout 0 turn 2. Model
# emitted this inside a ``<tool_call>...</tool_call>`` wrapper; original
# Hermes parser dropped it with ``Invalid \escape: line 1 column 114``.
BASE64_RECOVERY_BODY = (
    '{"name": "shell", "arguments": {"command": "mkdir -p /app/recovered '
    "&& find /app/sensitive_data/ -type f -name '*\\.b64_content' | while "
    'read file; do original_name=\\"$(basename \\"$file\\" | sed \'s/\\\\.b64_content$//\')\\"; '
    'original_name_decoded=\\"$(echo \\"$original_name\\" | base64 --decode)\\"; '
    'content=\\"$(cat \\"$file\\" | base64 --decode)\\"; '
    'echo \\"$content\\" > /app/recovered/\\"$original_name_decoded\\"; done"}}'
)


def test_passthrough_on_valid_json():
    """Fast path: valid JSON must be returned byte-identical."""
    body = '{"name": "shell", "arguments": {"command": "ls -la"}}'
    assert _fix_invalid_json_escapes(body) == body
    # And it must still parse.
    assert json.loads(_fix_invalid_json_escapes(body)) == json.loads(body)


def test_passthrough_preserves_valid_escapes():
    """``\\"``, ``\\\\``, ``\\n``, ``\\t``, ``\\/``, ``\\uXXXX`` must not be rewritten."""
    body = (
        '{"s": "a\\"b", "backslash": "c\\\\d", "newline": "e\\nf", '
        '"tab": "g\\th", "slash": "i\\/j", "unicode": "\\u00e9"}'
    )
    assert _fix_invalid_json_escapes(body) == body
    # Round-trip: original parses, and the rewritten version parses to same.
    assert json.loads(body) == json.loads(_fix_invalid_json_escapes(body))


def test_rewrites_shell_regex_escape():
    """``\\.`` inside a string is invalid JSON; fixer must double the backslash."""
    broken = '{"command": "ls *\\.txt"}'
    with pytest.raises(json.JSONDecodeError):
        json.loads(broken)
    fixed = _fix_invalid_json_escapes(broken)
    parsed = json.loads(fixed)
    # The decoded string must contain the literal backslash the model intended.
    assert parsed == {"command": "ls *\\.txt"}


def test_rewrites_multiple_invalid_escapes():
    """``grep -E``-style expressions bring many invalid escapes at once."""
    broken = '{"cmd": "grep -E \'^[0-9]+[\\+\\-\\*\\(\\)]+$\' file.txt"}'
    with pytest.raises(json.JSONDecodeError):
        json.loads(broken)
    parsed = json.loads(_fix_invalid_json_escapes(broken))
    # Each backslash in the decoded string is preserved as-is (literal).
    assert parsed["cmd"] == "grep -E '^[0-9]+[\\+\\-\\*\\(\\)]+$' file.txt"


def test_rewrites_captured_production_failure():
    """Regression: the exact body captured from step 17 rollout 0."""
    with pytest.raises(json.JSONDecodeError):
        json.loads(BASE64_RECOVERY_BODY)
    parsed = json.loads(_fix_invalid_json_escapes(BASE64_RECOVERY_BODY))
    assert parsed["name"] == "shell"
    assert "find /app/sensitive_data/" in parsed["arguments"]["command"]


def test_trailing_backslash_in_string():
    """Degenerate case: string literal ends with a lone backslash."""
    broken = '{"s": "abc\\"}'
    # Upstream json.loads treats this as ``\"`` (escaped quote) and raises
    # a different error (unterminated string). The fixer should leave it
    # parseable if possible; at minimum it must not crash.
    fixed = _fix_invalid_json_escapes(broken)
    assert isinstance(fixed, str)


def test_escaped_quote_does_not_toggle_string_state():
    """``\\"`` inside a string must not flip in_string off prematurely; any
    ``\\X`` that follows it is still inside the string and still subject
    to rewriting."""
    # Model emits:  {"s": "say \"hi\" then \.bye"}
    # The trailing \.  must be doubled; the preceding \" pair must be left
    # alone (otherwise we'd break a legal escape).
    broken = '{"s": "say \\"hi\\" then \\.bye"}'
    with pytest.raises(json.JSONDecodeError):
        json.loads(broken)
    parsed = json.loads(_fix_invalid_json_escapes(broken))
    assert parsed == {"s": 'say "hi" then \\.bye'}


def test_escape_outside_string_untouched():
    """Structural backslashes outside a string (already invalid JSON) must
    not be rewritten - doing so would only mask an unrelated bug."""
    # ``\`` between two tokens, outside any string literal.
    broken = '{"a": 1}\\nextra'
    out = _fix_invalid_json_escapes(broken)
    # The backslash after ``}`` is outside any string and is left alone
    # (still invalid JSON on re-parse, but not our problem to fix).
    assert out == '{"a": 1}\\nextra'


# -------------------------------------------------------------------------
# Integration: drive the patched Hermes parser end-to-end with a fake
# tokenizer. We verify both the rescue path (malformed body) and the
# passthrough path (well-formed body) against the same parser instance.
# -------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer stand-in.

    ``Hermes2ProToolParser.__init__`` calls ``tokenizer.encode`` and
    ``tokenizer.decode`` to determine the multi-token ``<tool_call>``
    sequence. For ``extract_tool_calls`` itself (the method we're patching)
    only the string-level tokens are used, so any stable mapping suffices.
    """

    def encode(self, text, add_special_tokens=False):
        # Single "virtual" token per distinct start/end string. Real value
        # doesn't matter for extract_tool_calls.
        return [hash(text) & 0xFFFF]

    def decode(self, token_ids):
        return "<tool_call>" if token_ids == [hash("<tool_call>") & 0xFFFF] else "</tool_call>"


def _build_parser():
    from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser

    monkey_patch_hermes_tool_parser_tolerant_json()
    tok = _FakeTokenizer()
    # Bypass the full __init__; set the minimal state extract_tool_calls reads.
    parser = Hermes2ProToolParser.__new__(Hermes2ProToolParser)
    parser.model_tokenizer = tok
    import regex as re

    parser.tool_call_start_token = "<tool_call>"
    parser.tool_call_end_token = "</tool_call>"
    parser.tool_call_regex = re.compile(
        r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL
    )
    return parser


def test_patched_parser_rescues_invalid_escape_body():
    parser = _build_parser()
    output = f"<tool_call>\n{BASE64_RECOVERY_BODY}\n</tool_call>"
    result = parser.extract_tool_calls(output, request=None)
    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.function.name == "shell"
    # ``arguments`` is re-serialized to JSON by the parser - round-trip to dict.
    args = json.loads(call.function.arguments)
    assert "find /app/sensitive_data/" in args["command"]


def test_patched_parser_passthrough_on_valid_body():
    parser = _build_parser()
    body = '{"name": "shell", "arguments": {"command": "ls -la /app"}}'
    output = f"<tool_call>\n{body}\n</tool_call>"
    result = parser.extract_tool_calls(output, request=None)
    assert result.tools_called is True
    assert result.tool_calls[0].function.name == "shell"
    assert json.loads(result.tool_calls[0].function.arguments) == {"command": "ls -la /app"}


def test_patched_parser_no_tool_call_token_fast_path():
    """When ``<tool_call>`` isn't present, behavior must match upstream:
    ``tools_called=False`` and the full model output becomes ``content``."""
    parser = _build_parser()
    output = "Just a plain response, no tool call here."
    result = parser.extract_tool_calls(output, request=None)
    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == output


def test_patched_parser_is_idempotent():
    """Double-applying the patch must not re-wrap an already-wrapped method."""
    from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser

    monkey_patch_hermes_tool_parser_tolerant_json()
    first = Hermes2ProToolParser.extract_tool_calls
    monkey_patch_hermes_tool_parser_tolerant_json()
    second = Hermes2ProToolParser.extract_tool_calls
    assert first is second
    assert getattr(second, "_prime_rl_tolerant_json_patch", False) is True
