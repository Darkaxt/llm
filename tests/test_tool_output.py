import copy
import json

import pytest

from llm.chat_ui import tool_output_text


@pytest.mark.parametrize(
    "value, expected",
    [
        (
            '{"rows":[{"count":7}],"truncated":false}',
            {"rows": [{"count": 7}], "truncated": False},
        ),
        (json.dumps('{"rows":[{"count":7}]}'), {"rows": [{"count": 7}]}),
        (
            [{"type": "tool_result", "text": '{"rows":[{"count":7}],"error":null}'}],
            [{"type": "tool_result", "text": {"rows": [{"count": 7}], "error": None}}],
        ),
        (
            {
                "content": [
                    {
                        "type": "text",
                        "text": '{"unicode":"café ☕","details":"[1,true,null]"}',
                    }
                ],
                "isError": False,
            },
            {
                "content": [
                    {
                        "type": "text",
                        "text": {"unicode": "café ☕", "details": [1, True, None]},
                    }
                ],
                "isError": False,
            },
        ),
        (
            '{"duration":0.125,"id":123456789012345678901234567890}',
            {"duration": 0.125, "id": 123456789012345678901234567890},
        ),
    ],
)
def test_json_text_is_expanded_for_display_without_mutating_payload(value, expected):
    original = copy.deepcopy(value)
    rendered = tool_output_text(value)
    assert rendered == json.dumps(expected, ensure_ascii=False, indent=2)
    assert value == original


@pytest.mark.parametrize(
    "value",
    [
        "ordinary\n  text",
        "",
        "123",
        "true",
        "null",
        '"ordinary quoted text"',
        '{"broken":',
        '{"duplicate":1,"duplicate":2}',
        '{"fraction":0.12345678901234567890123456789}',
        '{"large":1e1000}',
        '{"invalid":NaN}',
    ],
)
def test_noncontainer_invalid_or_lossy_json_strings_remain_exact(value):
    assert tool_output_text(value) == value
    # Unsafe embedded strings keep their exact value instead of losing data.
    wrapped = {"text": value, "error": "retained"}
    assert json.loads(tool_output_text(wrapped)) == wrapped
