"""Parser tests — happy path, malformed inputs, and structural edge cases.

The parser must never raise: every malformed input becomes a ``ParseError``
with a reason string the loop can feed back to the model.
"""

from __future__ import annotations

from agent.parser import FinalAnswer, ParseError, ToolCalls, parse


def test_well_formed_single_tool_call():
    out = parse(
        """
        <tool_use>
          <name>read_paper</name>
          <input>{"id": "2509.05591"}</input>
        </tool_use>
        """
    )
    assert isinstance(out, ToolCalls)
    assert len(out.calls) == 1
    assert out.calls[0].name == "read_paper"
    assert out.calls[0].input == {"id": "2509.05591"}


def test_multiple_tool_calls_in_one_response():
    out = parse(
        """
        I'll fetch both papers in parallel.

        <tool_use>
          <name>read_paper</name>
          <input>{"id": "a"}</input>
        </tool_use>

        <tool_use>
          <name>read_paper</name>
          <input>{"id": "b"}</input>
        </tool_use>
        """
    )
    assert isinstance(out, ToolCalls)
    assert [c.input["id"] for c in out.calls] == ["a", "b"]


def test_final_answer_block():
    out = parse(
        """
        Some prose the system ignores.
        <final_answer>
        The three highlights are A, B, and C.
        </final_answer>
        """
    )
    assert isinstance(out, FinalAnswer)
    assert out.text.startswith("The three highlights")


def test_final_answer_wins_over_tool_calls_in_same_response():
    """If the model emits both, the documented convention is FA wins."""
    out = parse(
        """
        <tool_use><name>x</name><input>{}</input></tool_use>
        <final_answer>Done.</final_answer>
        """
    )
    assert isinstance(out, FinalAnswer)
    assert out.text == "Done."


def test_no_xml_at_all_is_fallback_final_answer():
    """Documented in _tool_calling_format.md: no XML → treat as final answer."""
    out = parse("Just a plain text response with no tags.")
    assert isinstance(out, FinalAnswer)
    assert out.text == "Just a plain text response with no tags."


def test_malformed_json_trailing_comma_is_tolerated():
    """Lenient cleanup: a trailing comma before } gets fixed."""
    out = parse(
        """
        <tool_use>
          <name>echo</name>
          <input>{"message": "hi",}</input>
        </tool_use>
        """
    )
    assert isinstance(out, ToolCalls)
    assert out.calls[0].input == {"message": "hi"}


def test_malformed_json_unrecoverable_returns_parse_error():
    """Single quotes are NOT auto-fixed (would break apostrophes in strings)."""
    out = parse(
        """
        <tool_use>
          <name>echo</name>
          <input>{'message': 'hi'}</input>
        </tool_use>
        """
    )
    assert isinstance(out, ParseError)
    assert "JSON" in out.reason or "json" in out.reason


def test_input_must_be_a_json_object_not_a_list():
    out = parse(
        """
        <tool_use>
          <name>x</name>
          <input>["a", "b"]</input>
        </tool_use>
        """
    )
    assert isinstance(out, ParseError)
    assert "JSON object" in out.reason
    assert "list" in out.reason.lower()


def test_missing_name_element():
    out = parse(
        """
        <tool_use>
          <input>{"id": "a"}</input>
        </tool_use>
        """
    )
    assert isinstance(out, ParseError)
    assert "<name>" in out.reason


def test_missing_input_element():
    out = parse(
        """
        <tool_use>
          <name>read_paper</name>
        </tool_use>
        """
    )
    assert isinstance(out, ParseError)
    assert "<input>" in out.reason
    assert "read_paper" in out.reason


def test_missing_closing_tag_is_diagnosed():
    """A dangling <tool_use> should produce a useful error, not just 'no XML'."""
    out = parse(
        """
        <tool_use>
          <name>x</name>
          <input>{"id": "a"}</input>
        """
    )
    assert isinstance(out, ParseError)
    assert "closing" in out.reason.lower() or "no matching" in out.reason.lower()


def test_response_wrapped_in_code_fence_still_parses():
    """Models occasionally wrap the whole response in ```xml ... ```."""
    out = parse(
        """```xml
<tool_use>
  <name>echo</name>
  <input>{"message": "hi"}</input>
</tool_use>
```"""
    )
    assert isinstance(out, ToolCalls)
    assert out.calls[0].name == "echo"


def test_empty_final_answer_is_parse_error():
    out = parse("<final_answer>   </final_answer>")
    assert isinstance(out, ParseError)
    assert "empty" in out.reason.lower()


def test_first_malformed_tool_use_short_circuits_with_useful_reason():
    """When one block in a batch is malformed, error identifies *which* block."""
    out = parse(
        """
        <tool_use>
          <name>good</name>
          <input>{"x": 1}</input>
        </tool_use>
        <tool_use>
          <name>bad</name>
          <input>not json</input>
        </tool_use>
        """
    )
    assert isinstance(out, ParseError)
    assert "bad" in out.reason  # names the offending tool
    assert "#2" in out.reason  # identifies position
