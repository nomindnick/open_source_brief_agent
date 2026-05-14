"""Unit tests for the llama.cpp adapter's pure logic.

The HTTP path is exercised by the smoke script against a live server;
these tests cover the thinking-extraction regex in isolation.
"""

from __future__ import annotations

from agent.model.llamacpp import _split_thinking


def test_split_thinking_extracts_single_block():
    content, reasoning = _split_thinking(
        "<think>I should answer carefully.</think>\nThe capital is Paris."
    )
    assert content == "The capital is Paris."
    assert reasoning == "I should answer carefully."


def test_split_thinking_extracts_multiline_block():
    raw = "<think>\nLine 1.\nLine 2.\n</think>\n\nFinal answer."
    content, reasoning = _split_thinking(raw)
    assert content == "Final answer."
    assert "Line 1." in reasoning and "Line 2." in reasoning


def test_split_thinking_concatenates_multiple_blocks():
    raw = "<think>A</think>Some content.<think>B</think>More content."
    content, reasoning = _split_thinking(raw)
    # Both think blocks gone from content.
    assert "<think>" not in content
    assert "Some content.More content." == content
    # Both captured in reasoning.
    assert reasoning is not None and "A" in reasoning and "B" in reasoning


def test_split_thinking_returns_none_when_no_blocks():
    content, reasoning = _split_thinking("Plain response.")
    assert content == "Plain response."
    assert reasoning is None


def test_split_thinking_handles_case_insensitive_and_whitespace():
    raw = "<Think >  hidden  </Think >\nVisible."
    content, reasoning = _split_thinking(raw)
    assert content == "Visible."
    assert reasoning is not None and "hidden" in reasoning


def test_split_thinking_does_not_leak_into_content():
    """The whole point of the adapter — content must never contain <think>."""
    raw = "<think>secret reasoning</think>final"
    content, _reasoning = _split_thinking(raw)
    assert "<think>" not in content
    assert "secret" not in content
