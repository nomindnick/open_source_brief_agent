"""Loop tests using a scripted mock model.

The live smoke test (``--mission test``) covers the happy path. These tests
cover behaviors that are hard to trigger deterministically with a real
model: parse-error feedback, unknown-tool feedback, exception-in-tool
containment, and iteration-cap exhaustion.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent.loop import AgentResult, NoOpTrace, run_agent
from agent.model.base import ModelResponse
from agent.tools.base import Tool, ToolRegistry


@dataclass
class ScriptedModel:
    """A ``ModelInterface`` that replays a fixed list of response strings."""

    responses: list[str]
    calls_seen: list[list[dict[str, str]]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.calls_seen = []
        self._idx = 0

    def complete(self, messages, **kwargs):  # noqa: ARG002
        self.calls_seen.append(list(messages))
        if self._idx >= len(self.responses):
            raise AssertionError(
                f"ScriptedModel exhausted: loop asked for turn {self._idx + 1} "
                f"but only {len(self.responses)} responses were scripted."
            )
        content = self.responses[self._idx]
        self._idx += 1
        return ModelResponse(content=content, reasoning=None, raw=content, usage={})


class _CapturingEcho(Tool):
    name = "echo"
    description = "Returns its message."
    input_schema = {"type": "object", "properties": {"message": {"type": "string"}}}

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run(self, input):  # noqa: A002
        self.calls.append(input)
        return input.get("message", "")


class _BoomTool(Tool):
    name = "boom"
    description = "Always raises."
    input_schema = {"type": "object"}

    def run(self, input):  # noqa: A002, ARG002
        raise RuntimeError("kaboom")


def test_happy_path_one_tool_then_final_answer():
    model = ScriptedModel(
        responses=[
            '<tool_use><name>echo</name><input>{"message": "hi"}</input></tool_use>',
            "<final_answer>I said hi.</final_answer>",
        ]
    )
    tool = _CapturingEcho()
    registry = ToolRegistry([tool])

    result = run_agent(model, registry, "sys", "do it", NoOpTrace(), max_iter=5)

    assert result == AgentResult(final_answer="I said hi.", iterations=2, hit_cap=False)
    assert tool.calls == [{"message": "hi"}]


def test_parse_error_recovery_feeds_reason_to_model():
    model = ScriptedModel(
        responses=[
            "<tool_use><name>echo</name><input>not json</input></tool_use>",
            "<final_answer>Recovered.</final_answer>",
        ]
    )
    registry = ToolRegistry([_CapturingEcho()])

    result = run_agent(model, registry, "sys", "go", NoOpTrace(), max_iter=5)

    assert result.final_answer == "Recovered."
    assert result.iterations == 2
    # Turn 2 should have seen the parse-error observation injected.
    last_messages = model.calls_seen[1]
    assert any("could not be parsed" in m["content"] for m in last_messages)


def test_unknown_tool_is_observation_not_crash():
    model = ScriptedModel(
        responses=[
            '<tool_use><name>nope</name><input>{}</input></tool_use>',
            "<final_answer>Fine, I'll skip it.</final_answer>",
        ]
    )
    registry = ToolRegistry([_CapturingEcho()])

    result = run_agent(model, registry, "sys", "go", NoOpTrace(), max_iter=5)

    assert result.final_answer == "Fine, I'll skip it."
    # The unknown-tool observation should mention the unknown name and the
    # list of what *is* available.
    last_messages = model.calls_seen[1]
    obs = next(m["content"] for m in last_messages if "not available" in m["content"])
    assert "nope" in obs
    assert "echo" in obs


def test_tool_exception_is_contained():
    """Per Tool ABC contract, tools shouldn't raise — but if they do, contain it."""
    model = ScriptedModel(
        responses=[
            '<tool_use><name>boom</name><input>{}</input></tool_use>',
            "<final_answer>Survived.</final_answer>",
        ]
    )
    registry = ToolRegistry([_BoomTool()])

    result = run_agent(model, registry, "sys", "go", NoOpTrace(), max_iter=5)
    assert result.final_answer == "Survived."
    # The observation is the user-role message containing 'ERROR' (the
    # earlier assistant message contains the literal tool name 'boom' too,
    # so we filter for the actual error report).
    obs = next(
        m["content"]
        for m in model.calls_seen[1]
        if m["role"] == "user" and "ERROR" in m["content"]
    )
    assert "RuntimeError" in obs and "kaboom" in obs


def test_iteration_cap_returns_hit_cap_true():
    # Model never emits a final answer; loop should bail at max_iter.
    model = ScriptedModel(
        responses=['<tool_use><name>echo</name><input>{"message": "x"}</input></tool_use>'] * 3
    )
    registry = ToolRegistry([_CapturingEcho()])

    result = run_agent(model, registry, "sys", "go", NoOpTrace(), max_iter=2)
    assert result.final_answer is None
    assert result.iterations == 2
    assert result.hit_cap is True


def test_final_answer_short_circuits_remaining_tool_calls_in_same_turn():
    """Per parser contract: final_answer wins over co-occurring tool_use."""
    model = ScriptedModel(
        responses=[
            '<tool_use><name>echo</name><input>{"message":"x"}</input></tool_use>'
            "<final_answer>Already done.</final_answer>",
        ]
    )
    tool = _CapturingEcho()
    registry = ToolRegistry([tool])

    result = run_agent(model, registry, "sys", "go", NoOpTrace(), max_iter=5)
    assert result.final_answer == "Already done."
    # Tool was NOT called — final answer took precedence at parse time.
    assert tool.calls == []
