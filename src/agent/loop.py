"""The think-act-observe agent loop.

A hand-written loop with no agent framework. Given a configured model,
a tool registry, and a system prompt, runs the model until a final
answer is produced or the iteration cap is hit.

Shape:

    1. Model produces a turn (content + reasoning).
    2. Parser turns content into ToolCalls | FinalAnswer | ParseError.
    3. On FinalAnswer: return.
    4. On ToolCalls: execute each tool in order, fold results into one
       observation user message, continue.
    5. On ParseError: feed the reason back as an observation, continue.
    6. On cap hit: return AgentResult(hit_cap=True).

The loop accepts a ``trace`` object — for Sprint 2.2 this is
:class:`StdoutTrace` (or :class:`NoOpTrace`); Sprint 2.3 swaps in the
real :class:`agent.trace.writer.TraceWriter`. The surface is shaped now
so that swap is a one-line change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from agent.model.base import ModelInterface, ModelResponse
from agent.parser import FinalAnswer, ParseError, ToolCalls, parse
from agent.tools.base import ToolCall, ToolRegistry


@dataclass
class AgentResult:
    """Outcome of a single ``run_agent`` invocation.

    Attributes:
        final_answer: The body of the model's ``<final_answer>`` block,
            or None if the run hit the iteration cap before answering.
        iterations: How many model turns the run consumed.
        hit_cap: True if the loop exited because it reached ``max_iter``.
    """

    final_answer: str | None
    iterations: int
    hit_cap: bool = False


class TraceSink(Protocol):
    """The minimal trace surface the loop and pipeline stages call into.

    Sprint 2.3's :class:`TraceWriter` implements this surface plus file I/O.
    Stubs (:class:`StdoutTrace`, :class:`NoOpTrace`) implement just the surface.

    Surface is grouped by concern: agent-loop events live above
    ``log_iteration_cap``; pipeline-stage events (filter, summarize) live
    below. Stages are independent — a mission may call only some.
    """

    # ── agent loop ─────────────────────────────────────────────────────
    def log_model_turn(self, content: str, reasoning: str | None) -> None: ...
    def log_tool_call(self, name: str, input: dict[str, Any]) -> None: ...
    def log_tool_result(self, name: str, result: str) -> None: ...
    def log_parse_error(self, reason: str) -> None: ...
    def log_final_answer(self, text: str) -> None: ...
    def log_iteration_cap(self, max_iter: int) -> None: ...

    # ── pipeline stages (Sprint 3.2+) ──────────────────────────────────
    def log_filter_input(self, papers_count: int, interests_chars: int) -> None: ...
    def log_filter_response(self, content: str, reasoning: str | None) -> None: ...
    def log_filter_keepers(self, keepers: list[Any]) -> None: ...


class NoOpTrace:
    """A trace sink that drops every event. For tests."""

    def log_model_turn(self, content: str, reasoning: str | None) -> None: ...
    def log_tool_call(self, name: str, input: dict[str, Any]) -> None: ...
    def log_tool_result(self, name: str, result: str) -> None: ...
    def log_parse_error(self, reason: str) -> None: ...
    def log_final_answer(self, text: str) -> None: ...
    def log_iteration_cap(self, max_iter: int) -> None: ...
    def log_filter_input(self, papers_count: int, interests_chars: int) -> None: ...
    def log_filter_response(self, content: str, reasoning: str | None) -> None: ...
    def log_filter_keepers(self, keepers: list[Any]) -> None: ...


class StdoutTrace:
    """Minimal sink that prints each event to stderr for interactive debugging.

    Lives in this module (and not in agent.trace) so Sprint 2.2 is
    self-contained. Sprint 2.3's real :class:`TraceWriter` supersedes it.
    """

    def __init__(self) -> None:
        self._turn = 0

    def log_model_turn(self, content: str, reasoning: str | None) -> None:
        self._turn += 1
        self._w(f"--- turn {self._turn} ---")
        if reasoning:
            self._w("[reasoning]")
            self._w(reasoning)
        self._w("[content]")
        self._w(content)

    def log_tool_call(self, name: str, input: dict[str, Any]) -> None:
        self._w(f"[tool call] {name}({input})")

    def log_tool_result(self, name: str, result: str) -> None:
        # Truncate huge tool results to keep stderr readable.
        snippet = result if len(result) <= 400 else result[:400] + f" ... ({len(result)} chars)"
        self._w(f"[tool result] {name} -> {snippet}")

    def log_parse_error(self, reason: str) -> None:
        self._w(f"[parse error] {reason}")

    def log_final_answer(self, text: str) -> None:
        self._w("[final answer]")
        self._w(text)

    def log_iteration_cap(self, max_iter: int) -> None:
        self._w(f"[iteration cap hit] {max_iter} iterations consumed without final answer")

    def log_filter_input(self, papers_count: int, interests_chars: int) -> None:
        self._w(
            f"--- filter stage --- "
            f"({papers_count} papers, {interests_chars} chars of interests)"
        )

    def log_filter_response(self, content: str, reasoning: str | None) -> None:
        if reasoning:
            self._w("[filter reasoning]")
            self._w(reasoning)
        self._w("[filter response]")
        self._w(content)

    def log_filter_keepers(self, keepers: list[Any]) -> None:
        self._w(f"[filter keepers] {len(keepers)} papers")
        for k in keepers:
            self._w(f"  - {getattr(k, 'id', '?')}: {getattr(k, 'reason', '?')}")

    @staticmethod
    def _w(s: str) -> None:
        import sys

        print(s, file=sys.stderr, flush=True)


def _format_observation(call: ToolCall, result: str) -> str:
    """Render one tool result for inclusion in the next user message."""
    return f"Result of tool {call.name!r}:\n{result}"


def _format_parse_error_observation(reason: str) -> str:
    """Render a parse error as a user observation the model can recover from."""
    return (
        f"Your previous response could not be parsed.\n"
        f"Reason: {reason}\n"
        f"Re-emit the request in the documented format and try again."
    )


def _format_unknown_tool_observation(call: ToolCall, registry: ToolRegistry) -> str:
    """Render an unknown-tool error so the model corrects on the next turn."""
    return (
        f"Tool {call.name!r} is not available. "
        f"Available tools: {registry.names()}."
    )


def run_agent(
    model: ModelInterface,
    registry: ToolRegistry,
    system_prompt: str,
    user_task: str,
    trace: TraceSink,
    max_iter: int = 25,
) -> AgentResult:
    """Run the agent loop until a final answer or the iteration cap.

    Args:
        model: A configured :class:`ModelInterface`.
        registry: Tools available to this run.
        system_prompt: Fully-rendered mission system prompt (with
            ``{{tools}}`` and ``{{tool_calling_format}}`` already
            substituted by :func:`agent.prompts.load_prompt`).
        user_task: The initial user message describing the task.
        trace: A :class:`TraceSink` — typically Sprint 2.3's
            ``TraceWriter`` or a stub.
        max_iter: Hard cap on model turns. Defaults to 25 (SPEC's safety net).

    Returns:
        :class:`AgentResult`. ``final_answer`` is None and ``hit_cap`` is
        True if the cap was reached without a ``<final_answer>``.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_task},
    ]

    for iteration in range(1, max_iter + 1):
        response: ModelResponse = model.complete(messages)
        trace.log_model_turn(content=response.content, reasoning=response.reasoning)

        outcome = parse(response.content)

        if isinstance(outcome, FinalAnswer):
            trace.log_final_answer(outcome.text)
            return AgentResult(final_answer=outcome.text, iterations=iteration)

        if isinstance(outcome, ParseError):
            trace.log_parse_error(outcome.reason)
            # Append the model's malformed turn so the model can see what
            # *it* said, then the corrective observation.
            messages.append({"role": "assistant", "content": response.content})
            messages.append(
                {"role": "user", "content": _format_parse_error_observation(outcome.reason)}
            )
            continue

        # ToolCalls: execute sequentially, fold results into one user message.
        assert isinstance(outcome, ToolCalls)
        messages.append({"role": "assistant", "content": response.content})

        observations: list[str] = []
        for call in outcome.calls:
            tool = registry.get(call.name)
            if tool is None:
                # Unknown-tool is *not* a ParseError — the parse succeeded.
                # Surface it as an observation so the model can correct.
                obs = _format_unknown_tool_observation(call, registry)
                observations.append(obs)
                continue
            trace.log_tool_call(call.name, call.input)
            try:
                result = tool.run(call.input)
            except Exception as e:
                # Per Tool ABC contract, tools shouldn't raise — but if
                # one does, contain it as an observation rather than
                # crashing the run.
                result = f"ERROR: tool {call.name!r} raised {type(e).__name__}: {e}"
            trace.log_tool_result(call.name, result)
            observations.append(_format_observation(call, result))

        messages.append({"role": "user", "content": "\n\n".join(observations)})

    trace.log_iteration_cap(max_iter)
    return AgentResult(final_answer=None, iterations=max_iter, hit_cap=True)


__all__ = [
    "run_agent",
    "AgentResult",
    "TraceSink",
    "NoOpTrace",
    "StdoutTrace",
]
