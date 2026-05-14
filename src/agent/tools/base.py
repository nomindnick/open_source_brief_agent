"""Tool plumbing — ABC, registry, call record, parse error.

A :class:`Tool` is a unit of action the agent can invoke. Implementations
register themselves into a :class:`ToolRegistry`; the registry renders the
"tools available" block of the system prompt and dispatches by name when
the parser extracts a :class:`ToolCall`.

:class:`ParseError` is raised by the parser (Sprint 2.1) when a model
response can't be turned into a structured action. The loop catches it,
feeds the error reason back to the model as an observation, and continues.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    """A parsed ``<tool_use>`` block.

    Attributes:
        name: The tool name from the ``<name>`` element.
        input: Parsed JSON object from the ``<input>`` element.
    """

    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class FinalAnswer:
    """A parsed ``<final_answer>`` block (or a no-XML fallback).

    Attributes:
        text: The body of the answer. Whitespace-trimmed.
    """

    text: str


@dataclass(frozen=True)
class ParseError:
    """The parser could not turn this model output into an action.

    Attributes:
        reason: A short, model-readable description of the problem.
            Sent back to the model as an observation on the next turn
            so it can fix the formatting and retry.
    """

    reason: str


# A parse result is one of three shapes. The agent loop matches on it
# to decide its next move. Defined here (next to ToolCall) so the parser
# module can stay focused on parsing.
ParseResult = ToolCall | FinalAnswer | ParseError


class Tool(ABC):
    """Abstract base class for agent-callable tools.

    Subclass and implement :meth:`run`. The class-level attributes
    ``name``, ``description``, and ``input_schema`` are rendered into
    the "tools available" block of the system prompt by
    :meth:`ToolRegistry.render_for_prompt`.
    """

    #: Stable identifier the model uses in ``<name>``.
    name: str = ""
    #: One-line description shown to the model in the system prompt.
    description: str = ""
    #: JSON-schema-ish dict describing the ``<input>`` body. Rendered
    #: verbatim into the system prompt; keep it small and readable.
    input_schema: dict[str, Any] = {}

    @abstractmethod
    def run(self, input: dict[str, Any]) -> str:
        """Execute the tool and return a string observation.

        The returned string becomes the next model turn's user message
        ("Observation: ..."). Tools should never raise out of ``run`` —
        wrap errors in a returned string so the model can recover.
        """
        raise NotImplementedError


class ToolRegistry:
    """Holds the tools available to a given mission.

    Built once at the start of a run; passed to the agent loop, which
    dispatches incoming :class:`ToolCall` instances by name.
    """

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or ():
            self.register(t)

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError(f"Tool {type(tool).__name__} has empty .name")
        if tool.name in self._tools:
            raise ValueError(f"Tool name {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def render_for_prompt(self) -> str:
        """Render a "tools available" block for injection into a system prompt.

        Output is plain markdown with one section per tool, listing the
        name, description, and a compact JSON dump of the input schema.
        """
        if not self._tools:
            return "(no tools available)"
        sections: list[str] = []
        for name in self.names():
            t = self._tools[name]
            schema_json = json.dumps(t.input_schema, indent=2)
            sections.append(
                f"### {t.name}\n\n{t.description}\n\n"
                f"Input schema:\n```json\n{schema_json}\n```"
            )
        return "\n\n".join(sections)
