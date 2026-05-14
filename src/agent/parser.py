"""XML tool-call parser.

Turns a model response string into one of three structured outcomes:

  - :class:`ToolCalls` — one or more well-formed ``<tool_use>`` blocks.
  - :class:`FinalAnswer` — a ``<final_answer>`` block, **or** (per the
    system-prompt convention) a response containing no XML at all.
  - :class:`ParseError` — something is structurally wrong. The
    :attr:`ParseError.reason` is short enough to feed back to the
    model as an observation so it can retry on the next turn.

The parser is **lenient about formatting** (whitespace, surrounding code
fences, trailing commas in JSON) and **strict about structure**
(``<name>`` is required, ``<input>`` must parse as a JSON object).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from agent.tools.base import FinalAnswer, ParseError, ParseResult, ToolCall


@dataclass(frozen=True)
class ToolCalls:
    """One or more parsed ``<tool_use>`` blocks from a single model turn."""

    calls: list[ToolCall]


# Re-export the result-set so callers can ``from agent.parser import ParseResult``.
ParseOutcome = ToolCalls | FinalAnswer | ParseError

# Non-greedy match for paired tags spanning multiple lines.
_TOOL_USE_RE = re.compile(r"<tool_use\b[^>]*>(.*?)</tool_use\s*>", re.DOTALL | re.IGNORECASE)
_FINAL_ANSWER_RE = re.compile(
    r"<final_answer\b[^>]*>(.*?)</final_answer\s*>", re.DOTALL | re.IGNORECASE
)
_NAME_RE = re.compile(r"<name\b[^>]*>(.*?)</name\s*>", re.DOTALL | re.IGNORECASE)
_INPUT_RE = re.compile(r"<input\b[^>]*>(.*?)</input\s*>", re.DOTALL | re.IGNORECASE)

# Detect a dangling opener (a `<tool_use>` with no matching close anywhere
# later in the string). Used only to surface a clearer error than "no tool
# call found" when the model truncated.
_TOOL_USE_OPEN_RE = re.compile(r"<tool_use\b[^>]*>", re.IGNORECASE)
_TOOL_USE_CLOSE_RE = re.compile(r"</tool_use\s*>", re.IGNORECASE)

# Strip a single leading/trailing markdown fence pair around the whole
# response. The model sometimes wraps everything in ```xml ... ```.
_CODE_FENCE_RE = re.compile(
    r"^\s*```[a-zA-Z0-9_-]*\s*\n(.*)\n```\s*$",
    re.DOTALL,
)

# Lenient JSON cleanup: strip trailing commas before } or ].
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _strip_code_fence(text: str) -> str:
    m = _CODE_FENCE_RE.match(text)
    return m.group(1) if m else text


def _lenient_json_loads(raw: str) -> Any:
    """Try strict JSON; on failure, retry after light cleanup.

    The cleanup pass is deliberately conservative — only fixing things
    that are unambiguous mistakes (trailing commas). Aggressive cleanup
    (e.g. single-quote→double-quote substitution) breaks legitimate
    payloads whose string values contain apostrophes.
    """
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = _TRAILING_COMMA_RE.sub(r"\1", raw)
        # Surface the post-cleanup error so the message reflects what
        # the model would need to fix.
        return json.loads(cleaned)


def _parse_one_tool_use(block_body: str, index: int) -> ToolCall | ParseError:
    """Parse a single ``<tool_use>`` block's inner content."""
    name_match = _NAME_RE.search(block_body)
    if not name_match:
        return ParseError(
            reason=f"<tool_use> block #{index + 1} is missing a <name>...</name> element."
        )
    name = name_match.group(1).strip()
    if not name:
        return ParseError(
            reason=f"<tool_use> block #{index + 1} has an empty <name>."
        )

    input_match = _INPUT_RE.search(block_body)
    if not input_match:
        return ParseError(
            reason=(
                f"<tool_use> block #{index + 1} (name={name!r}) is missing an "
                "<input>...</input> element."
            )
        )
    input_raw = input_match.group(1)

    try:
        parsed = _lenient_json_loads(input_raw)
    except json.JSONDecodeError as e:
        return ParseError(
            reason=(
                f"<tool_use> block #{index + 1} (name={name!r}): <input> is not "
                f"valid JSON: {e.msg} at line {e.lineno} col {e.colno}."
            )
        )

    if not isinstance(parsed, dict):
        return ParseError(
            reason=(
                f"<tool_use> block #{index + 1} (name={name!r}): <input> must be "
                f"a JSON object (got {type(parsed).__name__})."
            )
        )

    return ToolCall(name=name, input=parsed)


def parse(model_output: str) -> ParseOutcome:
    """Parse a model response into a structured outcome.

    Args:
        model_output: The raw ``content`` field from a ``ModelResponse``.
            Thinking has already been stripped at the adapter layer.

    Returns:
        Exactly one of :class:`ToolCalls`, :class:`FinalAnswer`, or
        :class:`ParseError`. Never raises.
    """
    text = _strip_code_fence(model_output.strip())

    # Final answer wins. If the model emits both <final_answer> and
    # <tool_use>, we honor the final answer and ignore the tool calls —
    # this matches what we tell the model in _tool_calling_format.md.
    fa_match = _FINAL_ANSWER_RE.search(text)
    if fa_match:
        body = fa_match.group(1).strip()
        if not body:
            return ParseError(reason="<final_answer> block is empty.")
        return FinalAnswer(text=body)

    # Collect all well-formed tool_use blocks.
    tool_use_bodies = _TOOL_USE_RE.findall(text)

    # If we found no closed blocks but there *is* a dangling opener,
    # surface that as a more useful error than "no XML found".
    if not tool_use_bodies:
        opens = len(_TOOL_USE_OPEN_RE.findall(text))
        closes = len(_TOOL_USE_CLOSE_RE.findall(text))
        if opens > closes:
            return ParseError(
                reason=(
                    "<tool_use> block has no matching </tool_use> closing tag. "
                    "Make sure every <tool_use> opens and closes."
                )
            )
        # No XML at all — treat the whole response as a final answer.
        # This is the documented convention; see _tool_calling_format.md.
        return FinalAnswer(text=text)

    calls: list[ToolCall] = []
    for i, body in enumerate(tool_use_bodies):
        result = _parse_one_tool_use(body, i)
        if isinstance(result, ParseError):
            return result
        calls.append(result)

    return ToolCalls(calls=calls)


__all__ = ["parse", "ToolCalls", "FinalAnswer", "ParseError", "ToolCall", "ParseOutcome"]
