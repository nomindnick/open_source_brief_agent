"""Interest-based paper filter.

A single non-agentic LLM call that reads ``memory/Interests.md`` and the
full paper list (titles + abstracts), and returns a small JSON array of
**keepers** with one-line justifications.

This module does *not* use the agent loop. It's a function:

    keepers = filter_papers(model, list_markdown, interests)

The filter intentionally bounds input size at the entry of the pipeline
so Sprint 3.3's per-paper read+summarize stage has a predictable budget.

A keeper's ``reason`` field becomes user-facing in Sprint 4.1's brief
("How I picked these"), so the prompt coaches the model to write reasons
in the user's voice.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from agent.model.base import ModelInterface, ModelResponse
from agent.prompts import load_prompt


@dataclass(frozen=True)
class Keeper:
    """One paper the filter chose to surface in today's brief.

    Attributes:
        id: arxiv-style paper id from the daily list.
        reason: One short sentence on why this paper was picked. Written
            in the user's voice; reused verbatim in Sprint 4.1's brief.
    """

    id: str
    reason: str


@dataclass(frozen=True)
class FilterResult:
    """The output of one filter call.

    Attributes:
        keepers: The papers the filter chose to keep.
        response: The raw :class:`ModelResponse`. Trace writers log
            ``content`` and ``reasoning`` from this.
    """

    keepers: list[Keeper]
    response: ModelResponse


class FilterError(RuntimeError):
    """The filter call did not return a parseable JSON array of keepers."""


# Strip a leading/trailing ``` or ```json fence. Models love wrapping output.
_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$",
    re.DOTALL,
)

# Find the first top-level JSON array in the response — tolerates a leading
# "Here are my picks:" prose preamble that some models add despite instructions.
_FIRST_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)

# Trailing-comma cleanup, same as the XML parser uses.
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def parse_filter_response(content: str) -> list[Keeper]:
    """Lenient extraction of a JSON keeper array from a model response.

    Args:
        content: The model's ``content`` field (thinking already stripped
            by the adapter).

    Returns:
        A list of :class:`Keeper`. Empty list is a valid result (boring
        day, no papers matched interests).

    Raises:
        FilterError: When no array can be extracted, or entries are
            missing required fields. The message names the specific
            failure so it shows up clearly in the trace.
    """
    text = content.strip()

    # Strip a code fence around the whole response if present.
    m = _CODE_FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()

    # Find the first array literal. Without this, a prose preamble
    # ("Here are the keepers:") would fail strict json.loads.
    arr_match = _FIRST_ARRAY_RE.search(text)
    if not arr_match:
        raise FilterError(
            "Filter response did not contain a JSON array. "
            f"Got (first 200 chars): {text[:200]!r}"
        )
    raw_array = arr_match.group(0)
    raw_array = _TRAILING_COMMA_RE.sub(r"\1", raw_array)

    try:
        data: Any = json.loads(raw_array)
    except json.JSONDecodeError as e:
        raise FilterError(
            f"Filter response is not valid JSON: {e.msg} at line {e.lineno} col {e.colno}. "
            f"Snippet: {raw_array[:300]!r}"
        ) from e

    if not isinstance(data, list):
        raise FilterError(
            f"Filter response must be a JSON array; got {type(data).__name__}."
        )

    keepers: list[Keeper] = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise FilterError(
                f"Filter entry #{i + 1} is not an object: {entry!r}"
            )
        pid = entry.get("id")
        reason = entry.get("reason")
        if not isinstance(pid, str) or not pid.strip():
            raise FilterError(
                f"Filter entry #{i + 1} is missing a non-empty 'id'. Got: {entry!r}"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise FilterError(
                f"Filter entry #{i + 1} (id={pid!r}) is missing a non-empty 'reason'."
            )
        keepers.append(Keeper(id=pid.strip(), reason=reason.strip()))

    return keepers


def filter_papers(
    model: ModelInterface,
    list_markdown: str,
    interests: str,
) -> FilterResult:
    """Run the filter stage: pick keepers from the day's paper list.

    Args:
        model: A configured :class:`ModelInterface`. Can be a smaller/cheaper
            profile than the orchestrator — the filter is bounded I/O.
        list_markdown: Output of ``HfPapersListTool``.
        interests: Contents of ``memory/Interests.md``.

    Returns:
        :class:`FilterResult` carrying the keepers and the raw model response
        (so the caller can log content + reasoning to the trace).

    Raises:
        FilterError: When the filter's output can't be parsed. Filter is
            one-shot — there's no retry loop. If this is hit in production
            we tighten the prompt, not add a retry.
    """
    user_prompt = load_prompt(
        "system_filter",
        interests=interests,
        papers=list_markdown,
    )
    response = model.complete([{"role": "user", "content": user_prompt}])
    keepers = parse_filter_response(response.content)
    return FilterResult(keepers=keepers, response=response)
