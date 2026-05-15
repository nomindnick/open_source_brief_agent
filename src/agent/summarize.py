"""Per-paper summarization.

For each :class:`Keeper` the filter produced, fetch the full paper text
and run a single non-agentic summarization call to produce a structured
:class:`PaperSummary`. Output feeds Sprint 4.1's brief writer.

Failures degrade gracefully — per the SPEC, "a partial brief is better
than no brief." A bad read or unparseable summary skips that paper and
emits a trace event; the surrounding keepers proceed normally.

This module does *not* use the agent loop. It's a programmatic for-loop
of two function calls per keeper: read, then summarize.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from agent.filter import Keeper
from agent.loop import TraceSink
from agent.model.base import ModelInterface
from agent.prompts import load_prompt
from agent.tools.hf_papers import HfPapersReadTool

# Bound the paper text we pass to the summarizer. 24K chars ≈ 8K tokens;
# leaves room for system prompt + reasoning + structured output well
# inside our 64K context. First-N truncation is deliberately simple —
# smarter section extraction (abstract + intro + conclusion) is a future
# optimization once we have a real signal about quality.
DEFAULT_MAX_CHARS = 24_000


@dataclass(frozen=True)
class PaperSummary:
    """A structured summary of one paper, ready for the brief writer.

    Attributes:
        id: arxiv-style paper id.
        title: Paper title, as reported in the read output.
        tldr: 1–2 sentence punchline of what this paper is about.
        why_it_matters: ~2 sentences on why this is worth the user's
            attention — written post-read with knowledge of the full paper.
            Complements (and refines) the filter's pre-read justification.
        quote: One verbatim sentence from the paper that captures the
            strongest claim or detail. None if no clear quote-worthy line
            was found — better to omit than to paraphrase.
        link: arxiv URL.
        filter_reason: The keeper's original filter justification. Carried
            through so the brief writer can show pre-read vs. post-read.
    """

    id: str
    title: str
    tldr: str
    why_it_matters: str
    quote: str | None
    link: str
    filter_reason: str


class SummarizeError(RuntimeError):
    """A summary call did not return a parseable JSON object."""


# Same lenient-JSON pattern as agent.filter — strip code fence, find first
# {…} object, tolerate trailing commas before strict json.loads.
_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$",
    re.DOTALL,
)
_FIRST_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def parse_summary_response(content: str) -> dict[str, Any]:
    """Lenient extraction of a JSON object from a summary model response.

    Strategy: try strict parse first (handles the well-behaved case
    cleanly), then fall back to regex-extracting the first ``{...}``
    block (handles models that add a prose preamble). A response whose
    root is a JSON array always errors — that's a structural mistake,
    not a preamble.

    Returns the parsed dict on success; raises :class:`SummarizeError`
    with a descriptive message on structural failure. The caller is
    responsible for validating required fields.
    """
    text = content.strip()
    m = _CODE_FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()

    cleaned = _TRAILING_COMMA_RE.sub(r"\1", text)

    # Strict parse first. If the response is a well-formed JSON document,
    # this catches both the happy path (root object) and the "model
    # returned an array" mistake.
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = None

    if parsed is not None:
        if not isinstance(parsed, dict):
            raise SummarizeError(
                f"Summary response must be a JSON object; got {type(parsed).__name__}."
            )
        return parsed

    # Fall back to regex extraction for "Here's the JSON: {…}"-style preambles.
    obj_match = _FIRST_OBJECT_RE.search(cleaned)
    if not obj_match:
        raise SummarizeError(
            "Summary response did not contain a JSON object. "
            f"Got (first 200 chars): {text[:200]!r}"
        )
    raw_obj = obj_match.group(0)
    try:
        parsed = json.loads(raw_obj)
    except json.JSONDecodeError as e:
        raise SummarizeError(
            f"Summary response is not valid JSON: {e.msg} at line {e.lineno} col {e.colno}."
        ) from e
    if not isinstance(parsed, dict):
        raise SummarizeError(
            f"Summary response must be a JSON object; got {type(parsed).__name__}."
        )
    return parsed


def _require_non_empty_string(obj: dict[str, Any], field: str) -> str:
    val = obj.get(field)
    if not isinstance(val, str) or not val.strip():
        raise SummarizeError(
            f"Summary response is missing a non-empty {field!r} field."
        )
    return val.strip()


def _optional_string(obj: dict[str, Any], field: str) -> str | None:
    val = obj.get(field)
    if val is None:
        return None
    if not isinstance(val, str):
        return None
    val = val.strip()
    return val if val else None


def _arxiv_link(paper_id: str) -> str:
    return f"https://arxiv.org/abs/{paper_id}"


def summarize_keeper(
    model: ModelInterface,
    keeper: Keeper,
    paper_text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> PaperSummary:
    """Run one summary call for a single keeper.

    Args:
        model: A configured :class:`ModelInterface`. Often the same
            profile the filter used.
        keeper: The :class:`Keeper` from the filter stage.
        paper_text: Full paper text from :class:`HfPapersReadTool`.
            Will be truncated to ``max_chars`` before the prompt is built.
        max_chars: Truncation cap for ``paper_text``. Deliberately simple
            first-N-chars heuristic; section-aware extraction is a future
            improvement.

    Returns:
        A :class:`PaperSummary` ready for the brief writer.

    Raises:
        SummarizeError: When the model response can't be parsed as JSON
            with the required fields. The orchestrator catches this and
            skips the paper rather than aborting the whole run.
    """
    truncated = paper_text[:max_chars]
    prompt = load_prompt(
        "system_summarize",
        paper_text=truncated,
        filter_reason=keeper.reason,
        paper_id=keeper.id,
    )
    response = model.complete([{"role": "user", "content": prompt}])
    parsed = parse_summary_response(response.content)

    return PaperSummary(
        id=keeper.id,
        title=_require_non_empty_string(parsed, "title"),
        tldr=_require_non_empty_string(parsed, "tldr"),
        why_it_matters=_require_non_empty_string(parsed, "why_it_matters"),
        quote=_optional_string(parsed, "quote"),
        link=_arxiv_link(keeper.id),
        filter_reason=keeper.reason,
    )


def summarize_keepers(
    model: ModelInterface,
    keepers: list[Keeper],
    trace: TraceSink,
    max_chars: int = DEFAULT_MAX_CHARS,
    read_tool: HfPapersReadTool | None = None,
) -> list[PaperSummary]:
    """Read + summarize each keeper. Skip on failure.

    Per-paper failures (bad read, malformed summary, model exception)
    are logged to the trace via :meth:`TraceSink.log_summarize_skipped`
    and the loop continues. Returns the summaries that did succeed —
    possibly fewer than ``len(keepers)``.
    """
    read_tool = read_tool or HfPapersReadTool()
    summaries: list[PaperSummary] = []

    for keeper in keepers:
        paper_text = read_tool.run({"id": keeper.id})
        if paper_text.startswith("ERROR"):
            trace.log_summarize_skipped(keeper.id, paper_text)
            continue

        char_count = min(len(paper_text), max_chars)
        trace.log_summarize_input(keeper.id, char_count)

        try:
            summary = summarize_keeper(model, keeper, paper_text, max_chars=max_chars)
        except SummarizeError as e:
            trace.log_summarize_skipped(keeper.id, f"SummarizeError: {e}")
            continue
        except Exception as e:  # noqa: BLE001 — last-resort containment
            trace.log_summarize_skipped(
                keeper.id, f"{type(e).__name__}: {e}"
            )
            continue

        # Log the raw model response (so the trace shows reasoning) and
        # then the parsed result.
        # Note: we lose the response object after summarize_keeper returns,
        # so re-thread it through if/when the trace needs reasoning. For
        # now we log_summarize_result with the structured PaperSummary;
        # reasoning is captured by the model adapter and is in the JSONL
        # under the model_turn-style event we'd add if we wanted full
        # introspection. Sprint 3.3 keeps trace lean — Sprint 5.1 can
        # revisit if reasoning per summary turns out useful.
        trace.log_summarize_result(summary)
        summaries.append(summary)

    return summaries
