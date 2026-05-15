"""End-of-run reflection — one non-agentic LLM call.

Runs after the brief is written. Reads today's keepers + summaries +
``Interests.md``, and produces a short markdown reflection on themes,
what felt high vs low signal, and what to weigh differently tomorrow.

Output is written to ``memory/Reflections/<date>.md`` and injected into
the *next* day's filter prompt as ``{{recent_reflection}}``.

Same pattern as filter and summarize — non-agentic, structured input,
file output, no agent loop involvement.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.brief.writer import render_brief
from agent.filter import Keeper
from agent.model.base import ModelInterface, ModelResponse
from agent.prompts import load_prompt
from agent.summarize import PaperSummary


@dataclass(frozen=True)
class ReflectionResult:
    content: str
    response: ModelResponse


def reflect_on_brief(
    model: ModelInterface,
    *,
    date: str,
    summaries: list[PaperSummary],
    keepers: list[Keeper],
    model_profile: str,
    papers_total: int,
    interests: str,
) -> ReflectionResult:
    """Produce a 3–6 sentence reflection on today's brief.

    The reflection sees the *same* rendered brief markdown the user
    will read — passing structured data to the model would let it
    react to fields the user can't see (or vice versa). Rendering
    is shared with :func:`agent.brief.writer.render_brief`.
    """
    brief_md = render_brief(
        date=date,
        summaries=summaries,
        keepers=keepers,
        model_profile=model_profile,
        papers_total=papers_total,
    )
    prompt = load_prompt(
        "system_reflection",
        interests=interests,
        brief=brief_md,
    )
    response = model.complete([{"role": "user", "content": prompt}])
    return ReflectionResult(content=response.content.strip(), response=response)
