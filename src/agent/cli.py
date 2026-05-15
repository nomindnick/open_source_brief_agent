"""CLI entrypoint.

Parses arguments and dispatches to a mission. Each mission supplies its
own ``run`` callable, so agent-loop missions (``test``) and deterministic
pipeline missions (``paper_survey``) live behind the same interface.

Exit codes:
    0   success (final answer produced)
    1   run completed but hit iteration cap without a final answer
    2   bad config / unknown mission / unrecoverable setup error
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import date as date_cls
from pathlib import Path
from typing import Callable

from agent.config import Config, load_config
from agent.filter import FilterError, filter_papers
from agent.loop import AgentResult, TraceSink, run_agent
from agent.memory.io import read_interests
from agent.model import get_model
from agent.prompts import load_prompt
from agent.summarize import PaperSummary, summarize_keepers
from agent.tools.base import ToolRegistry
from agent.tools.echo import EchoTool
from agent.tools.hf_papers import HfPapersListTool, HfPapersReadTool
from agent.trace.writer import TraceWriter

# ── mission runners ────────────────────────────────────────────────────


def _close_quietly(obj: object) -> None:
    close = getattr(obj, "close", None)
    if callable(close):
        close()


def _run_test_mission(config: Config, args: argparse.Namespace, trace: TraceSink) -> AgentResult:
    """Smoke mission — exercises the agent loop with EchoTool."""
    registry = ToolRegistry([EchoTool()])
    system_prompt = load_prompt(
        "system_test",
        tools=registry.render_for_prompt(),
    )
    user_task = (
        "Echo the phrase 'hello' first, then echo the phrase 'world'. "
        "Each one is a separate tool call. Then emit a <final_answer> "
        "that quotes both phrases."
    )
    model = get_model(config, args.model)
    max_iter = args.max_iter if args.max_iter is not None else config.iteration_cap
    try:
        return run_agent(
            model=model,
            registry=registry,
            system_prompt=system_prompt,
            user_task=user_task,
            trace=trace,
            max_iter=max_iter,
        )
    finally:
        _close_quietly(model)


_PAPER_HEADING_RE = re.compile(r"^## ", re.MULTILINE)


def _count_papers_in_listing(list_markdown: str) -> int:
    """Count ``## ID (upvotes)`` sections in the list tool's output."""
    return len(_PAPER_HEADING_RE.findall(list_markdown))


def _run_paper_survey(config: Config, args: argparse.Namespace, trace: TraceSink) -> AgentResult:
    """Daily-papers mission — deterministic pipeline, not the agent loop.

    Sprint 3.2 implements list → filter; Sprint 3.3 adds per-paper
    read+summarize; Sprint 4.1 writes the brief to Obsidian.
    """
    # ── Stage 1: list (deterministic subprocess) ──
    papers_md = HfPapersListTool().run({})
    if papers_md.startswith("ERROR"):
        # Surface the CLI's error as a "final answer" so the user sees it
        # in the trace + stdout, but mark as no-real-answer via empty result.
        print(papers_md, file=sys.stderr)
        return AgentResult(final_answer=None, iterations=0, hit_cap=False)

    papers_count = _count_papers_in_listing(papers_md)

    # ── Stage 2: filter (non-agentic LLM call) ──
    interests = read_interests()
    trace.log_filter_input(papers_count=papers_count, interests_chars=len(interests))

    model = get_model(config, args.model)
    try:
        try:
            result = filter_papers(model, papers_md, interests)
        except FilterError as e:
            print(f"agent: filter failed: {e}", file=sys.stderr)
            trace.log_filter_response("", None)
            trace.log_filter_keepers([])
            return AgentResult(final_answer=None, iterations=0, hit_cap=False)
    finally:
        _close_quietly(model)

    trace.log_filter_response(result.response.content, result.response.reasoning)
    trace.log_filter_keepers(result.keepers)

    # ── Stage 3: per-paper read + summary (Sprint 3.3) ──
    # The same model handles summaries for now. summarize_keepers
    # opens its own model session not needed — we reuse the existing
    # model instance via a freshly built one for clean separation.
    if not result.keepers:
        body = "Nothing on today's list met the bar.\n"
        return AgentResult(final_answer=body, iterations=1, hit_cap=False)

    summary_model = get_model(config, args.model)
    try:
        summaries = summarize_keepers(summary_model, result.keepers, trace=trace)
    finally:
        _close_quietly(summary_model)

    # ── Provisional output (Sprint 4.1 replaces this with Obsidian write) ──
    body = _format_summaries_for_stdout(result.keepers, summaries)
    return AgentResult(final_answer=body, iterations=1, hit_cap=False)


def _format_summaries_for_stdout(
    keepers: list,
    summaries: list[PaperSummary],
) -> str:
    """Render summaries as markdown for the CLI's stdout output.

    Sprint 4.1's brief writer will produce a richer version of this for
    the Obsidian vault. This function exists so 3.3 has a useful end-to-
    end output without depending on 4.1.
    """
    lines: list[str] = []
    lines.append(f"# Today's brief — {len(summaries)}/{len(keepers)} papers summarized\n")

    if not summaries:
        lines.append("(No papers survived the summarize stage — see trace for skips.)\n")
        return "\n".join(lines)

    for s in summaries:
        lines.append(f"## {s.title}")
        lines.append(f"_{s.id}_ — [link]({s.link})\n")
        lines.append(f"**TL;DR:** {s.tldr}\n")
        lines.append(f"**Why it matters:** {s.why_it_matters}\n")
        if s.quote:
            lines.append(f"> {s.quote}\n")

    if len(summaries) < len(keepers):
        lines.append(
            f"\n_({len(keepers) - len(summaries)} keeper(s) skipped — see trace.)_"
        )
    return "\n".join(lines) + "\n"


# ── Mission registry ───────────────────────────────────────────────────


@dataclass(frozen=True)
class Mission:
    """A named runnable that produces an :class:`AgentResult`.

    Each mission encapsulates its own dispatch — agent-loop vs.
    deterministic pipeline. The CLI just calls ``mission.run(...)``.
    """

    name: str
    run: Callable[[Config, argparse.Namespace, TraceSink], AgentResult]


MISSIONS: dict[str, Mission] = {
    "test": Mission(name="test", run=_run_test_mission),
    "paper_survey": Mission(name="paper_survey", run=_run_paper_survey),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent",
        description=(
            "Overnight research agent. Runs a mission to completion: surveys "
            "open-source / SLM research, filters by personal interests, and "
            "writes a brief to your Obsidian vault."
        ),
    )
    parser.add_argument(
        "--mission",
        default="paper_survey",
        help=f"Mission name. Known: {sorted(MISSIONS)}. Default: paper_survey.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model profile name from config.toml. Defaults to config's default_model.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Override iteration cap (only used by agent-loop missions like 'test').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run everything except writing to the Obsidian vault.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mission not in MISSIONS:
        print(
            f"error: unknown mission {args.mission!r}. Known: {sorted(MISSIONS)}",
            file=sys.stderr,
        )
        return 2

    try:
        config: Config = load_config()
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    mission = MISSIONS[args.mission]
    profile_name = args.model or config.default_model
    run_date = date_cls.today().isoformat()
    max_iter = args.max_iter if args.max_iter is not None else config.iteration_cap

    trace = TraceWriter(
        traces_root=Path("traces"),
        date=run_date,
        mission=mission.name,
        model_profile=profile_name,
        max_iter=max_iter,
    )

    try:
        result = mission.run(config, args, trace)
    finally:
        trace.close()

    print("=" * 60, file=sys.stderr)
    print(
        f"agent: mission={mission.name!r} iterations={result.iterations} "
        f"hit_cap={result.hit_cap}",
        file=sys.stderr,
    )
    print(f"agent: trace written to {trace.run_dir}/", file=sys.stderr)

    if result.final_answer is not None:
        print(result.final_answer)
        return 0

    if result.hit_cap:
        print(
            f"agent: hit iteration cap ({max_iter}) before producing a final answer",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
