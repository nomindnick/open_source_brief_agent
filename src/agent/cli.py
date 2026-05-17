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

from agent.brief.writer import check_vault_writable, render_brief, write_brief
from agent.config import Config, load_config
from agent.feedback import ingest_pending_feedback
from agent.filter import FilterError, exclude_seen_papers, filter_papers
from agent.loop import AgentResult, TraceSink, run_agent
from agent.memory.io import (
    append_seen_ids,
    read_interests,
    read_latest_reflection,
    read_recent_feedback,
    read_seen_ids,
    write_reflection,
)
from agent.model import get_model
from agent.prompts import load_prompt
from agent.reflect import reflect_on_brief
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

    Pipeline: list → filter → per-paper read+summarize → write brief to
    Obsidian vault (or stdout in ``--dry-run``).
    """
    # ── Pre-flight: vault must be writable before we spend 10 min on summaries ──
    if not args.dry_run:
        try:
            check_vault_writable(config.vault_path)
        except (FileNotFoundError, NotADirectoryError, PermissionError) as e:
            print(f"agent: {e}", file=sys.stderr)
            return AgentResult(final_answer=None, iterations=0, hit_cap=False)

    run_date = args.date  # set by main() from local date

    ps_config = config.paper_survey

    # ── Stage 0: ingest user feedback from prior briefs (Sprint 6.1) ──
    # Scans the vault for briefs whose date is strictly < today and
    # whose date isn't already a heading in Feedback.md; parses filled
    # entries from each brief's Feedback section and appends a dated
    # block to Feedback.md. Legacy briefs (no Feedback section) are
    # silently no-op'd. Skipped in --dry-run so iterating prompts
    # doesn't quietly mutate memory.
    if not args.dry_run:
        try:
            ingest_pending_feedback(
                vault_path=config.vault_path,
                run_date=run_date,
                trace=trace,
            )
        except Exception as e:  # noqa: BLE001 — ingest is best-effort
            print(
                f"agent: feedback ingest failed (non-fatal): "
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )

    # ── Stage 1: list (deterministic subprocess) ──
    list_tool = HfPapersListTool(
        timeout_s=ps_config.hf_subprocess_timeout_s,
        abstract_chars=ps_config.list_abstract_chars,
    )
    papers_md_full = list_tool.run({})
    if papers_md_full.startswith("ERROR"):
        print(papers_md_full, file=sys.stderr)
        return AgentResult(final_answer=None, iterations=0, hit_cap=False)

    # ── Stage 1b: Seen.md dedup (deterministic, pre-LLM) ──
    # Strictly less than today: same-day reruns DO see yesterday's IDs
    # (useful when iterating); cross-day they don't.
    seen_ids = read_seen_ids(before_date=run_date)
    papers_md, dropped = exclude_seen_papers(papers_md_full, seen_ids)
    trace.log_seen_filter(dropped=dropped, total_seen=len(seen_ids))
    papers_count = _count_papers_in_listing(papers_md)

    # ── Stage 2: filter (non-agentic LLM call) ──
    interests = read_interests()
    recent_reflection = read_latest_reflection(before_date=run_date) or ""
    recent_feedback = read_recent_feedback(
        window=ps_config.feedback_window, before_date=run_date
    )
    trace.log_filter_input(papers_count=papers_count, interests_chars=len(interests))
    trace.log_feedback_inject(
        window_size=ps_config.feedback_window, chars=len(recent_feedback)
    )

    filter_profile = config.stage_model("filter", cli_override=args.model)
    model = get_model(config, filter_profile)
    try:
        try:
            result = filter_papers(
                model,
                papers_md,
                interests,
                recent_reflection=recent_reflection,
                recent_feedback=recent_feedback,
            )
        except FilterError as e:
            print(f"agent: filter failed: {e}", file=sys.stderr)
            trace.log_filter_response("", None)
            trace.log_filter_keepers([])
            return AgentResult(final_answer=None, iterations=0, hit_cap=False)
    finally:
        _close_quietly(model)

    trace.log_filter_response(result.response.content, result.response.reasoning)
    trace.log_filter_keepers(result.keepers)

    # ── Stage 3: per-paper read + summary ──
    if not result.keepers:
        summaries: list[PaperSummary] = []
    else:
        summarize_profile = config.stage_model("summarize", cli_override=args.model)
        summary_model = get_model(config, summarize_profile)
        read_tool = HfPapersReadTool(timeout_s=ps_config.hf_subprocess_timeout_s)
        try:
            summaries = summarize_keepers(
                summary_model,
                result.keepers,
                trace=trace,
                max_chars=ps_config.summary_max_chars,
                read_tool=read_tool,
            )
        finally:
            _close_quietly(summary_model)

    # ── Stage 4: render brief; write to vault unless --dry-run ──
    # Brief frontmatter records the *default* profile for this run, since
    # summary/filter/reflection may differ; the trace has the full breakdown.
    profile_name = args.model or config.default_model
    if args.dry_run:
        body = render_brief(
            date=run_date,
            summaries=summaries,
            keepers=result.keepers,
            model_profile=profile_name,
            papers_total=papers_count,
        )
        print(f"agent: --dry-run — brief NOT written to vault", file=sys.stderr)
        print(
            f"agent: --dry-run — Seen.md, Reflections, and Feedback.md NOT updated",
            file=sys.stderr,
        )
        return AgentResult(final_answer=body, iterations=1, hit_cap=False)

    brief_path = write_brief(
        vault_path=config.vault_path,
        date=run_date,
        summaries=summaries,
        keepers=result.keepers,
        model_profile=profile_name,
        papers_total=papers_count,
    )
    print(f"agent: brief written to {brief_path}", file=sys.stderr)

    # ── Stage 5: memory write-back (Sprint 4.2) ──
    # Append SUMMARIZED ids only (per plan: "only ones that made the
    # brief"). Skipped keepers don't get marked as seen — we may try
    # them again tomorrow.
    summarized_ids = [s.id for s in summaries]
    if summarized_ids:
        append_seen_ids(summarized_ids, run_date)

    # Reflection: only run if we actually produced summaries; reflecting
    # on an empty brief is wasted compute.
    if summaries:
        brief_md = brief_path.read_text(encoding="utf-8")
        trace.log_reflection_input(
            brief_chars=len(brief_md), interests_chars=len(interests)
        )
        reflection_profile = config.stage_model("reflection", cli_override=args.model)
        reflect_model = get_model(config, reflection_profile)
        try:
            try:
                ref = reflect_on_brief(
                    reflect_model,
                    date=run_date,
                    summaries=summaries,
                    keepers=result.keepers,
                    model_profile=profile_name,
                    papers_total=papers_count,
                    interests=interests,
                )
                trace.log_reflection_output(
                    ref.content, ref.response.reasoning
                )
                refl_path = write_reflection(run_date, ref.content)
                print(f"agent: reflection written to {refl_path}", file=sys.stderr)
            except Exception as e:  # noqa: BLE001 — reflection is best-effort
                print(
                    f"agent: reflection failed (non-fatal): {type(e).__name__}: {e}",
                    file=sys.stderr,
                )
        finally:
            _close_quietly(reflect_model)

    # Stdout: a one-liner pointer to the file so callers can pipe.
    return AgentResult(final_answer=str(brief_path), iterations=1, hit_cap=False)


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
    parser.add_argument(
        "--date",
        default=None,
        help=(
            "Override the run date (YYYY-MM-DD). Useful for replaying a "
            "specific day's papers and for simulating successive-day runs "
            "in testing. Default: system local date."
        ),
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
    run_date = args.date or date_cls.today().isoformat()
    args.date = run_date  # passed through to mission runners
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
