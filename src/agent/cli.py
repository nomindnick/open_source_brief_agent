"""CLI entrypoint.

Parses arguments and dispatches to a mission. Sprint 2.2 wires the
``test`` mission end-to-end; later sprints add ``paper_survey``.

Exit codes:
    0   success (final answer produced)
    1   run completed but hit iteration cap without a final answer
    2   bad config / unknown mission / unrecoverable setup error
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date as date_cls
from pathlib import Path
from typing import Callable

from agent.config import Config, load_config
from agent.loop import AgentResult, run_agent
from agent.model import get_model
from agent.prompts import load_prompt
from agent.tools.base import ToolRegistry
from agent.tools.echo import EchoTool
from agent.tools.hf_papers import HfPapersListTool, HfPapersReadTool
from agent.trace.writer import TraceWriter


@dataclass(frozen=True)
class Mission:
    """A registered mission — name, system prompt name, tools, default task."""

    name: str
    system_prompt_name: str
    build_registry: Callable[[], ToolRegistry]
    user_task: str


def _build_test_registry() -> ToolRegistry:
    return ToolRegistry([EchoTool()])


def _build_paper_survey_registry() -> ToolRegistry:
    return ToolRegistry([HfPapersListTool(), HfPapersReadTool()])


MISSIONS: dict[str, Mission] = {
    "test": Mission(
        name="test",
        system_prompt_name="system_test",
        build_registry=_build_test_registry,
        user_task=(
            "Echo the phrase 'hello' first, then echo the phrase 'world'. "
            "Each one is a separate tool call. Then emit a <final_answer> "
            "that quotes both phrases."
        ),
    ),
    "paper_survey": Mission(
        name="paper_survey",
        system_prompt_name="system_paper_survey",
        build_registry=_build_paper_survey_registry,
        user_task=(
            "Survey today's HuggingFace Daily Papers and summarize what stood out."
        ),
    ),
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
        help="Override iteration cap. Default: config.iteration_cap.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run everything except writing to the Obsidian vault (no-op in 2.2).",
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
    registry = mission.build_registry()

    system_prompt = load_prompt(
        mission.system_prompt_name,
        tools=registry.render_for_prompt(),
    )

    model = get_model(config, args.model)
    max_iter = args.max_iter if args.max_iter is not None else config.iteration_cap
    profile_name = args.model or config.default_model
    run_date = date_cls.today().isoformat()

    trace = TraceWriter(
        traces_root=Path("traces"),
        date=run_date,
        mission=mission.name,
        model_profile=profile_name,
        max_iter=max_iter,
    )

    try:
        result: AgentResult = run_agent(
            model=model,
            registry=registry,
            system_prompt=system_prompt,
            user_task=mission.user_task,
            trace=trace,
            max_iter=max_iter,
        )
    finally:
        trace.close()
        close = getattr(model, "close", None)
        if callable(close):
            close()

    print("=" * 60, file=sys.stderr)
    print(
        f"agent: mission={mission.name!r} iterations={result.iterations} "
        f"hit_cap={result.hit_cap}",
        file=sys.stderr,
    )
    print(f"agent: trace written to {trace.run_dir}/", file=sys.stderr)

    if result.final_answer is not None:
        # Stdout is reserved for the final answer so the caller can pipe it.
        print(result.final_answer)
        return 0

    print(
        f"agent: hit iteration cap ({max_iter}) before producing a final answer",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
