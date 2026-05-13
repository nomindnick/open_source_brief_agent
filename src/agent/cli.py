"""CLI entrypoint.

Parses arguments and dispatches to a mission. In Sprint 1.1 this only prints
a usage message; real mission dispatch arrives in Sprint 2.2 (test mission)
and Sprint 3.1 (paper_survey mission).
"""

from __future__ import annotations

import argparse
import sys


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
        help="Mission name (matches a system prompt in prompts/). Default: paper_survey.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model profile name from config.toml. Defaults to config's default_model.",
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
    # Sprint 1.1: no mission dispatch yet. Just acknowledge what would run.
    print(
        f"agent: would run mission={args.mission!r} "
        f"model={args.model or '<default>'} dry_run={args.dry_run}",
        file=sys.stderr,
    )
    print("(no mission dispatch implemented yet — Sprint 2.2 wires this up.)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
