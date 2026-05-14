"""Manual smoke test for a model profile.

Sends a hardcoded prompt to a profile and prints content + reasoning
separately so you can confirm:

  - the backend is up and reachable
  - thinking is being extracted (when supports_thinking=true)
  - the response shape is what the agent loop will see

Usage:

    uv run python scripts/smoke_model.py                       # uses default profile
    uv run python scripts/smoke_model.py --model qwen3-9b-ollama
    uv run python scripts/smoke_model.py --config path/to/config.toml
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from agent.config import load_config
from agent.model import get_model
from agent.model.base import ModelBackendError

SMOKE_PROMPT = "In one sentence: what is the capital of France?"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke test a model profile.")
    parser.add_argument(
        "--model",
        default=None,
        help="Profile name from config.toml. Defaults to config's default_model.",
    )
    parser.add_argument(
        "--config",
        default="config.toml",
        help="Path to config.toml.",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    profile_name = args.model or config.default_model
    profile = config.profile(profile_name)
    print(
        f"profile={profile_name!r} backend={profile.backend} "
        f"model={profile.model_name} thinking={profile.supports_thinking}",
        file=sys.stderr,
    )
    print(f"prompt: {SMOKE_PROMPT}", file=sys.stderr)
    print("-" * 60, file=sys.stderr)

    model = get_model(config, profile_name)
    try:
        try:
            response = model.complete(
                [{"role": "user", "content": SMOKE_PROMPT}],
            )
        except ModelBackendError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    finally:
        # Adapters expose close(); the Protocol doesn't declare it, but our
        # concrete adapters all support it. Call defensively.
        close = getattr(model, "close", None)
        if callable(close):
            close()

    if response.reasoning:
        print("=== reasoning ===")
        print(textwrap.indent(response.reasoning, "  "))
        print()
    print("=== content ===")
    print(response.content)
    print()
    print("=== usage ===", file=sys.stderr)
    print(response.usage, file=sys.stderr)

    # The whole point of the adapter: thinking must not leak into content.
    if response.reasoning and "<think>" in response.content.lower():
        print(
            "WARNING: <think> tag found in content — thinking extraction is leaking.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
