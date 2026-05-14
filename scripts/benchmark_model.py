"""Benchmark a model profile against an XML-tool-call-shaped prompt.

The workflow for "I downloaded a new model — does it handle our format?"

Reports total wall time, output token count (when the backend reports it),
and throughput. Time-to-first-token requires streaming; the v1 adapter
is non-streaming so TTFT is reported as N/A. Add streaming and revisit
if we want that metric.

The prompt below is a Sprint-1.2 stub. Sprint 2.2 lands a real test mission
which should replace ``BENCHMARK_PROMPT`` here.
"""

from __future__ import annotations

import argparse
import sys
import time

from agent.config import load_config
from agent.model import get_model
from agent.model.base import ModelBackendError

# Stress XML emission: the model has to produce a well-formed <tool_use>
# block. If it can't do this consistently, it won't work as our agent.
BENCHMARK_PROMPT = """\
You are a research assistant. To look up a paper, you must emit a tool call
in EXACTLY this format and nothing else (no prose before or after):

<tool_use>
  <name>read_paper</name>
  <input>{"id": "2401.12345"}</input>
</tool_use>

Now: emit one tool call for arxiv id "2509.05591"."""


def _fmt_throughput(tokens: int | None, elapsed_s: float) -> str:
    if not tokens:
        return "N/A (backend did not report eval_count)"
    return f"{tokens / elapsed_s:.1f} tok/s ({tokens} tokens in {elapsed_s:.2f}s)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark a model profile.")
    parser.add_argument(
        "--model",
        default=None,
        help="Profile name. Defaults to config's default_model.",
    )
    parser.add_argument("--config", default="config.toml")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    profile_name = args.model or config.default_model
    profile = config.profile(profile_name)
    model = get_model(config, profile_name)

    print(f"profile: {profile_name}")
    print(f"  backend: {profile.backend}")
    print(f"  model:   {profile.model_name}")
    print(f"  context: {profile.context_length}")
    print(f"  thinking: {profile.supports_thinking}")
    print()
    print("running benchmark prompt...", file=sys.stderr)

    t0 = time.perf_counter()
    try:
        response = model.complete([{"role": "user", "content": BENCHMARK_PROMPT}])
    except ModelBackendError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        close = getattr(model, "close", None)
        if callable(close):
            close()
    elapsed = time.perf_counter() - t0

    completion_tokens = response.usage.get("completion_tokens")
    prompt_tokens = response.usage.get("prompt_tokens")

    print("=== timing ===")
    print(f"  total wall time:   {elapsed:.2f}s")
    print(f"  time-to-first-tok: N/A (non-streaming)")
    print(f"  throughput:        {_fmt_throughput(completion_tokens, elapsed)}")
    print(f"  prompt tokens:     {prompt_tokens if prompt_tokens is not None else 'N/A'}")
    print()

    if response.reasoning:
        print("=== reasoning ===")
        print(response.reasoning)
        print()
    print("=== content ===")
    print(response.content)
    print()

    # Quick sanity: did the model emit the XML format we asked for?
    body = response.content.lower()
    well_formed = "<tool_use>" in body and "</tool_use>" in body and "<name>" in body
    print(
        "format check:",
        "OK" if well_formed else "FAILED — model did not emit a well-formed tool_use block",
    )
    return 0 if well_formed else 1


if __name__ == "__main__":
    raise SystemExit(main())
