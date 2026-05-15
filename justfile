# Convenience targets. Install `just` from https://github.com/casey/just
# (e.g. `cargo install just`, `brew install just`, or `apt install just`).
# Without `just`, the commands below are runnable directly by copying them.

# Default target: print the available recipes.
default:
    @just --list

# Run today's paper_survey end-to-end, writing brief + memory.
run:
    uv run python -m agent --mission paper_survey

# Run paper_survey but render the brief to stdout instead of the vault.
# Memory writes are also skipped. Useful for iterating on prompts.
dry-run:
    uv run python -m agent --mission paper_survey --dry-run

# Smoke test the agent loop with the EchoTool. No HF / network calls.
smoke:
    uv run python -m agent --mission test

# Replay paper_survey for a specific date (also used to simulate "tomorrow").
replay date:
    uv run python -m agent --mission paper_survey --date {{date}}

# Run a paper_survey with a specific model profile (overrides every stage).
run-with-model model:
    uv run python -m agent --mission paper_survey --model {{model}}

# Run the test suite.
test:
    uv run pytest -q

# Sync dependencies (after editing pyproject.toml).
sync:
    uv sync

# Smoke test a model profile (no agent — just verifies the backend is reachable).
smoke-model model:
    uv run python scripts/smoke_model.py --model {{model}}

# Benchmark a model profile against an XML-tool-call-shaped prompt.
benchmark-model model:
    uv run python scripts/benchmark_model.py --model {{model}}
