# Overnight Research Agent

Local-first AI agent that surveys open-source / SLM research overnight, filters by personal interests, and writes a morning brief to an Obsidian vault. Built from scratch (no agent frameworks) as a learning project.

See [planning_docs/SPEC.md](planning_docs/SPEC.md) for the full design, and [planning_docs/IMPLEMENTATION_PLAN.md](planning_docs/IMPLEMENTATION_PLAN.md) for the sprint plan. [CLAUDE.md](CLAUDE.md) lists the hard constraints (no frameworks, custom XML tool calling, file-only storage).

## Install

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```sh
git clone <repo> open_source_brief_agent
cd open_source_brief_agent
uv sync
```

This project is pre-MVP. The HuggingFace `hf` CLI, llama.cpp, and Ollama setup steps will be documented in Sprint 5.1; for now, see [CLAUDE.md](CLAUDE.md) for the dev-machine setup.

## Configure

```sh
cp config.toml.example config.toml
cp .env.example .env
```

Edit `config.toml`:
- Set `vault_path` to your Obsidian vault root (briefs land in `<vault_path>/Briefs/`).
- Pick a model profile under `[models.*]` and set `default_model` to its name, or pass `--model <profile>` per run.

Edit `memory/Interests.md` with what you want the agent to surface and what to skip.

## Run

```sh
uv run python -m agent --help
```

Pre-MVP: `--help` works; mission dispatch is wired up in Sprint 2.2. Full usage will be documented at MVP (Sprint 5.1).

## Development

```sh
uv run pytest          # tests (added from Sprint 2.1 onward)
```

Traces land in `traces/YYYY-MM-DD/<mission>/`. They are git-ignored once they grow; keep them around for diffing prompt changes.
