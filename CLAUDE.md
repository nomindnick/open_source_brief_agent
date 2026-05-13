# Overnight Research Agent

Local-first AI agent that surveys open-source / SLM research overnight, filters by personal interests, and writes a morning brief to an Obsidian vault. Built from scratch as a learning project for agent loops, custom tool calling, and overnight autonomy on local hardware.

## Read first

- **[planning_docs/SPEC.md](planning_docs/SPEC.md)** — goals, architecture, design decisions, scope
- **[planning_docs/IMPLEMENTATION_PLAN.md](planning_docs/IMPLEMENTATION_PLAN.md)** — phased sprint plan with tasks and acceptance criteria

The planning docs are the source of truth. This file is just the on-ramp.

## Hard constraints

- **No agent frameworks.** No LangChain, LangGraph, LlamaIndex, CrewAI, or similar. The agent loop, the parser, and the tool registry are hand-written. This is a deliberate learning constraint, not a preference.
- **Custom XML tool calling, not native function calling.** Models emit `<tool_use>` blocks; a lenient hand-written parser extracts them. Parse errors are fed back to the model as observations.
- **File-only storage.** Markdown for prompts/memory/briefs, JSONL for structured traces. No database, no vector store, no embeddings.
- **`uv` for environment and dependency management.** Not pip+venv.
- **Thin dependencies.** httpx, pydantic, pydantic-settings, python-dotenv, pytest. Anything else, ask first.

## Model layer principles

- **Multi-profile config.** Each `[models.<name>]` block in `config.toml` declares a swappable backend+model+params bundle. The CLI selects one per run.
- **Thinking is data, not noise.** Adapters separate reasoning from final output and return both via `ModelResponse(content, reasoning, raw, usage)`. The parser only sees `content`; the trace writer logs both.
- **Context length: default generous.** Per-profile `context_length` should be as large as the model + hardware afford. Constrain only when there's a specific reason (parallel agents, known long-context degradation).

## Dev environment

- **Hardware:** Framework Desktop, AMD Strix Halo (Radeon 8060S, gfx1151), 128GB unified memory, ~96GB allocated to GPU
- **Inference backends:** ROCm-built `llama-server` and `llama-cli` at `/usr/bin/`; system Ollama service runs as user `ollama` (start with `sudo systemctl start ollama`)
- **Models:** GGUFs in `~/models/`; Ollama models in `/usr/share/ollama/.ollama/models/` (root-owned)
- **HF CLI:** installed via `uv tool install "huggingface_hub[cli]"`, authenticated

## Working style

- **Sprint Updates are required.** End of each sprint, fill in the Sprint Update block in IMPLEMENTATION_PLAN.md with what was built, deviations, and notes for the next sprint.
- **Prompts and memory are version-controlled.** Treat changes to `prompts/` and `memory/` like code changes — commit them with intent, diff them when behavior changes.
- **Traces are eval data.** Don't delete old traces; diff them when prompts change.
- **Logging vs traces:** stdlib `logging` for timing/errors/debug; `TraceWriter` for the agent's conversation with itself and its tools. Don't conflate them.

## Out of scope for v1

Loop 2 (interest-tailored deep dive), Loop 3 (project-aware research), scheduling, web search, frontier API adapter, parallel worker models, email delivery, any GUI. Architecture accommodates them; the MVP ships without them. See SPEC's "Future Considerations."
