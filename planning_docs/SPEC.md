# Overnight Research Agent

> Working title. Rename freely.

## Overview

A local-first AI agent that runs overnight, surveys the latest research on open-source and small language models, filters for personal relevance, and delivers a digestible morning brief to an Obsidian vault. Built from scratch — custom agent loop, custom tool calling, version-controlled prompts and memory — as a learning project for agent architecture, agent memory patterns, and overnight autonomous workflows on local hardware (Framework Desktop, Strix Halo, 128GB unified memory).

## Problem Statement

The volume of open-source LLM development (model releases, research papers, technique innovations) has outpaced any one person's ability to follow it casually. Existing newsletters either cover too broad a beat (general AI news) or are paywalled and not personalizable. The user is a hands-on builder of local AI tooling who needs a continuously updated, personally relevant feed of what's new in the local/open-source LLM world — paper releases, model drops, inference techniques, context management methods, RL approaches — without spending an hour a day curating it himself.

Secondarily: the user wants direct experience building an agent loop from scratch, building agent memory patterns, and running an autonomous agent overnight. This project is the vehicle for that learning.

## Goals & Success Criteria

- **G1 (MVP):** A daily markdown brief lands in the user's Obsidian vault each morning containing a curated, personally relevant subset of HuggingFace Daily Papers, with TL;DR + per-paper summaries and links to source.
- **G2:** Personal relevance is meaningfully filtered — the brief omits topics the user has indicated no interest in (e.g., robotics, computer vision-only) and prioritizes topics he has (e.g., local inference, RL, context management, RAG).
- **G3:** The user can read the agent's reasoning trace after the fact and understand what the agent decided and why.
- **G4:** System prompts and memory are version-controlled markdown files the user can edit by hand between runs to experiment with behavior.
- **G5:** The model backend is swappable by config (llama.cpp ↔ Ollama at minimum, with a path to frontier APIs) without changing agent code.
- **G6 (learning):** The user finishes the MVP having personally written an agent loop, a tool-calling parser, and a memory pattern, without leaning on agent frameworks.

## Target Users

A single user: technically competent (Python-comfortable, has built and deployed prior tools), running locally on a Framework Desktop with a Strix Halo APU and 128GB unified memory. Will edit prompts, memory files, and config by hand. Reads markdown briefs on phone via Obsidian sync.

## Core Features

### Daily Paper Survey (MVP)

Each run: fetch HuggingFace Daily Papers for the current date, filter by personal interests, summarize the keepers, write a dated markdown brief into the user's Obsidian vault.

### From-Scratch Agent Loop

A hand-written think → act → observe → repeat loop with explicit iteration cap, final-answer detection, and graceful tool-error recovery. No agent framework (LangGraph, CrewAI, etc.). Loop logic lives in user code.

### Custom XML Tool Calling

Tools are defined in the system prompt with an XML schema. The model emits `<tool_use><name>...</name><input>{...}</input></tool_use>` blocks. A custom parser extracts calls; parse failures are sent back to the model as observations for retry. Native tool calling is intentionally avoided for transparency, portability across models, and learning value.

### Swappable Model Backend

A `ModelInterface` abstraction with concrete implementations for llama.cpp (primary, for parameter-level experimentation) and Ollama (fallback). A `FrontierAdapter` interface is reserved for a later frontier-API implementation. Backend, model name, temperature, and context length are config-driven, not hard-coded.

### Version-Controlled Prompts & Memory

System prompts live as markdown files in the project repo (`prompts/`). Memory files (`memory/Interests.md`, `memory/Seen.md`, `memory/Reflections/*.md`) also live in the repo. The whole project is git-tracked, so prompt experimentation produces a history the user can diff and revert.

### Memory Loop

At run start, the agent reads `memory/Interests.md` (user-curated) and recent entries from `memory/Reflections/` (agent-written). At run end, the agent appends seen paper IDs to `memory/Seen.md` (dedup across days) and writes a brief reflection to `memory/Reflections/YYYY-MM-DD.md` capturing what it noticed about the day's content and what might be worth weighting differently next time.

### Reasoning Traces

Every run writes both a structured `traces/YYYY-MM-DD/trace.jsonl` (one event per line: thoughts, tool calls, tool results, raw model output) and a human-readable `traces/YYYY-MM-DD/trace.md` for skimming. Traces are git-tracked so prompt changes can be evaluated against past behavior.

## Technical Architecture

### System Overview

A single-process Python application invoked by a CLI entrypoint. On invocation:

1. Load config (model backend, model name, vault path, etc.).
2. Initialize the chosen `ModelInterface`.
3. Load the system prompt template and inject the contents of `memory/Interests.md` and the latest reflection.
4. Start the agent loop with a fresh trace file.
5. Loop: model generates → parser extracts tool calls or final answer → tools execute → observations fed back → repeat (max N iterations).
6. On final answer: write brief to Obsidian vault, append to `Seen.md`, generate and save reflection.
7. Close trace files.

Three of the user's three originally-envisioned loops are explicitly deferred. MVP implements **only the broad-sweep loop** restricted to HuggingFace Daily Papers. Interest-tailored loop expansion and active-project-aware loop are out of scope for v1 but are accommodated by the architecture (additional missions = additional system prompts + tool inventories using the same loop core).

### Technology Stack

- **Language/Runtime:** Python 3.12 (managed via `uv`)
- **Inference backends:** llama.cpp (primary; `llama-server` HTTP endpoint) and Ollama (`/api/generate` or `/api/chat`)
- **External CLIs wrapped as tools:** `hf` (HuggingFace Hub CLI; subcommand `hf papers list/read`)
- **Storage:** plain files — markdown for prompts, memory, briefs, and trace.md; JSONL for structured traces. No database.
- **Dependencies (initial):** `httpx` (HTTP to llama.cpp/Ollama), `pydantic` (config validation), `python-dotenv` (env config), standard library `subprocess` for CLI tools. No agent frameworks. No LangChain.
- **Infrastructure:** runs entirely on the Framework Desktop. Output written to Obsidian vault path (synced separately via Obsidian's own sync).

### Data Model

No database. Files only:

```
repo/
├── config.toml                       # model backend, vault path, iteration cap, etc.
├── prompts/
│   ├── system_paper_survey.md        # system prompt for the daily-papers mission
│   └── system_filter.md              # system prompt for the batched relevance filter
├── memory/
│   ├── Interests.md                  # user-curated; read every run
│   ├── Seen.md                       # agent-appended; one paper id per line + date
│   └── Reflections/
│       └── YYYY-MM-DD.md             # agent-written end-of-run note
├── traces/
│   └── YYYY-MM-DD/
│       ├── trace.jsonl               # structured event log
│       └── trace.md                  # human-readable
└── src/
    └── ...
```

Brief output (written to Obsidian vault, not the repo):

```
<obsidian-vault>/Briefs/YYYY-MM-DD.md
```

### Key Design Decisions

1. **Custom XML tool calling, not native function calling.** Better debuggability via plain-text traces, model-portability across local backends, graceful retry on parse failure, and higher learning value. Native tool calling remains a viable later experiment behind the same `ModelInterface`.
2. **`hf papers` CLI as primary data source, not HTML scraping or arxiv API.** Returns structured JSON with `--format json`, no auth required for read operations, no HTML cleanup, and provides both list and full-text read in one tool family. Net: simpler ingest path with no loss of capability.
3. **Markdown everything (prompts, memory, briefs).** Hand-editable, diffable, greppable, version-controllable, and lets the user use Obsidian itself as the reading interface for briefs.
4. **Memory as prompt input, not retrieval.** No vector store, no embeddings. The agent reads `Interests.md` and the latest reflection directly into its system prompt at run start. Simple, transparent, and sufficient for current scale (~60 papers/day, single user).
5. **Two-stage paper processing (batched filter → per-paper summary).** A single model call ranks/filters all ~60 papers using titles + abstracts + Interests.md, returning IDs of keepers. Per-paper summarization runs only on keepers. Trades a small amount of compute for substantially better signal-to-noise and creates a natural seam for later orchestrator/worker model split.
6. **Separate repo from Obsidian vault.** Code, prompts, memory, and traces in a git repo. Briefs written to Obsidian vault (which has its own sync). This keeps version control clean and lets the user delete/reorganize briefs in Obsidian without polluting git history.
7. **Brief reflection at end of each run.** The agent writes a short note about what it saw — themes, surprises, what felt relevant vs noise — and this gets read at the start of the next run. This is the actual "agent memory" loop the user wants experience building.
8. **Single-process, manually-invoked for MVP.** Scheduling (cron / systemd timer / launchd) deferred. The user runs `python -m agent` by hand for v1; once behavior is good, scheduling is a one-sprint addition.

## Constraints & Considerations

### Known Challenges

- **Local-model agentic reliability.** Open-weight instruct models are weaker on multi-step tool-calling than frontier models. Expect to spend prompt-engineering time on the system prompts and on parser leniency. The XML format and retry-on-parse-failure are mitigations.
- **Filter prompt quality.** "Personal relevance" is a fuzzy judgment. Expect to iterate on `system_filter.md` and on `Interests.md` for the first several runs. Reflections should help surface where the filter went wrong.
- **`hf papers` CLI surface stability.** Newer CLI; subcommand surface may change. Wrap it behind a thin adapter so a future version change is a one-file fix.
- **Daily paper volume variance.** ~60 papers is typical but not guaranteed; the filter prompt and context budget need to handle 100+ paper days. Truncate or chunk if needed.

### Out of Scope (v1)

- Scheduling / overnight cron — manual invocation only
- Loop 2 (interest-tailored deep dive beyond daily papers)
- Loop 3 (active-project-aware research using git/blog/commit history)
- Web search tool (Tavily or similar) — deferred to loop 2 phase
- Frontier API backend implementation (interface reserved; no concrete adapter)
- Parallel worker models (small models for summaries, large for orchestration)
- Email delivery
- Any GUI / web UI
- Multi-user support
- Vector store / embeddings-based memory

### Security Considerations

- No secrets in the repo. `HF_TOKEN` (if needed) via `.env`, gitignored.
- Obsidian vault path is local-only; no network exfiltration of brief content.
- Trace files contain raw model output. Gitignore the `traces/` directory if it gets large; alternatively, keep last N days only.

### Future Considerations

- **Loop 2 (interest-tailored):** Extend with web search and arxiv full-text retrieval. Same agent loop, new mission, new system prompt, additional tools.
- **Loop 3 (active-project-aware):** Add `gh` CLI wrapper, read recent commits + READMEs from the user's repos, optionally read his GitHub Pages blog posts for "what am I working on" narrative. New mission, new system prompt, can run after loop 1 in the same overnight invocation.
- **Orchestrator/worker split:** When loop 3 lands, introduce a smaller worker model for parallel summarization and a larger model for orchestration. Both behind the same `ModelInterface`.
- **Frontier API adapter:** Implement a Claude or other frontier-API backend for direct comparison runs against the same prompts.
- **Native tool calling experiment:** Add a `NativeToolCallingAdapter` parallel to the XML adapter; run identical prompts through both and compare trace quality.
- **Scheduling:** systemd user timer (Linux) for the overnight run.
- **Obsidian CLI for vault-aware memory:** Once loops 2 and 3 land, expand memory beyond `Interests.md` and reflections by searching the broader Obsidian vault for related notes the user has written. Two options: the official `obsidian` CLI (v1.12+, rich command surface including `search:context`, `tags`, `properties`, `daily`, but requires the Obsidian GUI to be running and currently behind a Catalyst license — planned for free release) or the community `notesmd-cli` (headless, free, smaller command surface — `create`, `search`, `search-content`, frontmatter ops). Useful tools to add to the registry: `obsidian_search_context` (grep-with-context across vault), `obsidian_read_note` (pull a specific note's content into agent context), and optional `obsidian_set_property` for tagging briefs with run metadata.

---

## Notes for Claude Code

- **No agent frameworks.** Do not introduce LangChain, LangGraph, LlamaIndex, CrewAI, or similar. The agent loop, the parser, and the tool registry are all hand-written. This is a deliberate learning constraint.
- **Strong preference for the standard library + a few thin dependencies.** httpx, pydantic, python-dotenv are fine. Anything else, ask first.
- **Use `uv` for dependency and environment management** — the user has it installed.
- **Keep modules small and named for what they do.** Suggested layout:
  ```
  src/
    config.py          # pydantic settings, loads config.toml + .env
    model/
      base.py          # ModelInterface protocol/ABC
      llamacpp.py
      ollama.py
    tools/
      base.py          # Tool ABC, tool registry, parse-error type
      hf_papers.py     # list + read wrappers around `hf papers`
    agent/
      loop.py          # the think/act/observe loop
      parser.py        # XML tool-call parser
      prompts.py       # loads markdown prompts from disk
    memory/
      io.py            # read/write Interests.md, Seen.md, reflections
    trace/
      writer.py        # JSONL + markdown trace writer
    brief/
      writer.py        # final brief renderer + Obsidian write
    cli.py             # entry point
  ```
- **Parser leniency.** Tolerate extra whitespace, trailing commas in JSON inputs, single vs double quotes — pre-clean before strict JSON parse. On hard parse failure, return a structured error observation to the model rather than crashing the run.
- **Iteration cap.** Hard cap of 25 loop iterations per run as a safety net. Configurable in `config.toml`.
- **Logging vs traces.** Traces capture the agent's conversation with itself and its tools. Standard Python `logging` captures everything else (timing, errors, debug). Don't conflate them.
- **Tests:** unit tests for the parser (a handful of valid + malformed inputs), the tool wrappers (mocking the `hf` CLI subprocess), and the memory I/O. End-to-end test optional in MVP — the smoke test in Sprint 3.1 stands in for it.
- **Docstrings on the public surface of each module.** Type hints throughout. Keep functions short.
- **At end of each sprint, fill in the Sprint Update in IMPLEMENTATION_PLAN.md** with what was built, any deviations from the plan, and anything the next sprint should know.
