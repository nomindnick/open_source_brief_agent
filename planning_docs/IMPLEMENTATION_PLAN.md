# Implementation Plan: Overnight Research Agent

> **Reference:** See [SPEC.md](./SPEC.md) for full project context, architecture decisions, and feature details.

## Overview

Five phases of 1–3 sprints each, sequenced so the agent loop and traces are working end-to-end before any of the actual research pipeline is built. Earliest sprints establish patterns (model interface, tool calling, trace writing) that later sprints reuse. The MVP is reached at the end of Phase 5.

**Estimated Total Time:** ~15–20 hours across 10 sprints.

---

## Phase 1: Foundation

**Goal:** A runnable, configured Python project that can talk to both llama.cpp and Ollama through a single interface.

### Sprint 1.1: Project Scaffolding

**Estimated Time:** 1 hour

**Objective:** Bootstrap the repo, dependency management, and directory layout described in SPEC.md.

**Tasks:**
- [ ] `uv init` and pin Python 3.12
- [ ] Add initial dependencies: `httpx`, `pydantic`, `pydantic-settings`, `python-dotenv`, `pytest`
- [ ] Create the `src/` layout from SPEC's "Notes for Claude Code" (empty modules with docstrings are fine)
- [ ] Create `prompts/`, `memory/`, `traces/` directories with `.gitkeep`
- [ ] Create a placeholder `memory/Interests.md` with 3–5 example lines so the agent has something to read
- [ ] `config.toml.example` and `.env.example` checked in; real `config.toml` and `.env` gitignored
- [ ] `.gitignore`: `.env`, `config.toml`, `__pycache__`, `.venv`, `traces/`
- [ ] README skeleton: install, configure, run
- [ ] `git init`, initial commit

**Acceptance Criteria:**
- `uv sync` succeeds on a fresh clone
- `python -m agent --help` runs and prints a usage message (entrypoint exists, even if it does nothing yet)
- Repo is committable with no secrets

**Sprint Update:**
> _[To be completed]_

---

### Sprint 1.2: Model Interface

**Estimated Time:** 2–3 hours

**Objective:** A `ModelInterface` abstraction with working llama.cpp and Ollama implementations, selected by a named profile in config. Reasoning-model output is captured as a first-class field, not mixed into agent-facing content.

**Tasks:**
- [ ] `src/model/base.py`: define `ModelInterface` (ABC or Protocol) with a single `complete(messages: list[dict], **kwargs) -> ModelResponse` method. Keep it sync for v1.
- [ ] `src/model/base.py`: define `ModelResponse` dataclass with fields `content: str`, `reasoning: str | None`, `raw: str`, `usage: dict`.
- [ ] `src/model/llamacpp.py`: implementation hitting `llama-server`'s OpenAI-compatible `/v1/chat/completions` endpoint via httpx. Regex-splits `<think>…</think>` blocks out of the response, returning the thinking text in `reasoning` and the rest in `content`. If no think blocks present, `reasoning=None`.
- [ ] `src/model/ollama.py`: implementation hitting Ollama's `/api/chat` endpoint via httpx. Sends `"think": true` when the profile is marked `supports_thinking = true`; reads `message.thinking` into `reasoning` and `message.content` into `content`.
- [ ] `src/config.py`: pydantic-settings loading `config.toml` + `.env`. Supports a `[models.<profile_name>]` table list — each profile has `backend: Literal["llamacpp", "ollama"]`, `model_name`, `base_url`, `temperature`, `max_tokens`, `context_length`, `supports_thinking: bool = False`. Top-level keys: `default_model` (the profile name to use when CLI doesn't override), `vault_path`, `iteration_cap`.
- [ ] Factory function `get_model(config, profile_name: str | None = None) -> ModelInterface` — falls back to `default_model` when no name given.
- [ ] Pick known-working defaults for `config.toml.example`: a llama.cpp profile pointing at `~/models/qwen3-30b-a3b-q8/Qwen3-30B-A3B-UD-Q8_K_XL.gguf` and an Ollama profile pointing at `qwen3.5:9b` (both reasoning-capable; both verified working on the dev machine 2026-05-13). Set each profile's `context_length` to a deliberately generous value the model supports — start at 32768 for qwen3-30b (raise toward 65536+ if benchmarking shows VRAM headroom) and 32768 for qwen3.5:9b. Don't pick a small "safe" default; see SPEC's "Context length: default generous."
- [ ] For the llama.cpp profile, document that `llama-server` must be started with a matching `-c` flag (e.g., `llama-server -c 32768`) — the profile's `context_length` is the value the agent code budgets against; the server has to actually allocate that much KV cache.
- [ ] Manual smoke test script `scripts/smoke_model.py` that prints content + (if present) reasoning for a given profile.
- [ ] `scripts/benchmark_model.py`: takes a profile name; runs a fixed prompt designed to stress XML tool-call formatting (a stub prompt for now, will be replaced with the test mission from Sprint 2.2 once it exists); prints time-to-first-token, total wall time, tokens generated, and dumps content+reasoning. This is the workflow for "I downloaded a new model, does it handle our format well?"

**Acceptance Criteria:**
- With llama.cpp running locally, `python scripts/smoke_model.py --model <llamacpp-profile>` returns a non-empty `content`. For a reasoning model, `reasoning` is also non-empty and is *not* present in `content`.
- Switching `--model <ollama-profile>` uses Ollama with no code changes.
- A non-reasoning model (or `supports_thinking = false`) returns `reasoning=None` cleanly.
- Bad config (missing field, bad URL, unknown profile name) raises a clear validation error, not a stack trace deep in httpx.
- `scripts/benchmark_model.py <profile>` runs end-to-end and produces a summary block usable for comparing models.

**Sprint Update:**
> _[To be completed]_

---

## Phase 2: Agent Loop

**Goal:** A working think-act-observe loop with custom XML tool calling and full trace output, exercised against a trivial echo tool.

### Sprint 2.1: XML Tool Call Format & Parser

**Estimated Time:** 1–2 hours

**Objective:** Define the tool-call schema, write a lenient parser, and validate it against malformed inputs.

**Tasks:**
- [ ] Document the XML schema in `prompts/_tool_calling_format.md` (block name, expected fields, examples of well-formed and malformed inputs, final-answer marker). This file is meant to be included by reference in every mission's system prompt.
- [ ] `src/tools/base.py`: define `Tool` ABC (`name`, `description`, `input_schema`, `run(input: dict) -> str`), `ToolRegistry`, `ToolCall`, `ParseError`
- [ ] `src/agent/parser.py`: extract `<tool_use>` blocks and `<final_answer>` blocks from raw model output. Be lenient: tolerate whitespace, trailing commas in the JSON `<input>`, single quotes (pre-clean to double). Return a structured result type that says "found tool call X / found final answer / parse error with reason."
- [ ] Implement a stub `EchoTool` (`tools/echo.py`) that returns whatever string it's given, for testing the loop end-to-end without external dependencies.
- [ ] Unit tests: well-formed call, multiple calls in one response, malformed JSON, missing closing tag, final-answer detection, no tool call present (treat as final answer fallback per system prompt convention).

**Acceptance Criteria:**
- `pytest tests/test_parser.py` passes for at least 6 cases covering happy path and malformed inputs
- Parse errors return a descriptive reason string suitable for sending back to the model

**Sprint Update:**
> _[To be completed]_

---

### Sprint 2.2: Agent Loop Core

**Estimated Time:** 1–2 hours

**Objective:** The think-act-observe loop. Given a model, a registry, and a system prompt, it should run a task to completion or hit the iteration cap.

**Tasks:**
- [ ] `src/agent/loop.py`: `run_agent(model, registry, system_prompt, user_task, max_iter) -> AgentResult`. The loop:
  1. Initialize conversation with system + user message
  2. Call model
  3. Parse output → final answer? exit with result. Tool call? execute via registry, append result as new message, continue.
  4. Parse error? Append error description as observation, continue.
  5. Hit `max_iter`? Exit with timeout result.
- [ ] `src/agent/prompts.py`: load a system prompt file from `prompts/`, do simple `{{placeholder}}` interpolation for memory contents.
- [ ] Hook up `EchoTool` and a `prompts/system_test.md` that instructs the model to call echo twice and then return a final answer.
- [ ] Have the loop accept a `TraceWriter` (or a no-op stand-in for this sprint) and call its methods at every event boundary — model turn, tool call, tool result, parse error, final answer. The real writer arrives in Sprint 2.3; this sprint just ensures the loop is shaped to feed it. Pass `ModelResponse.reasoning` and `.content` through as separate args so 2.3 can render them distinctly.
- [ ] Manual run: `python -m agent --mission test` exercises the loop end-to-end.

**Acceptance Criteria:**
- Manual test mission runs to a final answer in under 25 iterations
- Iteration cap is respected (set to 3 for the test, confirm it bails cleanly)
- Tool call → tool result → next model turn is visible (via the trace writer stub's stdout output is fine for this sprint)
- The loop's call sites already pass `reasoning` and `content` separately — no retrofit needed in Sprint 2.3

**Sprint Update:**
> _[To be completed]_

---

### Sprint 2.3: Trace Logging

**Estimated Time:** 1 hour

**Objective:** Every run produces a structured JSONL trace and a human-readable markdown trace.

**Tasks:**
- [ ] `src/trace/writer.py`: `TraceWriter` that opens `traces/YYYY-MM-DD/<mission>/trace.jsonl` and `trace.md` on init. Methods: `log_model_turn(content, reasoning)`, `log_tool_call`, `log_tool_result`, `log_parse_error`, `log_final_answer`, `close`.
- [ ] Each JSONL line is a dict with `timestamp`, `event_type`, and event-specific fields. Model-turn events carry both `content` and `reasoning` (the latter may be null).
- [ ] Markdown version renders as a chronological narrative with section headers per turn — readable on phone via Obsidian. When `reasoning` is present, render it as a blockquote above the turn's content/action (visually subordinate but visible). This is the primary surface for reviewing agent decisions.
- [ ] Wire `TraceWriter` into the agent loop from the start — pass it into `run_agent`. (Sprint 2.2's loop should already be calling these methods; this sprint completes the writer side.)
- [ ] Handle multiple runs same day: append a suffix (`-1`, `-2`) to avoid overwrite.

**Acceptance Criteria:**
- Running the test mission produces `trace.jsonl` and `trace.md` under `traces/<today>/test/`
- `trace.md` is readable as a coherent story of what the agent did
- When the chosen model is a reasoning model, the trace's markdown clearly shows the reasoning per turn as a quoted block above the action
- `trace.jsonl` parses as valid JSON line-by-line

**Sprint Update:**
> _[To be completed]_

---

## Phase 3: HuggingFace Papers Pipeline

**Goal:** The agent can fetch the day's papers, filter them against `Interests.md`, and summarize the keepers.

### Sprint 3.1: HF Papers Tools + Smoke Test

**Estimated Time:** 1–2 hours

**Objective:** Two `Tool` implementations wrapping `hf papers list` and `hf papers read`, exercised end-to-end by a real agent run.

**Tasks:**
- [ ] Install and authenticate the `hf` CLI on the dev machine: `uv tool install "huggingface_hub[cli]"`, then `hf auth login` with a HF token (read-scope is sufficient for `papers list/read`). Confirm with `hf papers list --format json | head` returning structured JSON. Add the install steps to README.
- [ ] `src/tools/hf_papers.py`:
  - `HfPapersListTool`: input `{"date": "YYYY-MM-DD"}` (optional, defaults to today); subprocess-calls `hf papers list --date <date> --format json`; returns parsed JSON as a compact summary (id, title, abstract truncated to ~500 chars, upvotes).
  - `HfPapersReadTool`: input `{"id": "<arxiv-id>"}`; subprocess-calls `hf papers read <id>`; returns markdown content. The raw output wraps the paper text in arxiv HTML chrome (nav links, "Report GitHub Issue" form, TOC, image refs); strip these with a small regex pass before returning.
- [ ] Both tools: timeout on subprocess (60s), capture stderr into a structured error, never raise out of `run()`.
- [ ] Write `prompts/system_paper_survey.md` v1: a simple system prompt instructing the model to list papers, pick a few interesting ones, read one in full, and return a final summary. Pure smoke-test prompt, not the real one yet.
- [ ] Manual run: `python -m agent --mission paper_survey` against today's date.

**Acceptance Criteria:**
- Real `hf papers list` succeeds and returns >0 papers
- The agent calls list, then read on at least one paper, then issues a final answer
- Tool errors (e.g., bad arxiv id) come back as model observations, not exceptions

**Sprint Update:**
> _[To be completed]_

---

### Sprint 3.2: Interest-Based Filtering

**Estimated Time:** 1–2 hours

**Objective:** Replace "model picks a few interesting papers ad hoc" with a deliberate two-stage filter: read all papers' titles+abstracts in a single batched call against `Interests.md`, return IDs of keepers.

**Tasks:**
- [ ] `prompts/system_filter.md`: system prompt for the filter stage. Takes Interests.md and the full paper list, returns a JSON array of keeper IDs with one-line justifications. No tool calling — a single completion.
- [ ] `src/agent/filter.py`: thin module that runs the filter stage. Reads Interests.md, constructs the prompt, calls model, parses JSON response (lenient — strip code fences, handle leading/trailing whitespace).
- [ ] Restructure the mission flow in `loop.py` (or a thin orchestrator in `cli.py`): list papers → filter stage → agent loop with keepers in context to summarize. The filter stage is *not* inside the agent loop; it's a deterministic pre-step.
- [ ] Flesh out `memory/Interests.md` with the user's real interests (have him fill this in before the run).
- [ ] Manual run: confirm the filter trims ~60 papers to a handful of relevant ones.

**Acceptance Criteria:**
- Filter output is parseable JSON
- Filter rejects an obvious negative-interest paper (e.g., a pure robotics paper if Interests.md says no robotics)
- Filter keeps an obvious positive-interest paper
- Filter call is logged to the trace

**Sprint Update:**
> _[To be completed]_

---

### Sprint 3.3: Per-Paper Summarization

**Estimated Time:** 1–2 hours

**Objective:** For each keeper, the agent reads the paper and produces a short summary suited for the morning brief.

**Tasks:**
- [ ] Update `prompts/system_paper_survey.md` to its real v1: given a list of keeper IDs and brief justifications from the filter, read each one and produce a per-paper summary (TL;DR, why it might matter to the user, one quote-worthy detail, link).
- [ ] Decide loop structure: single agent loop iterating tools for all keepers, OR a programmatic for-loop calling `hf papers read` and a single-turn model call per keeper. **Recommend the latter** — more predictable, easier to debug, and the "agent loop" learning value is already captured in the test mission and the smoke test.
- [ ] Implement the per-keeper summary as a non-agentic model call in `src/agent/summarize.py` (consistent with the filter being non-agentic).
- [ ] Output is a list of structured summaries ready for the brief writer.

**Acceptance Criteria:**
- Run produces a coherent summary per keeper
- A typical keeper summary is 4–8 sentences plus the link
- Summarization failures (model produces garbage, paper read fails) degrade gracefully — partial brief is better than no brief

**Sprint Update:**
> _[To be completed]_

---

## Phase 4: Memory & Output

**Goal:** Briefs land in Obsidian; memory updates so tomorrow's run benefits from today's.

### Sprint 4.1: Obsidian Brief Writer

**Estimated Time:** 1 hour

**Objective:** Render summaries into a polished markdown brief and write it into the Obsidian vault.

> **Implementation note:** Use plain Python file I/O (`pathlib.Path.write_text`) — *not* the Obsidian CLI. The official `obsidian` CLI and community `notesmd-cli` both exist and are real options, but for the single-file write the MVP needs, plain I/O is simpler, faster, requires no external dependency, and doesn't care whether the Obsidian GUI is running. The CLI becomes interesting only when the agent needs to *read* the broader vault (deferred to loops 2/3 — see SPEC's Future Considerations).

**Tasks:**
- [ ] `src/brief/writer.py`: render a brief with frontmatter (date, paper count, mission), a TL;DR section, and per-paper sections with link to source.
- [ ] Write to `<vault_path>/Briefs/YYYY-MM-DD.md`. If a brief already exists for today, write to `YYYY-MM-DD-run-2.md` (don't overwrite — runs are cheap, accidental data loss is expensive).
- [ ] Add a top-of-brief "How I picked these" paragraph drawn from the filter's justifications so the user can sanity-check the filter on phone.
- [ ] Confirm Obsidian renders the resulting file correctly (frontmatter, links, headings).

**Acceptance Criteria:**
- File appears in the Obsidian vault and syncs to phone
- All links are clickable
- A complete brief is ≤ a phone-readable length (no walls of text)

**Sprint Update:**
> _[To be completed]_

---

### Sprint 4.2: Memory Write-Back

**Estimated Time:** 1–2 hours

**Objective:** End-of-run memory updates: `Seen.md` append + reflection note. Start-of-run memory reads use these.

**Tasks:**
- [ ] `src/memory/io.py`:
  - `read_interests() -> str` (returns markdown content)
  - `read_latest_reflection() -> str | None` (returns most recent file in `memory/Reflections/`)
  - `read_seen_ids() -> set[str]` (parses `Seen.md`)
  - `append_seen_ids(ids: list[str], date: str) -> None`
  - `write_reflection(date: str, content: str) -> None`
- [ ] In the filter stage: exclude papers whose ID is already in `Seen.md` *and* was seen >= 1 day ago. (Same-day reruns shouldn't deduplicate — useful for iterating prompts.)
- [ ] Add a final non-agentic model call: "reflect on today's brief — what themes did you see, what felt high-signal vs noise, what should be weighted more in tomorrow's filter?" Write the result to `memory/Reflections/YYYY-MM-DD.md`.
- [ ] At run start, inject `Interests.md` and the latest reflection into the filter system prompt.
- [ ] Append all keeper IDs (not all surveyed IDs — only ones that made the brief) to `Seen.md`.

**Acceptance Criteria:**
- After two consecutive daily runs, the second run doesn't re-surface yesterday's keepers
- A reflection file exists per run and is non-trivial (>3 sentences, mentions specific themes)
- The next run's filter system prompt visibly includes the previous reflection (check the trace)

**Sprint Update:**
> _[To be completed]_

---

## Phase 5: Polish

**Goal:** The system is something the user can actually rely on each morning. MVP is reached at end of this phase.

### Sprint 5.1: End-to-End Dry Run, Documentation, Configurable Parameters

**Estimated Time:** 1–2 hours

**Objective:** Treat this like the night before launch. Run it the way it will actually be used, fix what's awkward, document for future-you.

**Tasks:**
- [ ] Full end-to-end run as if it were an overnight run: fresh terminal, real Interests.md, real model, real vault. Time it. Note any issues.
- [ ] Move any hard-coded values (timeouts, iteration cap, filter top-N, model temperatures per stage) into `config.toml`.
- [ ] README: install steps, `hf` CLI setup, llama.cpp/Ollama setup, config walkthrough, how to edit Interests.md, where briefs land, how to read traces, how to recover from a failed run.
- [ ] `TROUBLESHOOTING.md`: model not responding, `hf` CLI auth issues, Obsidian path wrong, parse failures.
- [ ] Add a `--dry-run` flag that runs everything except the final write-to-vault, for iterating on prompts without polluting the brief archive.
- [ ] Optional: a `Makefile` or `justfile` with `run`, `dry-run`, `test`, `lint` targets.

**Acceptance Criteria:**
- End-to-end run completes in a reasonable time on Strix Halo with the chosen local model (record actual time in Sprint Update)
- A fresh-eyes read of README is enough to set the project up on a new machine
- All config is in `config.toml`; no magic numbers in code

**Sprint Update:**
> _[To be completed]_

---

## Implementation Notes

### Dependencies Between Sprints

- Sprint 1.2 (model interface) blocks everything after it.
- Sprint 2.1 (parser) and 2.2 (loop) are tightly coupled — finish 2.1 first.
- Sprint 2.3 (traces) can technically slip to after Phase 3 but is much more valuable for debugging Phase 3 prompts, so keep it ordered.
- Sprint 3.2 (filter) depends on Sprint 3.1 (HF tools) being real, not mocked.
- Sprint 4.2 (memory write-back) requires Sprint 3.2 to exist so there's something to update.

### Testing Strategy

- **Unit tests per sprint where natural:** parser (Sprint 2.1), tool wrappers with mocked subprocess (Sprint 3.1), memory I/O (Sprint 4.2).
- **No end-to-end test framework in MVP.** The smoke runs in Sprints 2.2, 3.1, and 5.1 substitute for an e2e harness. This is a small project for one user; over-investing in tests is the wrong trade.
- **Traces are eval data.** Keep them, diff them when prompts change, use them as the basis for any future eval harness.

### Definition of Done

A sprint is complete when:
1. All tasks are checked off
2. Acceptance criteria are met
3. Code runs without errors
4. Sprint Update is filled in with key decisions, deviations from plan, and notes for the next sprint
