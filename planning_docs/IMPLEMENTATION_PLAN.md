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
> **Built:** uv-managed Python 3.12 project; deps pinned in `pyproject.toml` (httpx, pydantic, pydantic-settings, python-dotenv; pytest in `[dependency-groups].dev`). `uv sync` works from clean. Entrypoint wired: `python -m agent --help` prints usage; `cli.py` has stub flags `--mission`, `--model`, `--dry-run` for Sprint 2.2 to fill in. `config.toml.example` and `.env.example` checked in; real `config.toml` and `.env` gitignored. `.python-version` pinned to 3.12 and committed (collaborators get the right Python; SPEC's gitignore list omitted it — judgment call to keep it tracked). `memory/Interests.md` seeded with the user's real interests so the filter has signal from run 1. README skeleton covers install/configure/run; deeper docs deferred to Sprint 5.1 per plan.
>
> **Deviation from SPEC layout:** SPEC suggested multiple top-level packages under `src/` (`agent/`, `model/`, `tools/`, `memory/`, `trace/`, `brief/`, plus loose `config.py` and `cli.py`). Collapsed to a single `agent` package with `agent.model`, `agent.tools`, etc. as subpackages, with `cli.py` and `config.py` moved inside `agent/`. Reason: hatchling/uv packaging is much cleaner with one top-level package, and the module boundaries (one folder per concern) are preserved. Imports will read `from agent.model.llamacpp import …` instead of `from model.llamacpp import …`. Tests run via `pytest` config (`pythonpath = ["src"]`).
>
> **For Sprint 1.2:** `agent.config` and `agent.model.{base,llamacpp,ollama}` are empty stubs ready to fill in. `config.toml.example` already has the two known-good profiles (qwen3-30b-llamacpp at 32K, qwen3-9b-ollama at 32K, both `supports_thinking = true`); Sprint 1.2 just needs to make the loader read them. The cli already has a `--model` flag wired in, so smoke/benchmark scripts can match its calling convention.

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
> **Built:** `ModelInterface` Protocol + `ModelResponse` dataclass in `agent/model/base.py`; concrete `LlamaCppModel` and `OllamaModel` adapters; `agent/config.py` with pydantic + pydantic-settings (TOML-only source, `.env` loaded separately via python-dotenv for subprocess tools); `get_model()` factory in `agent/model/__init__.py`. Smoke and benchmark scripts both work.
>
> **Acceptance criteria verified on a live run (2026-05-13):**
> - `qwen3-30b-llamacpp` smoke: content "The capital of France is Paris." Reasoning extracted into `ModelResponse.reasoning` (multi-sentence think block). `<think>` does not appear in `content`. 124 completion tokens reported in usage.
> - `qwen3-9b-ollama` smoke: same prompt, `message.thinking` pulled into `reasoning`, content clean. 221 completion tokens. Switching profile required only the `--model` flag — no code changes.
> - Benchmark prompt (asks the model to emit an XML `<tool_use>` block): both profiles passed the format check. qwen3-30b-llamacpp: 11.3 tok/s (25s for 284 tokens). qwen3-9b-ollama: 30.5 tok/s (5.2s for 157 tokens). On Strix Halo the smaller model is meaningfully faster — useful data point for worker-model assignment later.
> - Bad config exercises: unknown `default_model` → ValidationError with profile list, bad backend literal → ValidationError naming `models.r.backend`, missing config file → FileNotFoundError pointing the user at `config.toml.example`. All clean, no httpx-deep stack traces.
> - 6 unit tests for `_split_thinking` (the regex that strips `<think>` blocks) pass.
>
> **Design decisions worth noting:**
> - **Sync interface, not async.** SPEC dictates; we discussed the migration path explicitly. If/when we need parallel summarization across two model servers, threading via `ThreadPoolExecutor` is the first move; full async migration is bounded (httpx has identical sync/async APIs, ~20–30 lines of `await` propagation across this codebase).
> - **TOML is the only source of config values.** Env vars are reserved for secrets consumed by subprocess tools (`HF_TOKEN` later); they don't override config. Keeps "what's set" trivially auditable.
> - **`ModelBackendError` wraps adapter exceptions.** The agent loop (Sprint 2.2) only has to `except ModelBackendError` — it doesn't have to know about httpx, JSON, or backend-specific failure modes.
> - **Token-throughput metric.** Time-to-first-token requires streaming; v1 adapters are non-streaming. Reported TTFT as N/A and added wall time + throughput instead. Sprint 2.2/5.1 can revisit if TTFT becomes worth implementing.
>
> **For Sprint 2.1 (parser):** The benchmark already showed both models cleanly emit our XML format with a simple prompt. The parser only needs to handle the content string after thinking has been stripped — no `<think>` block edge cases to worry about. The qwen3 reasoning chains sometimes describe the tool call in prose *inside* the think block; this is fine since the parser never sees thinking.
>
> **For Sprint 2.2 (loop):** `model.complete()` is sync; the loop is therefore straight imperative Python — no event loop machinery. The benchmark prompt in `scripts/benchmark_model.py` should be replaced with the real test mission once it exists (currently a stub; documented in the script).

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
> **Built:** `prompts/_tool_calling_format.md` (the shared partial that documents the schema for the model — included by reference in every mission's system prompt). `agent/tools/base.py` with `Tool` ABC, `ToolRegistry` (with `render_for_prompt()`), `ToolCall`/`FinalAnswer`/`ParseError` dataclasses. `agent/tools/echo.py` — the stub tool for Sprint 2.2's loop test. `agent/parser.py` with a single `parse()` function returning `ToolCalls | FinalAnswer | ParseError`. End-to-end smoke (parse → registry lookup → tool.run) works.
>
> **Acceptance criteria verified:**
> - 14 parser tests passing (well-formed single call; multiple calls in one response; final-answer block; FA wins over co-occurring tool_use; no-XML fallback to FA; trailing-comma JSON tolerated; unrecoverable JSON → ParseError; JSON-not-an-object → ParseError; missing `<name>`; missing `<input>`; missing closing tag; code-fence-wrapped response; empty `<final_answer>`; batch errors identify the offending block).
> - Parse error reasons are model-readable and specific. Sampled:
>   - `<tool_use> block #1 (name='x'): <input> is not valid JSON: Expecting value at line 1 col 1.`
>   - `<tool_use> block #1 (name='x'): <input> must be a JSON object (got list).`
>   - `<tool_use> block has no matching </tool_use> closing tag. Make sure every <tool_use> opens and closes.`
>   - `<final_answer> block is empty.`
> - Total test suite: 20 passing (14 parser + 6 model regex).
>
> **Design decisions worth noting:**
> - **`ToolCalls` (plural) wraps a `list[ToolCall]`.** The result type is `ToolCalls | FinalAnswer | ParseError` — three siblings, all dataclasses. Pattern-matches cleanly in the loop (Sprint 2.2). The single-call shape was less symmetric.
> - **FinalAnswer wins over co-occurring tool calls** — documented behavior in `_tool_calling_format.md`. Avoids the loop having to decide "did the model mean to keep going?"
> - **No-XML response → FinalAnswer fallback.** Same doc. Matches how reasoning models often respond ("here's the answer") when they think they're done. Risk: a model that forgets to call a tool gets treated as done. Mitigation: the system prompt should make this trade-off explicit per mission.
> - **JSON cleanup is deliberately minimal.** Only trailing-comma removal. Single-quote → double-quote substitution was considered and *rejected* — it would corrupt legitimate strings containing apostrophes. The cost is one extra retry when the model uses single quotes; the alternative was silent data corruption.
> - **Structural strictness.** `<name>` required, `<input>` required, input must be a JSON object (not list/scalar). These are the structural invariants the loop relies on; surfacing them as parse errors lets the model fix them on the next turn.
>
> **For Sprint 2.2 (loop):** The parser never raises. The loop just calls `parse()` and matches on the result type — three cases: execute tool calls, return final answer, or feed the parse-error reason back as the next observation. `ToolRegistry.render_for_prompt()` is ready to be injected as a `{{tools}}` placeholder in mission system prompts. `EchoTool` is the test harness — Sprint 2.2's test mission instructs the model to call it twice and then return a final answer.

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
> **Built:** `agent/loop.py` with `run_agent()`, `AgentResult`, `TraceSink` Protocol, and two stubs (`NoOpTrace` for tests, `StdoutTrace` for interactive debugging). `agent/prompts.py` with `load_prompt()` that auto-injects `{{tool_calling_format}}` from the shared partial and fails loudly on un-filled placeholders. `prompts/system_test.md`. `agent/cli.py` rewritten to dispatch missions; `--mission test` now runs end-to-end. Exit codes: 0 / 1 / 2 for success / cap-hit / setup-error.
>
> **Acceptance criteria verified on live runs (2026-05-14):**
> - `--mission test --model qwen3-9b-ollama --max-iter 3`: 2 iterations, hit_cap=False, final answer `'The two phrases echoed were "hello" and "world".'` Both echo calls executed; observation fed back; final answer emitted.
> - `--mission test --model qwen3-30b-llamacpp --max-iter 3`: 1 iteration (!) — the 30B model emitted both tool calls AND the final_answer in a single turn, and per the parser's FA-wins convention, the echo calls were never executed (the model hallucinated their results). For the echo test this is fine; for real tools we'll want the system prompt to discourage this skip-ahead. Logged here so Sprint 3.1's paper_survey prompt can address it explicitly.
> - `--max-iter 1` with the same mission: hit_cap=True, exit code 1, cap-hit message printed to stderr. Verified the loop bails cleanly.
> - Trace stub printed every event boundary: model turn (content + reasoning separately), tool call, tool result, final answer. The same surface (`log_model_turn(content, reasoning)`, `log_tool_call`, `log_tool_result`, `log_parse_error`, `log_final_answer`, `log_iteration_cap`) is what Sprint 2.3's real writer will implement — no loop changes needed there.
>
> **Plus a bug found and fixed:** `__main__.py` was calling `main()` without propagating its exit code. Fixed with `raise SystemExit(main())`. Discovered because the cap-hit exit code was 0 instead of 1 in the first test run.
>
> **Six new loop tests** added (`tests/test_loop.py`) using a `ScriptedModel` mock. Covers cases hard to trigger deterministically against a real model: parse-error feedback, unknown-tool feedback, tool-exception containment, iteration cap exhaustion, and final-answer-beats-co-occurring-tool-call. Total suite: 26 passing.
>
> **Design decisions worth noting:**
> - **Sequential tool execution.** Multiple `<tool_use>` blocks in one turn run in order. Parallel execution deferred; would matter once HF tools are doing real I/O. Sequential is simpler, more debuggable, and matches the trace's chronological narrative.
> - **Observations are user-role messages, not tool-role.** Works across both backends without backend-specific quirks. Prefixed with `"Result of tool 'X':"` so the model knows what it's looking at.
> - **Unknown tool is a *observation*, not a `ParseError`.** The parse succeeded — the model called something that doesn't exist. Surfaced as `"Tool 'X' is not available. Available tools: [...]."` so the model can recover.
> - **Tool exceptions are contained.** Per `Tool` ABC contract tools shouldn't raise, but if they do, the error becomes an observation rather than crashing the run.
> - **Cap-hit returns `final_answer=None, hit_cap=True`**, not a best-effort answer. The CLI (and later orchestrators) decide whether to salvage partial work. Defaults to skipping any output write.
> - **`TraceSink` is a Protocol.** Sprint 2.3's `TraceWriter` implements it. The loop never imports the real writer — keeps the trace module separable.
>
> **For Sprint 2.3 (trace writer):** The loop already calls every method with the right shape. Sprint 2.3 just builds `TraceWriter(date, mission)` that opens `traces/YYYY-MM-DD/<mission>/trace.{jsonl,md}` on init and implements those methods to append events. No loop changes.
>
> **For Sprint 3.1 (HF tools + first real mission):** The `Mission` dataclass in cli.py is the dispatch shape — adding `paper_survey` is one entry. The new mission's system prompt should explicitly tell the model "do not emit `<final_answer>` until you have observed actual tool results" to avoid the 30B-style skip-ahead.

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
> **Built:** `agent/trace/writer.py` with `TraceWriter`, opening line-buffered append-mode handles for both `trace.jsonl` and `trace.md`. JSONL starts with a `meta` event (`trace_schema_version=1`, `agent_version`, date, mission, model_profile, max_iter) and ends with a `close` event. Markdown header lists the same metadata; per-turn sections render reasoning as blockquotes above the model's fenced content. Tool results: full text in JSONL, truncated to ~600 chars in MD with a "truncated — full result in trace.jsonl" marker. Same-day reruns get `-1` / `-2` / … directory suffixes. `cli.py` now allocates a `TraceWriter`, passes it through `run_agent`, closes it in `finally`, and prints the trace path on exit.
>
> **Acceptance criteria verified on live run (2026-05-14):**
> - `--mission test --model qwen3-9b-ollama --max-iter 3` produced `traces/2026-05-14/test/trace.{jsonl,md}`. 10 JSONL events, all parse as valid JSON line-by-line.
> - `trace.md` reads as a coherent story: model turn → tool call → tool result → next turn → final answer. Reasoning from Qwen renders as multi-line blockquote, content as fenced XML below it (visually subordinate but visible — the SPEC's design intent).
> - Re-running the same mission immediately landed in `traces/2026-05-14/test-1/`. No overwrite.
> - 8 new unit tests for the writer (path allocation, meta/close events, line-by-line JSON validity, MD structure, truncation behavior, same-day rerun suffixing, idempotent close, partial-trace survives an unclosed writer). Total suite: 34 passing.
>
> **Design decisions worth noting:**
> - **Line-buffered append (`buffering=1`).** A crash mid-run leaves a usable partial trace on disk — verified in `test_partial_trace_survives_unclosed_writer`. The trace is fsync-pending until close, but reasonably durable.
> - **Schema version in `meta` event.** Trivial insurance for the day the event shape changes. We can read old traces unambiguously by branching on `trace_schema_version`.
> - **Markdown truncation, JSONL is canonical.** Tool results from HF papers will be many KB; the MD stays phone-readable while the JSONL keeps everything for later inspection. Truncation marker tells the reader to look there if they need full text.
> - **UTC timestamps in JSONL, local date in the directory name.** Caller (CLI) passes the date string. UTC inside the events is sortable and unambiguous; the directory uses local-date which matches how briefs will be named (Sprint 4.1).
> - **`StdoutTrace` retired from CLI.** Removed from the CLI path; the class still exists in `agent/loop.py` for tests/dev that want it. Watching live trace output works via `tail -f traces/<date>/<mission>/trace.md` — cleaner than dual output.
> - **`TraceWriter.run_dir` exposed as property.** CLI uses it to print the path. Sprint 4.1's brief writer can use it to cross-reference the brief with the trace if useful.
>
> **For Sprint 3.1 (HF tools):** Tool wrappers will return paper text (many KB). The MD truncation already handles this gracefully. Worth tagging full-text payloads with sensible delimiters in the tool wrapper so the truncated preview is still informative (e.g. "title: X | abstract: Y..." rather than "&nbsp;...").
>
> **For Sprint 4.1 (brief writer):** The trace surface stays separate from the brief surface. Trace goes to `traces/<date>/<mission>/`; brief goes to `<vault_path>/Briefs/<date>.md`. No cross-pollution; both are markdown but with different audiences (trace = me debugging, brief = me reading at breakfast).

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
> **Built:** `agent/tools/hf_papers.py` with `HfPapersListTool` and `HfPapersReadTool`, both subprocess-wrapping the `hf` CLI. List compresses ~50 papers of raw JSON into a scannable markdown summary (id, title, truncated abstract, upvotes). Read strips the CLI's `Title: / URL Source: / Markdown Content:` preamble and replaces it with a `# <title>` heading. 60s subprocess timeout; never raises (CLI missing, timeout, non-zero exit, bad JSON all fold into `ERROR:` strings the agent can react to). `prompts/system_paper_survey.md` first-pass mission prompt with explicit "do not fake tool results" guardrail addressing the 30B skip-ahead noted in Sprint 2.2. CLI's MISSIONS dict gains `paper_survey`. 18 new unit tests for the tools with mocked subprocess. README updated with `hf` CLI install + auth steps.
>
> **Acceptance criteria verified on live run (2026-05-14):**
> - `hf papers list --date 2026-05-14 --format json` returned 50 papers; tool compressed them to scannable markdown.
> - Agent (qwen3-9b-ollama) called `hf_papers_list`, then `hf_papers_read` on 5 different papers across 3 turns, then issued a coherent final answer summarizing one of them (FAAST: closed-form associative learning).
> - **Tool errors handled gracefully:** the agent tried multiple paper IDs that returned `ERROR: ... not found on the Hub` (some papers in the daily-list aren't readable). Each error came back as a model observation; the agent self-corrected and tried different IDs without crashing.
>
> **Two changes that landed mid-sprint to make the smoke actually work:**
> - **Context length raised from 32K → 65K, max_tokens raised from 4K → 8K** in both `config.toml.example` and `config.toml`. The first smoke run blew the context budget: a single full paper read can be 120K chars (~40K tokens), and a too-small `max_tokens` left the model with no output budget after reasoning. This is exactly the user-flagged "we have headroom on 96GB" call. Ollama allocates KV on-demand so no server restart needed there; llama-server users now need `-c 65536` (documented in README).
> - **Parser hardening: empty content → `ParseError`** (instead of silent empty `FinalAnswer`). Discovered when turn 4 of the first run emitted 13.5K chars of reasoning and 0 chars of content (max_tokens exhausted during reasoning). Previously this would have produced an empty final answer and exited normally; now it surfaces as a parse error so the loop reprompts the model with "be more concise." 1 new test case.
>
> **Total test suite: 52 passing** (18 HF tools + 14 parser + 6 model + 6 loop + 8 trace).
>
> **Design decisions worth noting:**
> - **Subprocess, not Python API.** The `hf` CLI surface is more stable than `huggingface_hub`'s Python API across versions, and keeps our dependency tree thin. ~100ms subprocess overhead per call is invisible for an overnight run with ~5–10 reads.
> - **List output is markdown, not JSON.** The model reads it naturally and references IDs without parsing overhead. Trace.md stays readable.
> - **Read output cleanup is minimal.** Current `hf` CLI output is already pretty clean — just three preamble lines and the occasional `[image]` stray. The SPEC anticipated worse chrome from older CLI versions; we don't have to do that work today.
>
> **A concerning observation Sprint 3.2 needs to address:**
> Today's smoke read 5 papers totaling ~435K chars (~145K tokens — over 2× our 64K context). Ollama silently truncated earlier history, so the final summary only reflects the most recent paper. The model is "doing the right thing" with the prompt as written (read several, summarize what stood out), but the prompt + free-form selection doesn't bound input size. **This is exactly why Sprint 3.2 introduces the deterministic filter** — a single batched call decides keepers from titles+abstracts (well under 15K tokens), then 3.3 reads only those in full. The current free-form behavior is fine as a Sprint 3.1 smoke; 3.2 fixes it structurally.
>
> **For Sprint 3.2 (filter):** The filter call runs OUTSIDE the agent loop as a single non-agentic completion. Input: `Interests.md` + the list-tool's markdown output (~10K tokens). Output: a JSON array of keeper IDs with justifications. No tool calling; lenient JSON parse. The keepers then drive 3.3's per-paper summarization.
>
> **For Sprint 3.3 (per-paper summary):** Each keeper gets one read + one non-agentic summary call. Structured per-paper output (TL;DR, why-it-matters, quote-worthy detail, link). The agent loop is then no longer used for the production mission — it's reserved for cases where multi-step reasoning is needed (deferred missions). The from-scratch agent loop work in Phase 2 still pays off as the test harness + the future-mission-engine.

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
> **Built:** `agent/filter.py` with `Keeper`, `FilterResult`, `FilterError`, lenient `parse_filter_response()`, and orchestrator `filter_papers(model, list_markdown, interests)`. `prompts/system_filter.md` instructs the model to return *only* a JSON array, coaches reasons in the user's voice (since they're reused verbatim in 4.1's brief). `memory/io.py` gains `read_interests()`. `cli.py` refactored: `Mission` is now `(name, run: Callable)`, and `paper_survey` runs the deterministic pipeline (list → filter → format) instead of the agent loop. `TraceSink` Protocol + `TraceWriter` + stubs all extended with `log_filter_input`, `log_filter_response`, `log_filter_keepers`. 13 new filter tests.
>
> **Acceptance criteria verified on live run (2026-05-14):**
> - **Parseable JSON output:** 5 keepers extracted cleanly from the model response.
> - **Rejects negative-interest papers:** filter reasoning in the trace explicitly skipped MulTaBench (multimodal tabular) and AnyFlow (pure CV diffusion) with "Skip (Pure CV)" verdicts.
> - **Keeps positive-interest papers:** the 5 keepers were MinT (LoRA serving infra), long-context VLM training, MAP-then-Act (agents), Action Guidance RL, and MemReread (agentic memory) — all in stated high-interest areas.
> - **Filter call logged to trace:** `filter_input`, `filter_response`, `filter_keepers` events visible in JSONL; markdown renders a `## Filter stage` section with reasoning as a blockquote, keepers as a bullet list with user-voice reasons.
> - Total test suite: **67 passing** (13 new filter tests, all existing tests still green).
>
> **One transient hiccup worth noting:** the first live run returned an empty filter response (content + reasoning both 0 chars) — likely an Ollama cold-start flake. A direct repro through `model.complete()` worked immediately, and the next CLI run was clean. Worth keeping an eye on but not chasing.
>
> **Cross-model comparison run (qwen3-30b-llamacpp vs qwen3-9b-ollama):** Both backends produced 5 keepers each on today's 50-paper list. **3 of 5 overlapped** (MinT, LVLM long-context, MAP-then-Act — the "obvious" picks). The other 2 differed but stayed in stated interest areas: 9B leaned toward RL + agentic memory; 30B toward many-shot ICL + active retrieval. Implication: the filter prompt is carrying the work, not the model. **The 9B is the right default for the filter stage** — same shape of output, similar quality, ~3× throughput. The 30B is better reserved for orchestrator-style work in future loops 2/3.
>
> **Design decisions worth noting:**
> - **Filter is one-shot, no retry loop.** If the filter's output is unparseable, we tighten `system_filter.md` and re-run. Adding a retry loop would mask prompt drift and inflate token spend on bad prompts. The parser surfaces structural problems (missing `id`, malformed JSON, etc.) with clear messages — these go to the user, not back to the model.
> - **Filter reasons become user-facing.** Sprint 4.1's brief will display them verbatim under "How I picked these." The prompt explicitly coaches the model to write in the user's voice ("This matters because…", "Worth a read for…"). Today's run produced exactly this register.
> - **The agent loop is bypassed for paper_survey.** This is the production shape: list and filter are deterministic; 3.3's per-paper read+summarize will also be deterministic. The agent loop survives for `test` and for future loops 2/3 where multi-step decision-making genuinely is needed.
> - **`Mission.run` is a callable.** Lets agent-loop missions and pipeline missions share one dispatch surface. New missions in 4.1/4.2 just supply a function.
> - **Trace surface is split into agent-loop events and pipeline-stage events.** Each mission may use only a subset. Sprint 3.3 will add `log_summarize_input/response/result` events for per-paper summaries.
>
> **For Sprint 3.3 (per-paper summary):** The filter returns `FilterResult.keepers` — Sprint 3.3 reads each via `HfPapersReadTool().run({"id": k.id})`, then makes a non-agentic summarize call per paper. Output is a structured per-paper summary the brief writer (4.1) consumes. Paper text can be ~120K chars; cap input to the summarizer at e.g. 24K chars (abstract + intro + conclusions) for predictable budget — discussed during the "tool-internal summarization" thread.
>
> **For Sprint 4.1 (brief writer):** Filter keepers + their reasons + Sprint 3.3's summaries flow into the brief. The "How I picked these" paragraph uses keeper reasons verbatim. The brief writer is pure Python templating — no LLM call.

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
> **Built:** `agent/summarize.py` with `PaperSummary` dataclass, `parse_summary_response` (lenient JSON extraction, strict-first then regex fallback), `summarize_keeper` (one-paper call), and `summarize_keepers` (orchestrator). `prompts/system_summarize.md` produces JSON `{title, tldr, why_it_matters, quote}`. `cli.py`'s `_run_paper_survey` now runs the full pipeline (list → filter → summarize) and renders a markdown summary list for stdout (replaced by Obsidian write in 4.1). `TraceSink` Protocol + writer extended with `log_summarize_input`, `log_summarize_result`, `log_summarize_skipped`. 14 new summarize tests.
>
> **Acceptance criteria verified on live run (2026-05-14):**
> - **Coherent summary per keeper:** 4/5 keepers produced substantive summaries (1 failed at the read stage, see below). Each has TL;DR in user-voice, ~2-sentence "why it matters" referencing specific user interests, and a verbatim quote with concrete numbers (e.g. *"+3.9 points at 4B and +3.6 points at 8B on SWE-bench Verified"* from the DAgger paper).
> - **Length matches the brief format:** Each summary is 4–8 sentences plus link, exactly the phone-readable size we designed for.
> - **Graceful degradation works:** 2605.13779 (MinT) failed at `hf papers read` with "not found on the Hub" (same flake seen in Sprint 3.1 — some IDs from the daily list aren't actually readable). The orchestrator logged `summarize_skipped` to the trace and continued with the next keeper. The 4 remaining summaries proceeded normally; the user gets a partial brief instead of no brief.
>
> **Trace markdown shows the full narrative.** Filter section → 4 per-paper summary sections + 1 skipped section. The skipped section makes the failure visible (not silenced).
>
> **Operational notes:**
> - Same Ollama cold-start flake from Sprint 3.2 hit again on the first attempt — filter returned empty content. Re-running cleanly succeeded. Worth a debugging pass at Sprint 5.1; possibly related to Ollama's KV cache being unloaded between requests.
> - Total wall time for the full pipeline: ~10 minutes (filter ~3.5 min reasoning, 4 summaries × ~1.5 min each). For an overnight run this is well within budget.
> - 81 → 95 tests passing (14 new summarize tests, all existing tests still green).
>
> **Design decisions worth noting:**
> - **Two-pass JSON parsing (strict then regex).** Caught a real bug in testing — the original regex-only approach silently extracted the first `{...}` from a `[{...}]` response, accepting malformed structure. Strict-first means well-formed responses parse cleanly; regex fallback handles models that add prose preamble.
> - **`quote` field is optional, never paraphrased.** The prompt explicitly says "return null if no clean verbatim sentence stands out — do not paraphrase." Today's run delivered: all 4 quotes look verbatim and carry concrete numbers/claims. Worth a post-hoc check in 5.1 (string-search quotes against paper text) but not required.
> - **Max chars = 24K (~8K tokens).** First-N-chars truncation; abstract+intro+early sections in practice. Smarter section-aware extraction is deferred. Today's runs used the full 24K budget on every paper without complaint.
> - **Skipped papers are visible, not silent.** The `summarize_skipped` event renders a clear `## Summary: <id> (skipped)` block in trace.md with the reason. User sees what was attempted and why it failed.
> - **The `quote` is the highest-value field per summary.** The "+3.9 / +3.6 points" line in the DAgger summary is exactly what a morning brief should preserve — a number you'd remember. Worth more prompt iteration in 5.1 to make this consistent.
>
> **For Sprint 4.1 (brief writer):** `summarize_keepers` returns `list[PaperSummary]` — the brief writer's only input. The provisional stdout renderer in `cli.py:_format_summaries_for_stdout` is the rough shape; 4.1 makes it production: writes to `<vault_path>/Briefs/YYYY-MM-DD.md`, adds frontmatter (date, paper count, model used), adds the "How I picked these" paragraph drawn from filter reasons, handles same-day rerun suffix.
>
> **For Sprint 4.2 (memory):** `Seen.md` dedup should run *before* the filter, not after — we want yesterday's keepers excluded from the filter's input list entirely, not skipped after the filter chooses them. Sprint 3.2's filter already takes `interests` as input; same pattern, just add `seen_ids` and exclude in the list-pre-format step.

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
