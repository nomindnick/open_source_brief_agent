# Overnight Research Agent

Local-first AI agent that surveys open-source / SLM research overnight, filters by personal interests, and writes a morning brief to an Obsidian vault. Built from scratch (no agent frameworks) as a learning project.

```
list 50 papers  →  filter to 5 keepers  →  read + summarize each  →  brief to Obsidian  →  reflection for tomorrow
```

Reference docs:

- **[planning_docs/SPEC.md](planning_docs/SPEC.md)** — design, architecture, scope, future considerations
- **[planning_docs/IMPLEMENTATION_PLAN.md](planning_docs/IMPLEMENTATION_PLAN.md)** — sprint-by-sprint history with decisions
- **[CLAUDE.md](CLAUDE.md)** — hard constraints (no frameworks, custom XML tool calling, file-only storage)
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** — known failure modes and fixes

## Hardware assumptions

Built and tested on a Framework Desktop with AMD Strix Halo (Radeon 8060S, gfx1151), 128GB unified memory, ~96GB allocated to GPU. The default config (`config.toml.example`) is sized for that:

- 30B Q8 model (~36GB weights) with 64K context KV cache fits comfortably.
- 9B Ollama model is the recommended default for filter/summarize/reflection stages.

Should run on smaller machines with smaller models. The minimum requirement is "a model big enough to follow XML tool-call format reliably." 7B+ instruct models trained recently usually qualify; the project was validated on Qwen3-30B (llama.cpp) and Qwen3.5-9B (Ollama).

## Install

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```sh
git clone <repo> open_source_brief_agent
cd open_source_brief_agent
uv sync
```

### HuggingFace CLI

Primary data source. Install once and authenticate:

```sh
uv tool install "huggingface_hub[cli]"
hf auth login              # paste a read-scope token from huggingface.co/settings/tokens
hf papers list --format json | head    # smoke test — should print JSON
```

### Inference backend (one of)

**llama.cpp.** Start `llama-server` with the model from `config.toml`'s `qwen3-30b-llamacpp` profile (or your equivalent):

```sh
llama-server \
  -m ~/models/qwen3-30b-a3b-q8/Qwen3-30B-A3B-UD-Q8_K_XL.gguf \
  -c 65536 \
  --host 127.0.0.1 --port 8080
```

The `-c` flag MUST match (or exceed) the profile's `context_length` in `config.toml`. Server allocates the KV cache at startup.

**Ollama.** Start the daemon and pull a model:

```sh
sudo systemctl start ollama    # or: ollama serve
ollama pull qwen3.5:9b         # or whatever model your config profile names
```

Ollama allocates KV cache on demand per request — no server restart needed when bumping `context_length` in `config.toml`.

## Configure

```sh
cp config.toml.example config.toml
cp .env.example .env
```

Edit `config.toml`:

| Field | What it does |
|---|---|
| `default_model` | Profile name (must match a `[models.<name>]` table). Used when no per-stage override or `--model` flag is given. |
| `vault_path` | Absolute path to your Obsidian vault. Briefs land in `<vault_path>/Briefs/<date>.md`. Must exist and be writable. |
| `iteration_cap` | Hard cap on agent-loop iterations (used by `test` mission and future loops; daily-papers pipeline is non-agentic so doesn't use it). |
| `[models.<name>]` | Per-model profile bundle: backend, model name, base URL, temperature, max tokens, context length, supports_thinking flag. |
| `[paper_survey]` | Mission tunables: `list_abstract_chars`, `summary_max_chars`, `hf_subprocess_timeout_s`. Sensible defaults; see comments in the example. |
| `[paper_survey.models]` | Optional per-stage model overrides (`filter` / `summarize` / `reflection`). Defaults to `default_model` if unset. |

Edit `memory/Interests.md` (committed to the repo with sample content). This is the highest-leverage knob you have on what the brief surfaces. Edit freely between runs.

## Run

```sh
# Default: today's papers, write brief to Obsidian vault, write reflection
uv run python -m agent --mission paper_survey

# Iterate on prompt without polluting the vault or memory
uv run python -m agent --mission paper_survey --dry-run

# Replay a specific day (or simulate "tomorrow")
uv run python -m agent --mission paper_survey --date 2026-05-15

# Override the model for the whole run (sets every stage to this profile)
uv run python -m agent --mission paper_survey --model qwen3-30b-llamacpp

# Smoke test the agent loop (no LLM call to HuggingFace, just echo tool)
uv run python -m agent --mission test
```

A typical run takes ~15 minutes on a 9B Ollama model: filter (~3 min) + 5 summaries × ~2 min each. Long enough that you'll want to run it once before sleep, not at the breakfast table.

### Schedule overnight runs (systemd timer)

Linux only. Generates and installs systemd **user units** so the agent fires at a fixed time daily:

```sh
./scripts/systemd/install.sh           # default: 01:00 daily
./scripts/systemd/install.sh 03:30     # custom HH:MM
./scripts/systemd/install.sh --uninstall
```

The script:

- Generates `brief-agent.service` and `brief-agent.timer` under `~/.config/systemd/user/` with the right paths for your machine.
- Validates the unit files via `systemd-analyze --user verify`.
- Runs `daemon-reload` and `enable --now` so the timer starts immediately.
- Warns if Ollama isn't enabled (so it'd be down at fire time) or if `config.toml` is missing.
- Prints the next firing time on success.

**One manual step the script can't do for you:**

```sh
sudo loginctl enable-linger $USER
```

Without this, user services stop when you log out, and the overnight run won't fire if you've logged out of your desktop. The script reminds you if you haven't run this.

**Test before bed:**

```sh
systemctl --user start brief-agent.service           # fires the real 15-min run
journalctl --user -u brief-agent.service -f          # watch progress
systemctl --user list-timers brief-agent.timer       # confirm next firing
```

### Where output lands

- **Brief:** `<vault_path>/Briefs/<date>.md`. Same-day reruns become `<date>-run-2.md`, `-run-3.md`, etc. — no overwrites.
- **Trace:** `traces/<date>/<mission>/trace.{jsonl,md}` (in repo). Same-day reruns get a `-1`, `-2` suffix on the directory. JSONL is canonical structured data; markdown is the phone-readable narrative.
- **Reflection:** `memory/Reflections/<date>.md` (in repo, committed). Used in tomorrow's filter prompt.
- **Seen IDs:** `memory/Seen.md`. Tracked, used as a set for dedup.

### Reading a brief

The brief opens with frontmatter (date, paper count, model used), then a `## How I picked these` section — verbatim filter-stage justifications, including any papers that got picked but failed to summarize (marked as skipped). Then one section per summarized paper:

- **TL;DR** — what the paper is about
- **Why it matters** — post-read refinement of the filter reason, addressed to you directly
- **Quote** — one verbatim sentence from the paper if a memorable line existed
- **Link** — clickable arxiv URL

If a brief feels off, the trace is the place to look. Reasoning blockquotes show what the model was deliberating about; you can see exactly why a paper was picked, summarized one way vs. another, etc. Use traces to iterate on prompts.

### Recovering from a bad run

- **A paper got into `Seen.md` you wanted to re-surface tomorrow:** edit `memory/Seen.md` and delete the row. It's plain markdown, one entry per line as `<id>\t<date>`.
- **The brief picked weirdly:** edit `memory/Interests.md` to clarify. Tomorrow's run will see the change. If you want to also revisit today, rerun with `--dry-run` first to confirm picks, then drop the `--dry-run` to write.
- **Yesterday's reflection feels wrong:** edit `memory/Reflections/<date>.md`. The next run will see your edits.
- **Run errored out partway:** the trace under `traces/<date>/<mission>/` shows where. Re-run; no destructive side effects until the brief writes. `Seen.md` and reflections are only written after a successful brief.

## Development

```sh
uv run pytest           # 108+ tests; fast (~0.1s)
just test               # same, via justfile
just dry-run            # full pipeline without writing to vault
just run                # full pipeline, writes to vault
```

### Project layout (one-paragraph version)

The agent code is under `src/agent/`. `model/` is the swappable backend interface (llama.cpp, Ollama). `tools/` are the `Tool` subclasses the agent loop can call (echo for testing, hf_papers for the real mission). `filter.py`, `summarize.py`, `reflect.py` are the non-agentic stage orchestrators — each one a single LLM call wrapped in lenient parsing. `loop.py` is the from-scratch agent loop (used by the `test` mission and future loops). `brief/writer.py` renders summaries to Obsidian markdown. `memory/io.py` reads/writes `Interests.md`, `Seen.md`, `Reflections/`. `trace/writer.py` emits the JSONL + MD trace. `cli.py` dispatches missions. Full detail in [SPEC.md](planning_docs/SPEC.md).

## Status

MVP. The daily-papers pipeline runs end-to-end and produces a brief in your vault that syncs to your phone. Loops 2 (interest-tailored research) and 3 (project-aware research) are designed-for-not-implemented; see SPEC § Future Considerations.
