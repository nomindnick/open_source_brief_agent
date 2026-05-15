# Troubleshooting

Failure modes hit during MVP development and how to fix them. Roughly ordered from "most likely you'll see this on the first run" to "rare edge cases."

## "Config file not found: config.toml"

You haven't copied the example yet.

```sh
cp config.toml.example config.toml
```

Edit `vault_path` to your real Obsidian vault path and you're good.

## "Obsidian vault path does not exist"

The pre-flight check fires *before* the 15-minute pipeline runs (that's the point). Two cases:

- **Typo in `config.toml`.** Open it, fix `vault_path`. Use an absolute path.
- **Vault doesn't exist yet.** Open Obsidian and create the vault folder, or `mkdir -p` it manually. The `Briefs/` subdirectory gets created automatically on first write.

## "Filter failed: Filter response did not contain a JSON array" (empty response)

Root cause is the model burning its entire output budget on reasoning and producing zero content tokens. You'll see `completion_tokens` equal to `max_tokens` in the `filter_response` event in `trace.jsonl`.

Fixes, in order of cheapness:

1. **Verify `max_tokens` is at least 16384** in your model profile (`config.toml`). Default `config.toml.example` has 16384 since Sprint 4.1 made this the floor.
2. **Tighten `prompts/system_filter.md`.** The "Be efficient with your reasoning" paragraph is the load-bearing part. If a model is being verbose, strengthen the instruction.
3. **Try a different model profile.** Different reasoning-models budget thinking differently. The 9B Ollama (`qwen3-9b-ollama`) and 30B llama.cpp (`qwen3-30b-llamacpp`) both work; some smaller models won't.

## "Empty filter response on the first try, works on retry"

We've seen this on Ollama as a cold-start flake. Direct repro via `model.complete()` fails, then succeeds. If it happens, just retry the run. If it happens *repeatedly*, the root cause is almost certainly the max_tokens issue above (see `completion_tokens` in the trace) — Ollama's cold start has been a red herring.

## "ERROR: `hf papers read 2509.XXXXX` failed: Paper not found on the Hub"

Some IDs the daily-list returns aren't actually readable. This is a real upstream condition, not a bug in the agent. The orchestrator handles it gracefully:

- Paper gets a `summarize_skipped` event in the trace.
- The "How I picked these" section in the brief marks that paper as `_(read failed; skipped from summaries)_`.
- The remaining papers continue normally — partial brief is better than no brief.

If this happens for *every* read, then the `hf` CLI auth is broken. Run `hf auth whoami` to check.

## "agent: hit iteration cap (3) before producing a final answer"

The `test` mission uses an explicit cap of 3 (or whatever `--max-iter` you pass). The `paper_survey` mission doesn't use the agent loop, so this won't fire for it.

If you're seeing it on `test` runs, your model isn't following the XML format reliably. Try a different model or check `prompts/_tool_calling_format.md` for clarity.

## "llama-server returned 500 / context size mismatch"

Most likely cause: you bumped `context_length` in `config.toml` but didn't restart `llama-server` with a matching `-c` flag. Restart it:

```sh
llama-server -m /path/to/model.gguf -c 65536 --host 127.0.0.1 --port 8080
```

The `-c` value must be ≥ the profile's `context_length`.

Ollama doesn't have this issue — it allocates KV cache per request based on the `num_ctx` we send (= `context_length` from the profile).

## "ModelBackendError: connection refused"

Server isn't running, or it's on a different port than the profile expects.

```sh
# For llama.cpp:
ps aux | grep llama-server
curl -s http://127.0.0.1:8080/v1/models

# For Ollama:
sudo systemctl status ollama
curl -s http://127.0.0.1:11434/api/tags
```

Match the URLs in your profile's `base_url`.

## "Brief written but Obsidian doesn't show it"

The file is on disk; Obsidian just needs to re-index. Open Obsidian, navigate to the `Briefs/` folder; the new file should appear. If it doesn't:

- Confirm `<vault_path>/Briefs/<date>.md` exists on disk (`ls $vault_path/Briefs/`).
- Confirm Obsidian's vault root matches `vault_path` in `config.toml`.

If you're using Obsidian Sync to your phone, the file appears once the desktop client syncs it up — give it a minute.

## "Brief picked a paper that's not actually relevant to my interests"

`memory/Interests.md` is the highest-leverage knob you have. Edit it to be more specific about what you do/don't want. The filter and reflection both read it.

The reflection from the previous day also feeds into today's filter. If you've been picking a category you regret, write that explicitly into the reflection file (`memory/Reflections/<yesterday>.md`): "Lean away from X tomorrow." The model will see it.

## "I want to undo today's run"

- Delete `<vault_path>/Briefs/<today>.md` (or the suffixed `-run-N.md` if multiple).
- Open `memory/Seen.md` and remove the rows dated today.
- Delete `memory/Reflections/<today>.md`.
- Optionally delete `traces/<today>/paper_survey/` if you don't want the trace around.

Then re-run.

## "Tests are failing"

```sh
uv run pytest -v
```

If a specific test fails, read the failure. If everything fails, you probably haven't `uv sync`'d or your Python is wrong (need 3.12). The pinned `.python-version` should handle this; if it doesn't, `uv python pin 3.12`.

## "Trace files are taking up too much space"

`.gitignore` already excludes `traces/`. They're for local-dev iteration. If you don't need history, just `rm -rf traces/`. If you want to keep recent ones for prompt diffing, `find traces/ -mtime +30 -type d -exec rm -rf {} +` archives anything older than 30 days.

---

If you hit a failure not listed here, the trace's `.md` is the best place to start. It shows the full conversation the agent had with itself and its tools — usually the answer is visible there.
