"""``TraceWriter`` — JSONL + markdown trace logger.

Every run writes two files in parallel:

  * ``trace.jsonl`` — one event per line; the canonical structured record.
    Includes ``trace_schema_version`` and ``agent_version`` in a leading
    ``meta`` event so old traces remain interpretable after schema changes.

  * ``trace.md`` — a human-readable narrative meant to be skimmed on a
    phone via Obsidian. Reasoning renders as a blockquote *above* each
    turn's content (visually subordinate but visible).

Both files are opened in append mode with line buffering so a crash mid-run
still leaves a usable partial trace on disk.

The same date+mission running twice gets a numeric suffix on the directory
(``test``, ``test-1``, ``test-2``) — runs are cheap, accidental overwrites
are not.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, IO

# Bump if the JSONL event shape changes in a backwards-incompatible way.
TRACE_SCHEMA_VERSION = 1

# Tool results can be many KB (paper full text, etc). Markdown shows a
# preview to keep the file phone-readable; full content lives in JSONL.
MD_RESULT_PREVIEW_CHARS = 600


def _now_iso() -> str:
    """UTC timestamp in ISO-8601 with millisecond precision."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _agent_version() -> str:
    from agent import __version__

    return __version__


def _allocate_run_dir(traces_root: Path, date: str, mission: str) -> Path:
    """Pick a directory that does not already exist.

    First try ``traces/<date>/<mission>/``; if taken, try ``-1``, ``-2``, ...
    """
    base = traces_root / date / mission
    if not base.exists():
        base.mkdir(parents=True, exist_ok=True)
        return base
    for i in range(1, 1000):
        candidate = traces_root / date / f"{mission}-{i}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
    raise RuntimeError(
        f"Could not allocate a trace directory under {traces_root}/{date}/ — "
        "too many same-day reruns. Clean up old runs."
    )


def _truncate_for_md(text: str, limit: int = MD_RESULT_PREVIEW_CHARS) -> tuple[str, bool]:
    """Return (preview, was_truncated)."""
    if len(text) <= limit:
        return text, False
    return text[:limit] + "…", True


class TraceWriter:
    """Writes structured + markdown traces for a single agent run.

    The shape matches :class:`agent.loop.TraceSink` so the loop never
    needs to know which sink it's writing through.
    """

    def __init__(
        self,
        traces_root: Path | str,
        date: str,
        mission: str,
        *,
        model_profile: str | None = None,
        max_iter: int | None = None,
    ) -> None:
        """Open trace files under ``traces_root/<date>/<mission>/``.

        Args:
            traces_root: Directory under which to create the run dir.
            date: e.g. ``"2026-05-14"``. Caller decides UTC vs local —
                the SPEC writes briefs in local-date terms.
            mission: Mission name. Used as the leaf directory.
            model_profile: Optional name of the model profile, recorded
                in the trace header for later reference.
            max_iter: Optional iteration cap, recorded in the header.
        """
        self._run_dir = _allocate_run_dir(Path(traces_root), date, mission)
        self._jsonl_path = self._run_dir / "trace.jsonl"
        self._md_path = self._run_dir / "trace.md"
        # buffering=1 → line-buffered. A crash mid-run still leaves a
        # readable partial trace on disk.
        self._jsonl: IO[str] = self._jsonl_path.open("a", encoding="utf-8", buffering=1)
        self._md: IO[str] = self._md_path.open("a", encoding="utf-8", buffering=1)
        self._turn = 0
        self._closed = False

        # Header
        self._meta = {
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "agent_version": _agent_version(),
            "date": date,
            "mission": mission,
            "model_profile": model_profile,
            "max_iter": max_iter,
        }
        self._write_jsonl_event("meta", **self._meta)
        self._write_md_header()

    # ── public properties ──────────────────────────────────────────────

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def jsonl_path(self) -> Path:
        return self._jsonl_path

    @property
    def md_path(self) -> Path:
        return self._md_path

    # ── TraceSink surface ──────────────────────────────────────────────

    def log_model_turn(self, content: str, reasoning: str | None) -> None:
        self._turn += 1
        self._write_jsonl_event(
            "model_turn",
            turn=self._turn,
            content=content,
            reasoning=reasoning,
        )
        self._write_md_model_turn(content=content, reasoning=reasoning)

    def log_tool_call(self, name: str, input: dict[str, Any]) -> None:
        self._write_jsonl_event(
            "tool_call",
            turn=self._turn,
            name=name,
            input=input,
        )
        self._write_md(f"**→ Tool call:** `{name}({json.dumps(input, ensure_ascii=False)})`\n\n")

    def log_tool_result(self, name: str, result: str) -> None:
        # Full text in JSONL; truncated preview in markdown.
        self._write_jsonl_event(
            "tool_result",
            turn=self._turn,
            name=name,
            result=result,
        )
        preview, truncated = _truncate_for_md(result)
        suffix = (
            f"\n\n_({len(result)} chars; truncated — full result in trace.jsonl)_"
            if truncated
            else ""
        )
        # Fence the result in a code block so multi-line JSON / paper text
        # renders predictably in Obsidian.
        self._write_md(
            f"**← Tool result** (`{name}`):\n\n```\n{preview}\n```{suffix}\n\n"
        )

    def log_parse_error(self, reason: str) -> None:
        self._write_jsonl_event("parse_error", turn=self._turn, reason=reason)
        self._write_md(f"**⚠ Parse error:** {reason}\n\n")

    def log_final_answer(self, text: str) -> None:
        self._write_jsonl_event("final_answer", turn=self._turn, text=text)
        self._write_md(f"---\n\n## Final answer\n\n{text}\n")

    def log_iteration_cap(self, max_iter: int) -> None:
        self._write_jsonl_event("iteration_cap", turn=self._turn, max_iter=max_iter)
        self._write_md(
            f"---\n\n## Iteration cap hit\n\nRan {max_iter} iterations without a "
            f"final answer.\n"
        )

    def close(self) -> None:
        if self._closed:
            return
        self._write_jsonl_event("close", turn=self._turn)
        self._jsonl.close()
        self._md.close()
        self._closed = True

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ── internals ──────────────────────────────────────────────────────

    def _write_jsonl_event(self, event: str, **fields: Any) -> None:
        record = {"timestamp": _now_iso(), "event": event, **fields}
        self._jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_md(self, s: str) -> None:
        self._md.write(s)

    def _write_md_header(self) -> None:
        self._write_md(
            f"# Trace: {self._meta['mission']} — {self._meta['date']}\n\n"
            f"- **Model profile:** {self._meta['model_profile'] or '<unspecified>'}\n"
            f"- **Iteration cap:** {self._meta['max_iter'] or '<unspecified>'}\n"
            f"- **Started:** {_now_iso()}\n"
            f"- **Schema version:** {self._meta['trace_schema_version']}\n"
            f"- **Agent version:** {self._meta['agent_version']}\n\n"
            "---\n\n"
        )

    def _write_md_model_turn(self, content: str, reasoning: str | None) -> None:
        self._write_md(f"## Turn {self._turn}\n\n")
        if reasoning:
            # Render every line of reasoning as a blockquote, visually
            # subordinate but always visible (per Sprint 2.3 design choice).
            quoted = "\n".join(f"> {line}" if line else ">" for line in reasoning.splitlines())
            self._write_md(f"> **Reasoning**\n>\n{quoted}\n\n")
        # Fence the model's literal output so any XML it emitted renders
        # as a code block rather than being interpreted by markdown.
        self._write_md(f"```xml\n{content}\n```\n\n")


__all__ = ["TraceWriter", "TRACE_SCHEMA_VERSION"]
