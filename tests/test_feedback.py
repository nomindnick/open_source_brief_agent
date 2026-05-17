"""Tests for the user-feedback loop (Sprint 6.1).

Covers:
  - ``parse_brief_feedback`` — extracting filled entries from a brief.
  - ``feedback_dates`` / ``append_feedback`` / ``read_recent_feedback`` —
    the Feedback.md on-disk surface.
  - ``ingest_pending_feedback`` — scanning the vault and reconciling with
    Feedback.md (idempotency, lex-last variant per date, legacy briefs).
  - ``filter_papers`` — that ``recent_feedback`` reaches the prompt.

Synthetic data only; no LLM is invoked.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from agent.feedback import ingest_pending_feedback
from agent.filter import filter_papers
from agent.memory.io import (
    FeedbackEntry,
    append_feedback,
    feedback_dates,
    parse_brief_feedback,
    read_recent_feedback,
)
from agent.model.base import ModelResponse


# ── parse_brief_feedback ───────────────────────────────────────────────


def _brief_with_feedback(*entry_blocks: str) -> str:
    """Build a synthetic brief that has a Feedback section."""
    head = (
        "---\ndate: 2026-05-15\n---\n# Morning Brief — 2026-05-15\n\n"
        "## How I picked these\n- **x.y** — r\n\n---\n\n"
        "## A paper\n\n**TL;DR:** body\n\n[x.y](https://arxiv.org/abs/x.y)\n\n"
        "---\n\n## Feedback\n\n"
    )
    return head + "\n".join(entry_blocks)


def test_parse_returns_empty_when_no_feedback_section():
    brief = (
        "---\ndate: 2026-05-15\n---\n# Morning Brief\n\n"
        "## How I picked these\n- **a** — r\n\n## A paper\n\nbody\n"
    )
    assert parse_brief_feedback(brief) == []


def test_parse_returns_empty_when_all_entries_unfilled():
    brief = _brief_with_feedback(
        "### Title One (2605.001)\n- Signal: [ ]\n- Notes:\n\n",
        "### Title Two (2605.002)\n- Signal: [ ]\n- Notes:\n\n",
    )
    assert parse_brief_feedback(brief) == []


def test_parse_extracts_positive_signal_with_notes():
    brief = _brief_with_feedback(
        "### Title One (2605.001)\n- Signal: [+]\n- Notes: Want more on memory validity.\n\n",
    )
    entries = parse_brief_feedback(brief)
    assert entries == [
        FeedbackEntry(
            id="2605.001",
            title="Title One",
            signal="+",
            notes="Want more on memory validity.",
        )
    ]


def test_parse_extracts_negative_signal():
    brief = _brief_with_feedback(
        "### A (1)\n- Signal: [-]\n- Notes:\n\n",
    )
    entries = parse_brief_feedback(brief)
    assert len(entries) == 1
    assert entries[0].signal == "-"
    assert entries[0].notes == ""


def test_parse_keeps_notes_only_entry():
    brief = _brief_with_feedback(
        "### A (1)\n- Signal: [ ]\n- Notes: Mixed feelings.\n\n",
    )
    entries = parse_brief_feedback(brief)
    assert len(entries) == 1
    assert entries[0].signal is None
    assert entries[0].notes == "Mixed feelings."


def test_parse_skips_filled_and_unfilled_in_same_brief():
    brief = _brief_with_feedback(
        "### A (1)\n- Signal: [+]\n- Notes:\n\n",
        "### B (2)\n- Signal: [ ]\n- Notes:\n\n",  # unfilled — skip
        "### C (3)\n- Signal: [-]\n- Notes: nope.\n\n",
    )
    ids = [e.id for e in parse_brief_feedback(brief)]
    assert ids == ["1", "3"]


def test_parse_skips_heading_without_id():
    # User hand-typed a heading without parens — defensive parse skips it.
    brief = _brief_with_feedback(
        "### Title Without ID\n- Signal: [+]\n- Notes: lost\n\n",
        "### Real (real-id)\n- Signal: [+]\n- Notes: kept\n\n",
    )
    entries = parse_brief_feedback(brief)
    assert [e.id for e in entries] == ["real-id"]


def test_parse_handles_multiline_notes():
    brief = _brief_with_feedback(
        "### A (1)\n- Signal: [+]\n- Notes: first line\n  continued on second line\n\n",
        "### B (2)\n- Signal: [-]\n- Notes: short\n\n",
    )
    entries = parse_brief_feedback(brief)
    assert entries[0].notes.startswith("first line")
    assert "continued on second line" in entries[0].notes
    assert entries[1].notes == "short"


def test_parse_stops_at_next_h2():
    # Anything after the next ## isn't part of Feedback.
    brief = _brief_with_feedback(
        "### A (1)\n- Signal: [+]\n- Notes: in feedback\n\n",
    ) + "## Other Section\n\n### B (2)\n- Signal: [+]\n- Notes: NOT in feedback\n"
    entries = parse_brief_feedback(brief)
    assert [e.id for e in entries] == ["1"]


def test_parse_signal_with_whitespace_in_brackets_is_no_signal():
    brief = _brief_with_feedback(
        "### A (1)\n- Signal: [  ]\n- Notes: present\n\n",
    )
    entries = parse_brief_feedback(brief)
    assert entries[0].signal is None
    assert entries[0].notes == "present"


# ── feedback_dates / append_feedback / read_recent_feedback ───────────


def test_feedback_dates_empty_when_file_missing(tmp_path: Path):
    assert feedback_dates(memory_dir=tmp_path) == set()


def test_append_feedback_writes_header_and_entry(tmp_path: Path):
    append_feedback(
        "2026-05-15",
        [FeedbackEntry(id="x.y", title="Sample", signal="+", notes="hi")],
        memory_dir=tmp_path,
    )
    content = (tmp_path / "Feedback.md").read_text()
    assert content.startswith("# Feedback.md")
    assert "## 2026-05-15" in content
    assert "### Sample (x.y)" in content
    assert "- Signal: [+]" in content
    assert "- Notes: hi" in content


def test_append_feedback_noop_on_empty(tmp_path: Path):
    append_feedback("2026-05-15", [], memory_dir=tmp_path)
    assert not (tmp_path / "Feedback.md").exists()


def test_feedback_dates_round_trip(tmp_path: Path):
    append_feedback(
        "2026-05-14",
        [FeedbackEntry("a", "A", "+", "")],
        memory_dir=tmp_path,
    )
    append_feedback(
        "2026-05-15",
        [FeedbackEntry("b", "B", "-", "")],
        memory_dir=tmp_path,
    )
    assert feedback_dates(memory_dir=tmp_path) == {"2026-05-14", "2026-05-15"}


def test_read_recent_feedback_empty_when_missing(tmp_path: Path):
    assert read_recent_feedback(window=10, memory_dir=tmp_path) == ""


def test_read_recent_feedback_respects_window(tmp_path: Path):
    for d in ("2026-05-10", "2026-05-11", "2026-05-12", "2026-05-13"):
        append_feedback(
            d, [FeedbackEntry(d, f"T{d}", "+", "")], memory_dir=tmp_path
        )
    out = read_recent_feedback(window=2, memory_dir=tmp_path)
    # Only the two newest dates appear.
    assert "2026-05-13" in out
    assert "2026-05-12" in out
    assert "2026-05-11" not in out
    assert "2026-05-10" not in out


def test_read_recent_feedback_before_date_excludes_same_day(tmp_path: Path):
    append_feedback(
        "2026-05-14",
        [FeedbackEntry("a", "A", "+", "")],
        memory_dir=tmp_path,
    )
    append_feedback(
        "2026-05-15",
        [FeedbackEntry("b", "B", "+", "")],
        memory_dir=tmp_path,
    )
    out = read_recent_feedback(
        window=10, memory_dir=tmp_path, before_date="2026-05-15"
    )
    assert "2026-05-14" in out
    assert "2026-05-15" not in out


def test_read_recent_feedback_blocks_are_dated_in_order(tmp_path: Path):
    append_feedback(
        "2026-05-13",
        [FeedbackEntry("a", "A", "+", "")],
        memory_dir=tmp_path,
    )
    append_feedback(
        "2026-05-14",
        [FeedbackEntry("b", "B", "+", "")],
        memory_dir=tmp_path,
    )
    out = read_recent_feedback(window=10, memory_dir=tmp_path)
    # Both dates present; 2026-05-13 comes before 2026-05-14.
    i13 = out.find("2026-05-13")
    i14 = out.find("2026-05-14")
    assert 0 <= i13 < i14


# ── ingest_pending_feedback ───────────────────────────────────────────


def _write_brief(
    briefs_dir: Path, filename: str, feedback_section: str | None = None
) -> Path:
    """Helper: write a fake brief, optionally with a Feedback section."""
    briefs_dir.mkdir(parents=True, exist_ok=True)
    body = "# Morning Brief\n\n## How I picked these\n- **a** — r\n\n"
    if feedback_section:
        body += "---\n\n" + feedback_section
    p = briefs_dir / filename
    p.write_text(body, encoding="utf-8")
    return p


@dataclass
class _RecordingTrace:
    """Captures only the events this sprint emits."""

    ingest_calls: list[tuple[int, int, int]] | None = None
    inject_calls: list[tuple[int, int]] | None = None

    def __post_init__(self) -> None:
        if self.ingest_calls is None:
            self.ingest_calls = []
        if self.inject_calls is None:
            self.inject_calls = []

    def log_feedback_ingest(
        self, briefs_processed: int, new_entries: int, total_dates: int
    ) -> None:
        assert self.ingest_calls is not None
        self.ingest_calls.append((briefs_processed, new_entries, total_dates))

    def log_feedback_inject(self, window_size: int, chars: int) -> None:
        assert self.inject_calls is not None
        self.inject_calls.append((window_size, chars))

    # Everything else is a no-op for these tests.
    def __getattr__(self, _name: str):
        return lambda *a, **k: None


def test_ingest_no_briefs_dir_is_noop(tmp_path: Path):
    n = ingest_pending_feedback(
        vault_path=tmp_path,
        run_date="2026-05-15",
        memory_dir=tmp_path / "memory",
    )
    assert n == 0
    assert not (tmp_path / "memory" / "Feedback.md").exists()


def test_ingest_legacy_brief_without_feedback_section_is_noop(tmp_path: Path):
    briefs = tmp_path / "vault" / "Briefs"
    _write_brief(briefs, "2026-05-14.md")  # no Feedback section
    n = ingest_pending_feedback(
        vault_path=tmp_path / "vault",
        run_date="2026-05-15",
        memory_dir=tmp_path / "memory",
    )
    assert n == 0
    assert not (tmp_path / "memory" / "Feedback.md").exists()


def test_ingest_filled_brief_appends_block(tmp_path: Path):
    fb = (
        "## Feedback\n\n"
        "### A (1)\n- Signal: [+]\n- Notes: yes please\n\n"
        "### B (2)\n- Signal: [ ]\n- Notes:\n\n"
    )
    _write_brief(tmp_path / "vault" / "Briefs", "2026-05-14.md", fb)

    n = ingest_pending_feedback(
        vault_path=tmp_path / "vault",
        run_date="2026-05-15",
        memory_dir=tmp_path / "memory",
    )
    assert n == 1
    content = (tmp_path / "memory" / "Feedback.md").read_text()
    assert "## 2026-05-14" in content
    assert "### A (1)" in content
    assert "yes please" in content
    # The unfilled B was skipped.
    assert "### B (2)" not in content


def test_ingest_idempotent_on_already_ingested_date(tmp_path: Path):
    fb = "## Feedback\n\n### A (1)\n- Signal: [+]\n- Notes: ok\n\n"
    _write_brief(tmp_path / "vault" / "Briefs", "2026-05-14.md", fb)

    n1 = ingest_pending_feedback(
        vault_path=tmp_path / "vault",
        run_date="2026-05-15",
        memory_dir=tmp_path / "memory",
    )
    n2 = ingest_pending_feedback(
        vault_path=tmp_path / "vault",
        run_date="2026-05-15",
        memory_dir=tmp_path / "memory",
    )
    assert n1 == 1
    assert n2 == 0  # second pass is a no-op
    content = (tmp_path / "memory" / "Feedback.md").read_text()
    assert content.count("## 2026-05-14") == 1


def test_ingest_skips_today_and_future_briefs(tmp_path: Path):
    fb = "## Feedback\n\n### A (1)\n- Signal: [+]\n- Notes: ok\n\n"
    _write_brief(tmp_path / "vault" / "Briefs", "2026-05-15.md", fb)  # today
    _write_brief(tmp_path / "vault" / "Briefs", "2026-05-20.md", fb)  # future
    _write_brief(tmp_path / "vault" / "Briefs", "2026-05-14.md", fb)  # past

    n = ingest_pending_feedback(
        vault_path=tmp_path / "vault",
        run_date="2026-05-15",
        memory_dir=tmp_path / "memory",
    )
    assert n == 1
    content = (tmp_path / "memory" / "Feedback.md").read_text()
    assert "## 2026-05-14" in content
    assert "## 2026-05-15" not in content
    assert "## 2026-05-20" not in content


def test_ingest_picks_lex_last_variant_per_date(tmp_path: Path):
    # Same date, two variants. -run-3 should win over -run-2 over plain.
    briefs = tmp_path / "vault" / "Briefs"
    _write_brief(
        briefs,
        "2026-05-14.md",
        "## Feedback\n\n### A (1)\n- Signal: [+]\n- Notes: plain\n\n",
    )
    _write_brief(
        briefs,
        "2026-05-14-run-2.md",
        "## Feedback\n\n### A (1)\n- Signal: [+]\n- Notes: run-2\n\n",
    )
    _write_brief(
        briefs,
        "2026-05-14-run-3.md",
        "## Feedback\n\n### A (1)\n- Signal: [+]\n- Notes: run-3\n\n",
    )
    ingest_pending_feedback(
        vault_path=tmp_path / "vault",
        run_date="2026-05-15",
        memory_dir=tmp_path / "memory",
    )
    content = (tmp_path / "memory" / "Feedback.md").read_text()
    assert "run-3" in content
    assert "run-2" not in content
    assert "plain" not in content


def test_ingest_calls_trace_with_counts(tmp_path: Path):
    fb = "## Feedback\n\n### A (1)\n- Signal: [+]\n- Notes: ok\n\n"
    _write_brief(tmp_path / "vault" / "Briefs", "2026-05-14.md", fb)
    _write_brief(tmp_path / "vault" / "Briefs", "2026-05-13.md")  # legacy

    trace = _RecordingTrace()
    ingest_pending_feedback(
        vault_path=tmp_path / "vault",
        run_date="2026-05-15",
        memory_dir=tmp_path / "memory",
        trace=trace,
    )
    assert trace.ingest_calls == [(2, 1, 1)]
    # briefs_processed=2 (both attempted), new_entries=1, total_dates=1.


# ── filter_papers receives recent_feedback in the prompt ───────────────


class _CaptureModel:
    """Stub model that captures the most recent prompt sent."""

    def __init__(self) -> None:
        self.last_prompt: str | None = None

    def complete(self, messages):  # type: ignore[no-untyped-def]
        self.last_prompt = messages[-1]["content"]
        # Return a minimal well-formed filter response.
        return ModelResponse(
            content='[{"id": "2605.001", "reason": "test reason"}]',
            reasoning=None,
            raw='[{"id": "2605.001", "reason": "test reason"}]',
            usage={},
        )

    def close(self) -> None: ...


def test_filter_papers_injects_recent_feedback_into_prompt():
    model = _CaptureModel()
    list_md = "# HuggingFace Papers — 1 result(s)\n\n## 2605.001 (3 upvotes)\n**T**\n\nA.\n"
    fb = "## 2026-05-15\n\n### Sample (2605.001)\n- Signal: [+]\n- Notes: relevant\n"
    result = filter_papers(
        model=model,
        list_markdown=list_md,
        interests="- general interest in agents",
        recent_reflection="",
        recent_feedback=fb,
    )
    assert result.keepers[0].id == "2605.001"
    assert model.last_prompt is not None
    assert "## Recent reader feedback" in model.last_prompt
    assert "[+]" in model.last_prompt
    assert "relevant" in model.last_prompt


def test_filter_papers_empty_feedback_renders_cleanly():
    model = _CaptureModel()
    list_md = "# HuggingFace Papers — 1 result(s)\n\n## a (1 upvotes)\n**T**\n\nA.\n"
    filter_papers(
        model=model,
        list_markdown=list_md,
        interests="interest",
        recent_reflection="",
        recent_feedback="",
    )
    # Heading present, body empty — no leftover {{recent_feedback}}.
    assert model.last_prompt is not None
    assert "{{" not in model.last_prompt
    assert "## Recent reader feedback" in model.last_prompt
