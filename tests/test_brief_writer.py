"""Tests for the brief writer.

The writer is pure templating + file I/O, so unit tests are most of the
coverage. The Sprint 4.1 live run verifies the brief actually shows up
in Obsidian on the phone.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.brief.writer import (
    check_vault_writable,
    render_brief,
    write_brief,
)
from agent.filter import Keeper
from agent.summarize import PaperSummary


def _sample_keeper(id: str = "2509.05591", reason: str = "matches local-inference") -> Keeper:
    return Keeper(id=id, reason=reason)


def _sample_summary(
    id: str = "2509.05591",
    title: str = "Test Paper",
    quote: str | None = "A verbatim sentence.",
) -> PaperSummary:
    return PaperSummary(
        id=id,
        title=title,
        tldr="This is the punchline.",
        why_it_matters="Two sentences. Concrete refinement.",
        quote=quote,
        link=f"https://arxiv.org/abs/{id}",
        filter_reason="pre-read reason",
    )


# ── render_brief (pure function) ───────────────────────────────────────


def test_render_includes_frontmatter():
    md = render_brief(
        date="2026-05-15",
        summaries=[_sample_summary()],
        keepers=[_sample_keeper()],
        model_profile="qwen3-9b-ollama",
        papers_total=50,
    )
    assert md.startswith("---\n")
    assert "date: 2026-05-15" in md
    assert "model: qwen3-9b-ollama" in md
    assert "papers_total: 50" in md
    assert "keepers: 1" in md
    assert "summarized: 1" in md


def test_render_how_i_picked_uses_keeper_reasons_verbatim():
    keepers = [
        _sample_keeper("a", "reason A"),
        _sample_keeper("b", "reason B"),
    ]
    md = render_brief(
        date="2026-05-15",
        summaries=[_sample_summary("a"), _sample_summary("b")],
        keepers=keepers,
        model_profile="x",
        papers_total=50,
    )
    assert "## How I picked these" in md
    assert "**a** — reason A" in md
    assert "**b** — reason B" in md
    assert "skipped" not in md.lower()


def test_render_marks_skipped_papers_in_how_i_picked():
    keepers = [
        _sample_keeper("a", "reason A"),
        _sample_keeper("b", "reason B"),
    ]
    md = render_brief(
        date="2026-05-15",
        summaries=[_sample_summary("a")],  # b is missing → skipped
        keepers=keepers,
        model_profile="x",
        papers_total=50,
    )
    assert "**a** — reason A" in md
    assert "skipped" in md
    # The skipped marker is on the right keeper.
    line_b = next(line for line in md.splitlines() if line.startswith("- **b**"))
    assert "skipped" in line_b
    line_a = next(line for line in md.splitlines() if line.startswith("- **a**"))
    assert "skipped" not in line_a


def test_render_per_paper_section_has_h2_tldr_why_quote_link():
    summary = _sample_summary(
        id="x.y",
        title="The Title",
        quote="A memorable line.",
    )
    md = render_brief(
        date="2026-05-15",
        summaries=[summary],
        keepers=[_sample_keeper("x.y", "r")],
        model_profile="m",
        papers_total=1,
    )
    assert "## The Title" in md
    assert "**TL;DR:**" in md
    assert "**Why it matters:**" in md
    assert "> A memorable line." in md
    assert "[x.y](https://arxiv.org/abs/x.y)" in md


def test_render_omits_quote_blockquote_when_null():
    summary = _sample_summary(quote=None)
    md = render_brief(
        date="2026-05-15",
        summaries=[summary],
        keepers=[_sample_keeper(summary.id)],
        model_profile="m",
        papers_total=1,
    )
    # No blockquote leading-arrow line in the rendered output.
    assert "\n> " not in md
    # Other fields still present.
    assert "**TL;DR:**" in md


def test_render_empty_keepers_says_nothing_met_bar():
    md = render_brief(
        date="2026-05-15",
        summaries=[],
        keepers=[],
        model_profile="m",
        papers_total=42,
    )
    assert "Nothing on today's list met the bar" in md


# ── write_brief (file I/O) ────────────────────────────────────────────


def test_write_brief_creates_briefs_dir_and_file(tmp_path: Path):
    path = write_brief(
        vault_path=tmp_path,
        date="2026-05-15",
        summaries=[_sample_summary()],
        keepers=[_sample_keeper()],
        model_profile="m",
        papers_total=10,
    )
    assert path == tmp_path / "Briefs" / "2026-05-15.md"
    assert path.is_file()
    assert "Morning Brief" in path.read_text()


def test_write_brief_same_day_rerun_gets_suffix(tmp_path: Path):
    args = dict(
        vault_path=tmp_path,
        date="2026-05-15",
        summaries=[_sample_summary()],
        keepers=[_sample_keeper()],
        model_profile="m",
        papers_total=10,
    )
    a = write_brief(**args)
    b = write_brief(**args)
    c = write_brief(**args)
    assert a.name == "2026-05-15.md"
    assert b.name == "2026-05-15-run-2.md"
    assert c.name == "2026-05-15-run-3.md"


def test_write_brief_does_not_leave_temp_files_on_success(tmp_path: Path):
    write_brief(
        vault_path=tmp_path,
        date="2026-05-15",
        summaries=[_sample_summary()],
        keepers=[_sample_keeper()],
        model_profile="m",
        papers_total=10,
    )
    leftover = [p.name for p in (tmp_path / "Briefs").iterdir() if p.name.startswith(".")]
    assert leftover == []


# ── check_vault_writable ──────────────────────────────────────────────


def test_check_vault_writable_passes_for_real_dir(tmp_path: Path):
    check_vault_writable(tmp_path)  # no exception


def test_check_vault_writable_raises_on_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="vault_path"):
        check_vault_writable(tmp_path / "does-not-exist")


def test_check_vault_writable_raises_on_file_not_dir(tmp_path: Path):
    f = tmp_path / "vault"
    f.write_text("oops, file not dir")
    with pytest.raises(NotADirectoryError):
        check_vault_writable(f)


def test_check_vault_writable_raises_on_readonly(tmp_path: Path):
    ro = tmp_path / "readonly"
    ro.mkdir()
    ro.chmod(0o555)
    try:
        with pytest.raises(PermissionError):
            check_vault_writable(ro)
    finally:
        # Restore write so pytest can clean up.
        ro.chmod(0o755)
