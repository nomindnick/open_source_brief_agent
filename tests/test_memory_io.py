"""Tests for memory I/O: Seen.md ledger + Reflections directory.

Seen.md is the dedup ledger — used as a set lookup, never loaded into
context. Reflections are one-file-per-day; the most recent strictly-
before-today is injected into the next morning's filter prompt.
"""

from __future__ import annotations

from pathlib import Path

from agent.filter import exclude_seen_papers
from agent.memory.io import (
    DEFAULT_MEMORY_DIR,
    append_seen_ids,
    read_latest_reflection,
    read_seen_ids,
    write_reflection,
)


# ── Seen.md round-trip ─────────────────────────────────────────────────


def test_read_seen_ids_empty_when_file_missing(tmp_path: Path):
    assert read_seen_ids(memory_dir=tmp_path) == set()


def test_append_and_read_round_trip(tmp_path: Path):
    append_seen_ids(["a", "b"], date="2026-05-14", memory_dir=tmp_path)
    append_seen_ids(["c"], date="2026-05-15", memory_dir=tmp_path)
    assert read_seen_ids(memory_dir=tmp_path) == {"a", "b", "c"}


def test_before_date_excludes_same_day_entries(tmp_path: Path):
    append_seen_ids(["yesterday-id"], date="2026-05-14", memory_dir=tmp_path)
    append_seen_ids(["today-id"], date="2026-05-15", memory_dir=tmp_path)
    # When we ask "what was seen BEFORE today (2026-05-15)?",
    # today's entry shouldn't appear — same-day reruns must not dedupe.
    assert read_seen_ids(memory_dir=tmp_path, before_date="2026-05-15") == {
        "yesterday-id"
    }


def test_append_creates_file_with_header(tmp_path: Path):
    append_seen_ids(["x"], date="2026-05-14", memory_dir=tmp_path)
    content = (tmp_path / "Seen.md").read_text()
    assert content.startswith("# ")
    assert "x\t2026-05-14" in content


def test_append_empty_list_is_noop(tmp_path: Path):
    append_seen_ids([], date="2026-05-14", memory_dir=tmp_path)
    assert not (tmp_path / "Seen.md").exists()


def test_read_seen_skips_blank_lines_and_header(tmp_path: Path):
    p = tmp_path / "Seen.md"
    p.write_text(
        "# header comment\n"
        "\n"
        "a\t2026-05-14\n"
        "  \n"
        "b\t2026-05-15\n"
    )
    assert read_seen_ids(memory_dir=tmp_path) == {"a", "b"}


# ── exclude_seen_papers (the pre-filter step) ─────────────────────────


def _list_md(*ids: str) -> str:
    lines = [f"# HuggingFace Papers — {len(ids)} result(s)\n"]
    for i in ids:
        lines.append(f"## {i} (5 upvotes)\n**Title for {i}**\n\nAbstract for {i}.\n")
    return "\n".join(lines)


def test_exclude_seen_papers_removes_matching_sections():
    md = _list_md("a", "b", "c")
    out, dropped = exclude_seen_papers(md, {"b"})
    assert dropped == 1
    assert "## b " not in out
    assert "## a " in out
    assert "## c " in out


def test_exclude_seen_papers_updates_header_count():
    md = _list_md("a", "b", "c")
    out, _ = exclude_seen_papers(md, {"a", "b"})
    assert "HuggingFace Papers — 1 result(s)" in out


def test_exclude_seen_papers_no_seen_is_noop():
    md = _list_md("a", "b")
    out, dropped = exclude_seen_papers(md, set())
    assert dropped == 0
    assert out == md


def test_exclude_seen_papers_handles_id_with_no_match():
    md = _list_md("a", "b")
    out, dropped = exclude_seen_papers(md, {"z"})  # nothing matches
    assert dropped == 0
    assert "## a " in out and "## b " in out


# ── Reflections ────────────────────────────────────────────────────────


def test_write_and_read_latest_reflection(tmp_path: Path):
    write_reflection("2026-05-14", "Old reflection.", memory_dir=tmp_path)
    write_reflection("2026-05-15", "Newer reflection.", memory_dir=tmp_path)
    assert "Newer" in (read_latest_reflection(memory_dir=tmp_path) or "")


def test_read_latest_reflection_none_when_dir_missing(tmp_path: Path):
    assert read_latest_reflection(memory_dir=tmp_path) is None


def test_read_latest_reflection_ignores_non_date_files(tmp_path: Path):
    refl_dir = tmp_path / "Reflections"
    refl_dir.mkdir()
    (refl_dir / "notes.md").write_text("not a reflection")
    (refl_dir / "2026-05-14.md").write_text("real reflection")
    assert "real reflection" in (read_latest_reflection(memory_dir=tmp_path) or "")


def test_read_latest_reflection_before_date_excludes_today(tmp_path: Path):
    write_reflection("2026-05-14", "yesterday", memory_dir=tmp_path)
    write_reflection("2026-05-15", "today", memory_dir=tmp_path)
    latest = read_latest_reflection(memory_dir=tmp_path, before_date="2026-05-15")
    assert latest is not None and "yesterday" in latest
