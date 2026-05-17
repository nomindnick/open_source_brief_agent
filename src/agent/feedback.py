"""User-feedback ingest from prior briefs.

Run at the very start of the paper_survey pipeline. Scans the Obsidian
vault for previously-written briefs, picks up any that haven't been
ingested yet, parses their ``## Feedback`` sections, and appends filled
entries to ``memory/Feedback.md`` under a dated heading.

Cross-cutting design:
  - **Source of truth** is ``Feedback.md`` (dated blocks), not the brief.
    The brief is just the input surface — once a date is in Feedback.md,
    the brief can be edited freely without re-triggering ingest.
  - **Idempotent** at the date level via :func:`feedback_dates`. Manual
    re-ingest = remove the heading from Feedback.md.
  - **Defensive**: legacy briefs without a Feedback section parse to
    empty entry lists, which :func:`append_feedback` no-ops on. No
    spurious empty dated blocks land on disk.
  - **Same-day-rerun safe**: today's brief is never ingested into
    today's Feedback.md — only briefs whose date is strictly less than
    the run date.
"""

from __future__ import annotations

import re
from pathlib import Path

from agent.loop import TraceSink
from agent.memory.io import (
    DEFAULT_MEMORY_DIR,
    append_feedback,
    feedback_dates,
    parse_brief_feedback,
)

# Match brief filenames: "<date>.md" or "<date>-run-<n>.md".
# Captures the date and (optional) run number so we can pick the
# freshest variant per date.
_BRIEF_FILENAME_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})(?:-run-(?P<run>\d+))?\.md$"
)


def _canonical_brief_per_date(
    briefs_dir: Path,
) -> dict[str, Path]:
    """Group brief files by date; pick the highest ``-run-N`` per date.

    Plain ``<date>.md`` is treated as run 1 (the first / canonical write);
    ``-run-2`` / ``-run-3`` / … come after. Whichever has the highest run
    number wins — that's the freshest brief the user is likeliest to
    have annotated.

    Naive lex sort would *not* work here: ``-`` (0x2D) sorts before ``.``
    (0x2E), so ``"2026-05-14-run-9.md"`` < ``"2026-05-14.md"`` as
    strings. We parse the run explicitly to avoid that footgun.
    """
    grouped: dict[str, list[tuple[int, Path]]] = {}
    if not briefs_dir.is_dir():
        return {}
    for p in briefs_dir.iterdir():
        if not p.is_file():
            continue
        m = _BRIEF_FILENAME_RE.match(p.name)
        if not m:
            continue
        date = m.group("date")
        run = int(m.group("run")) if m.group("run") else 1
        grouped.setdefault(date, []).append((run, p))
    return {
        date: max(entries, key=lambda t: t[0])[1]
        for date, entries in grouped.items()
    }


def ingest_pending_feedback(
    vault_path: Path | str,
    run_date: str,
    *,
    memory_dir: Path | str = DEFAULT_MEMORY_DIR,
    trace: TraceSink | None = None,
) -> int:
    """Ingest unprocessed brief feedback into ``Feedback.md``.

    For each brief in ``<vault_path>/Briefs/`` whose date is strictly
    less than ``run_date`` and whose date is not already a heading in
    ``Feedback.md``: parse its Feedback section, append filled entries.

    Args:
        vault_path: Obsidian vault root. Briefs live at
            ``<vault_path>/Briefs/``.
        run_date: Today's run date as ``YYYY-MM-DD``.
        memory_dir: Memory directory containing ``Feedback.md``.
        trace: Optional sink for the ``feedback_ingest`` event.

    Returns:
        Number of dated blocks newly added to ``Feedback.md``.
    """
    briefs_dir = Path(vault_path) / "Briefs"
    canonical = _canonical_brief_per_date(briefs_dir)
    if not canonical:
        if trace is not None:
            trace.log_feedback_ingest(
                briefs_processed=0, new_entries=0, total_dates=0
            )
        return 0

    already = feedback_dates(memory_dir=memory_dir)
    processed = 0
    new_blocks = 0
    total_new_entries = 0

    for date in sorted(canonical):
        if date >= run_date:
            continue
        if date in already:
            continue
        brief_path = canonical[date]
        try:
            text = brief_path.read_text(encoding="utf-8")
        except OSError:
            # Don't let a single unreadable brief abort the whole run.
            continue
        processed += 1
        entries = parse_brief_feedback(text)
        if not entries:
            # Legacy brief or all-blank Feedback section — record nothing,
            # leave the date out of Feedback.md so a future edit of the
            # same brief (adding signals later) can still ingest.
            continue
        append_feedback(date, entries, memory_dir=memory_dir)
        new_blocks += 1
        total_new_entries += len(entries)

    if trace is not None:
        trace.log_feedback_ingest(
            briefs_processed=processed,
            new_entries=total_new_entries,
            total_dates=len(already) + new_blocks,
        )
    return new_blocks


__all__ = ["ingest_pending_feedback"]
