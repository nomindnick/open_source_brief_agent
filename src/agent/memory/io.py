"""Memory file I/O.

Read/write helpers for the markdown files under the project's top-level
``memory/`` directory. The ``memory/`` directory lives at the project
root, not under ``src/agent``. See SPEC § "Data Model".

Three pieces of state live here:

  - ``Seen.md`` — flat append-only ledger of paper IDs that have been
    surfaced in past briefs. One entry per line as ``<id>\\t<date>``.
    Used as a Python ``set`` to **deterministically filter** the daily
    paper list before the LLM filter call. Never loaded into context.

  - ``Reflections/YYYY-MM-DD.md`` — one file per run, written
    end-of-run by the reflection LLM call. The *most recent* file is
    injected into the next morning's filter prompt as
    ``{{recent_reflection}}``.

  - ``Feedback.md`` — append-only log of user feedback on past briefs.
    Each ``## YYYY-MM-DD`` block holds filled-in ``### <Title> (<id>)``
    entries parsed from that brief's Feedback section. The last N
    dated blocks are injected into the next morning's filter prompt
    as ``{{recent_feedback}}``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MEMORY_DIR = Path("memory")
SEEN_FILE = "Seen.md"
REFLECTIONS_DIR = "Reflections"
FEEDBACK_FILE = "Feedback.md"

# A reflection file is named YYYY-MM-DD.md. The regex is used for
# "latest" lookup so we don't accidentally pick up a Welcome.md or
# similar.
_REFLECTION_NAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.md$")


def read_interests(memory_dir: Path | str = DEFAULT_MEMORY_DIR) -> str:
    """Return the contents of ``memory/Interests.md``.

    Raises ``FileNotFoundError`` if missing — Interests.md is user-curated
    and is checked into the repo with seed content, so this should always
    succeed in a properly configured project.
    """
    path = Path(memory_dir) / "Interests.md"
    return path.read_text(encoding="utf-8")


# ── Seen.md (paper-ID dedup ledger) ────────────────────────────────────


def read_seen_ids(
    memory_dir: Path | str = DEFAULT_MEMORY_DIR,
    *,
    before_date: str | None = None,
) -> set[str]:
    """Return the set of paper IDs in ``Seen.md``.

    Args:
        memory_dir: Directory containing ``Seen.md``.
        before_date: If provided, return only IDs whose recorded date is
            **strictly less than** this value. Pass today's date here to
            implement the SPEC's "exclude papers seen >= 1 day ago"
            rule: same-day reruns will not deduplicate (so you can iterate
            on prompts without surprises), but yesterday's keepers won't
            resurface tomorrow.

    Returns:
        A ``set[str]`` of paper IDs. Empty set if ``Seen.md`` doesn't
        exist yet — first-run behavior.
    """
    path = Path(memory_dir) / SEEN_FILE
    if not path.is_file():
        return set()
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Each line is "<id>\t<date>". Tolerate missing date columns
        # (older files, hand-edited content) by accepting either shape.
        parts = line.split(None, 1)  # any whitespace
        if not parts:
            continue
        pid = parts[0]
        recorded_date = parts[1].strip() if len(parts) > 1 else ""
        if before_date is not None and recorded_date and recorded_date >= before_date:
            continue
        seen.add(pid)
    return seen


def append_seen_ids(
    ids: list[str],
    date: str,
    memory_dir: Path | str = DEFAULT_MEMORY_DIR,
) -> None:
    """Append ``<id>\\t<date>`` lines to ``Seen.md`` for each id.

    Idempotent at the *line* level — re-running a same-day mission adds
    fresh lines with the same date. Dedup at read time handles this.

    The file is created if missing, with a tiny header comment so a
    hand-editor knows what the columns mean.
    """
    if not ids:
        return
    path = Path(memory_dir) / SEEN_FILE
    new_file = not path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        if new_file:
            f.write(
                "# Seen.md — paper IDs surfaced in past briefs.\n"
                "# Format: <id>\\t<date>. Used as a set for dedup; "
                "never loaded into model context.\n\n"
            )
        for pid in ids:
            f.write(f"{pid}\t{date}\n")


# ── Reflections ────────────────────────────────────────────────────────


def write_reflection(
    date: str,
    content: str,
    memory_dir: Path | str = DEFAULT_MEMORY_DIR,
) -> Path:
    """Write a reflection to ``memory/Reflections/<date>.md``.

    Same-day reruns overwrite. (Cross-day uniqueness is the property
    that matters; a same-day rerun's reflection is more recent and
    just as informative.)

    Returns the path written.
    """
    dir_path = Path(memory_dir) / REFLECTIONS_DIR
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / f"{date}.md"
    path.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")
    return path


def read_latest_reflection(
    memory_dir: Path | str = DEFAULT_MEMORY_DIR,
    *,
    before_date: str | None = None,
) -> str | None:
    """Return the most recent reflection's content, or None if none exist.

    Args:
        memory_dir: Directory containing ``Reflections/``.
        before_date: If provided, only consider reflections whose
            date is **strictly less than** this value. Pass today's
            date to avoid the same-day-rerun case where today's
            reflection (just written) would feed back into the same
            day's filter call.

    Returns:
        The reflection content, or None when no qualifying file exists.
    """
    dir_path = Path(memory_dir) / REFLECTIONS_DIR
    if not dir_path.is_dir():
        return None

    candidates: list[tuple[str, Path]] = []
    for p in dir_path.iterdir():
        m = _REFLECTION_NAME_RE.match(p.name)
        if not m:
            continue
        rdate = m.group(1)
        if before_date is not None and rdate >= before_date:
            continue
        candidates.append((rdate, p))

    if not candidates:
        return None
    # Lexicographic sort works for YYYY-MM-DD.
    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1].read_text(encoding="utf-8")


# ── Feedback.md (user reactions to past briefs) ───────────────────────


@dataclass(frozen=True)
class FeedbackEntry:
    """One filled-in feedback entry parsed from a brief.

    Attributes:
        id: arxiv-style paper id, taken from the heading's ``(<id>)``.
        title: Paper title from the heading. May be empty if the user
            hand-edited the brief and stripped it; not load-bearing.
        signal: ``"+"`` (more like this), ``"-"`` (less like this), or
            None (no signal given). Anything other than ``[+]`` / ``[-]``
            in the Signal line is treated as None.
        notes: Free-text body after ``Notes:``. Empty string when none.
    """

    id: str
    title: str
    signal: str | None
    notes: str


# Pattern: "### <Title> (<id>)" — the title text is everything up to the
# last "(<id>)" group, captured greedily.
_FEEDBACK_HEADING_RE = re.compile(
    r"^###\s+(?P<title>.*?)\s*\((?P<id>[^()\s]+)\)\s*$",
    re.MULTILINE,
)
# Find the "## Feedback" section start; we stop at the next H2 or EOF.
_FEEDBACK_SECTION_RE = re.compile(
    r"^##\s+Feedback\s*$",
    re.MULTILINE,
)
# Signal line — strict format. Empty brackets, blanks, or anything other
# than "+" / "-" inside the brackets means "no signal given."
_SIGNAL_LINE_RE = re.compile(
    r"^[-*]\s*Signal\s*:\s*\[\s*(?P<mark>[+\-]?)\s*\]\s*$",
    re.MULTILINE,
)
# Notes line — captures the rest of the entry up to the next ``### `` or
# end of section.
_NOTES_LINE_RE = re.compile(
    r"^[-*]\s*Notes\s*:\s*(?P<body>.*?)(?=^###\s|\Z)",
    re.MULTILINE | re.DOTALL,
)
# Match the ``## YYYY-MM-DD`` headings in Feedback.md.
_FEEDBACK_DATE_HEADING_RE = re.compile(
    r"^##\s+(\d{4}-\d{2}-\d{2})\s*$",
    re.MULTILINE,
)


def parse_brief_feedback(brief_text: str) -> list[FeedbackEntry]:
    """Extract filled feedback entries from a brief's Feedback section.

    Strategy:
      1. Locate ``## Feedback``. Return [] if absent.
      2. Take the section body up to the next H2 or end-of-text.
      3. For each ``### <Title> (<id>)`` sub-section, extract signal +
         notes. Skip entries with no id, or with both signal=None and
         empty notes (those are unfilled stubs).

    Defensive:
      - Heading without ``(<id>)``: skipped (user-typo case).
      - Missing Signal line: signal=None.
      - Missing Notes line: notes="".
      - Multiline notes content (user typed paragraphs): preserved up
        to the next ``### `` heading.

    Args:
        brief_text: Full content of a brief markdown file.

    Returns:
        Filled :class:`FeedbackEntry` list. Empty list when the brief
        has no Feedback section, or when every entry is unfilled.
    """
    section_match = _FEEDBACK_SECTION_RE.search(brief_text)
    if not section_match:
        return []

    section_start = section_match.end()
    # Body runs from the end of "## Feedback" to the next H2 or EOF.
    next_h2 = re.search(r"^##\s+", brief_text[section_start:], re.MULTILINE)
    section_end = (
        section_start + next_h2.start() if next_h2 else len(brief_text)
    )
    section_body = brief_text[section_start:section_end]

    headings = list(_FEEDBACK_HEADING_RE.finditer(section_body))
    if not headings:
        return []

    entries: list[FeedbackEntry] = []
    for i, h in enumerate(headings):
        body_start = h.end()
        body_end = (
            headings[i + 1].start() if i + 1 < len(headings) else len(section_body)
        )
        entry_body = section_body[body_start:body_end]

        title = h.group("title").strip()
        pid = h.group("id").strip()
        if not pid:
            continue

        signal_m = _SIGNAL_LINE_RE.search(entry_body)
        mark = signal_m.group("mark") if signal_m else ""
        signal: str | None = mark if mark in ("+", "-") else None

        notes_m = _NOTES_LINE_RE.search(entry_body)
        notes = notes_m.group("body").strip() if notes_m else ""

        # Unfilled stub — skip.
        if signal is None and not notes:
            continue

        entries.append(
            FeedbackEntry(id=pid, title=title, signal=signal, notes=notes)
        )

    return entries


def feedback_dates(memory_dir: Path | str = DEFAULT_MEMORY_DIR) -> set[str]:
    """Return the set of ``YYYY-MM-DD`` blocks already in ``Feedback.md``.

    Used for idempotency: if a brief's date is already a heading in
    ``Feedback.md``, we skip re-ingesting that brief (the user may have
    edited the brief further, but the original ingest stands; manual
    re-ingest = delete the heading).
    """
    path = Path(memory_dir) / FEEDBACK_FILE
    if not path.is_file():
        return set()
    return set(_FEEDBACK_DATE_HEADING_RE.findall(path.read_text(encoding="utf-8")))


def append_feedback(
    date: str,
    entries: list[FeedbackEntry],
    memory_dir: Path | str = DEFAULT_MEMORY_DIR,
) -> None:
    """Append a ``## <date>`` block of entries to ``Feedback.md``.

    No-op when ``entries`` is empty (don't pollute the file with empty
    dated blocks). Creates the file with a header comment on first write.
    Caller is responsible for the idempotency check via :func:`feedback_dates`
    — this function does not check.
    """
    if not entries:
        return
    path = Path(memory_dir) / FEEDBACK_FILE
    new_file = not path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        if new_file:
            f.write(
                "# Feedback.md — user reactions to past briefs.\n"
                "# Dated blocks parsed from each brief's Feedback section "
                "at run start.\n"
                "# `[+]` = more like this, `[-]` = less like this.\n\n"
            )
        f.write(f"## {date}\n\n")
        for e in entries:
            heading_title = e.title if e.title else "(untitled)"
            f.write(f"### {heading_title} ({e.id})\n")
            signal_str = f"[{e.signal}]" if e.signal in ("+", "-") else "[ ]"
            f.write(f"- Signal: {signal_str}\n")
            f.write(f"- Notes: {e.notes}\n\n")


def read_recent_feedback(
    window: int = 10,
    memory_dir: Path | str = DEFAULT_MEMORY_DIR,
    *,
    before_date: str | None = None,
) -> str:
    """Return the last ``window`` dated blocks from ``Feedback.md`` as one string.

    Args:
        window: Number of most-recent dated blocks to include. Defaults
            to 10 (~2 weeks accounting for weekends).
        memory_dir: Directory containing ``Feedback.md``.
        before_date: If provided, only consider blocks whose date is
            **strictly less than** this value. Mirrors the same-day-rerun
            rule used for ``Seen.md`` / Reflections — same-day reruns
            don't see feedback that was ingested earlier in the same day
            (very edge-case; mainly belt-and-suspenders).

    Returns:
        A single string of the form

            ## 2026-05-15

            ### Title (id)
            - Signal: [+]
            - Notes: ...

            ## 2026-05-14

            ...

        ready to be slotted into the filter prompt. Empty string when
        no qualifying blocks exist.
    """
    path = Path(memory_dir) / FEEDBACK_FILE
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")

    # Find each "## YYYY-MM-DD" heading and the span up to the next "## " or EOF.
    matches = list(_FEEDBACK_DATE_HEADING_RE.finditer(text))
    if not matches:
        return ""

    blocks: list[tuple[str, str]] = []  # (date, block_text)
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].rstrip() + "\n"
        date = m.group(1)
        if before_date is not None and date >= before_date:
            continue
        blocks.append((date, block))

    if not blocks:
        return ""

    # Sort by date (lex sort works for YYYY-MM-DD); take the last N.
    blocks.sort(key=lambda t: t[0])
    recent = blocks[-window:]
    return "\n".join(b for _, b in recent).rstrip() + "\n"


__all__ = [
    "FeedbackEntry",
    "read_interests",
    "read_seen_ids",
    "append_seen_ids",
    "write_reflection",
    "read_latest_reflection",
    "parse_brief_feedback",
    "feedback_dates",
    "append_feedback",
    "read_recent_feedback",
]
