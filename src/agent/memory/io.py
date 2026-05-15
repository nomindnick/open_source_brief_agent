"""Memory file I/O.

Read/write helpers for the markdown files under the project's top-level
``memory/`` directory. The ``memory/`` directory lives at the project
root, not under ``src/agent``. See SPEC § "Data Model".

Two pieces of state live here:

  - ``Seen.md`` — flat append-only ledger of paper IDs that have been
    surfaced in past briefs. One entry per line as ``<id>\\t<date>``.
    Used as a Python ``set`` to **deterministically filter** the daily
    paper list before the LLM filter call. Never loaded into context.

  - ``Reflections/YYYY-MM-DD.md`` — one file per run, written
    end-of-run by the reflection LLM call. The *most recent* file is
    injected into the next morning's filter prompt as
    ``{{recent_reflection}}``.
"""

from __future__ import annotations

import re
from pathlib import Path

DEFAULT_MEMORY_DIR = Path("memory")
SEEN_FILE = "Seen.md"
REFLECTIONS_DIR = "Reflections"

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


__all__ = [
    "read_interests",
    "read_seen_ids",
    "append_seen_ids",
    "write_reflection",
    "read_latest_reflection",
]
