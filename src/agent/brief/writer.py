"""Brief renderer + Obsidian vault writer.

Pure markdown templating — no LLM calls. Takes the structured outputs
of the filter (:class:`Keeper`) and summarize (:class:`PaperSummary`)
stages and emits a single markdown file into ``<vault_path>/Briefs/``.

Same-day reruns get a ``-run-N`` suffix on the filename so accidental
overwrites are impossible. The write is atomic (temp file + rename) so
a crash mid-write doesn't leave a half-written brief in the vault.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from agent.filter import Keeper
from agent.summarize import PaperSummary


def _allocate_brief_path(briefs_dir: Path, date: str) -> Path:
    """Return a non-existent path under ``briefs_dir`` for today's brief.

    First choice: ``<date>.md``. If taken, append ``-run-2``, ``-run-3``,
    etc. Runs are cheap; overwriting yesterday's reading would not be.
    """
    candidate = briefs_dir / f"{date}.md"
    if not candidate.exists():
        return candidate
    for i in range(2, 1000):
        candidate = briefs_dir / f"{date}-run-{i}.md"
        if not candidate.exists():
            return candidate
    raise RuntimeError(
        f"Could not allocate a brief filename under {briefs_dir} "
        f"for date {date} — too many same-day reruns."
    )


def _atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically.

    Writes to a temp file in the same directory, then renames into place.
    On POSIX the rename is atomic, so a crash mid-write can't leave a
    partial file under ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_name, path)
    except Exception:
        # Best-effort cleanup; ignore failures so we don't mask the
        # original exception.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def render_brief(
    date: str,
    summaries: list[PaperSummary],
    keepers: list[Keeper],
    model_profile: str,
    papers_total: int,
) -> str:
    """Render the brief as a markdown string. Pure function; no I/O."""
    summarized_ids: set[str] = {s.id for s in summaries}

    # Frontmatter — Obsidian parses this natively. Lets the user query
    # briefs by date, keeper count, or model.
    fm_lines = [
        "---",
        f"date: {date}",
        "mission: paper_survey",
        f"papers_total: {papers_total}",
        f"keepers: {len(keepers)}",
        f"summarized: {len(summaries)}",
        f"model: {model_profile}",
        "---",
        "",
    ]
    header = f"# Morning Brief — {date}\n"
    summary_line = (
        f"{len(summaries)} of {len(keepers)} keepers summarized from "
        f"today's {papers_total} papers."
    )

    # "How I picked these" — verbatim from filter reasons, with skipped
    # papers flagged inline so the reader sees what was attempted.
    how_lines = ["## How I picked these", ""]
    if not keepers:
        how_lines.append("_Nothing on today's list met the bar._")
    else:
        for k in keepers:
            marker = (
                ""
                if k.id in summarized_ids
                else " _(read failed; skipped from summaries)_"
            )
            how_lines.append(f"- **{k.id}** — {k.reason}{marker}")
    how_block = "\n".join(how_lines)

    # Per-paper sections. Each summary gets its own H2 with the paper's
    # actual title, then TL;DR, why-it-matters, optional verbatim quote,
    # and a link.
    if not summaries:
        body = "_No papers survived the summarize stage._\n"
    else:
        sections: list[str] = []
        for s in summaries:
            section_lines = [
                f"## {s.title}",
                "",
                f"**TL;DR:** {s.tldr}",
                "",
                f"**Why it matters:** {s.why_it_matters}",
                "",
            ]
            if s.quote:
                section_lines.append(f"> {s.quote}")
                section_lines.append("")
            section_lines.append(f"[{s.id}]({s.link})")
            sections.append("\n".join(section_lines))
        body = ("\n\n---\n\n").join(sections) + "\n"

    # Feedback section — one prepopulated entry per summarized paper.
    # Skipped keepers are omitted (no body to react to). When there are
    # no summaries, the whole section is omitted.
    feedback_block = ""
    if summaries:
        fb_lines = ["## Feedback", ""]
        for s in summaries:
            fb_lines.append(f"### {s.title} ({s.id})")
            fb_lines.append("- Signal: [ ]")
            fb_lines.append("- Notes:")
            fb_lines.append("")
        feedback_block = "\n---\n\n" + "\n".join(fb_lines)

    return (
        "\n".join(fm_lines)
        + header
        + "\n"
        + summary_line
        + "\n\n"
        + how_block
        + "\n\n---\n\n"
        + body
        + feedback_block
    )


def write_brief(
    vault_path: Path | str,
    date: str,
    summaries: list[PaperSummary],
    keepers: list[Keeper],
    model_profile: str,
    papers_total: int,
) -> Path:
    """Render and write today's brief into ``<vault_path>/Briefs/``.

    Args:
        vault_path: Root of the Obsidian vault. Must exist and be writable;
            caller is expected to validate before invoking (see
            :func:`check_vault_writable`).
        date: ISO date like ``"2026-05-14"``. Used in the filename and
            in the frontmatter.
        summaries: Per-paper summaries from :func:`agent.summarize.summarize_keepers`.
        keepers: Filter keepers — used to render "How I picked these"
            with reasons in the user's voice.
        model_profile: Profile name used for this run; recorded in
            frontmatter for later reference.
        papers_total: Total papers on the day's list (before filtering).

    Returns:
        The path written. ``<vault_path>/Briefs/<date>.md`` normally;
        ``<date>-run-N.md`` on same-day rerun.
    """
    vault_path = Path(vault_path)
    briefs_dir = vault_path / "Briefs"
    target = _allocate_brief_path(briefs_dir, date)

    content = render_brief(
        date=date,
        summaries=summaries,
        keepers=keepers,
        model_profile=model_profile,
        papers_total=papers_total,
    )
    _atomic_write_text(target, content)
    return target


def check_vault_writable(vault_path: Path | str) -> None:
    """Raise an informative error if ``vault_path`` isn't usable.

    Called as a pre-flight before the pipeline runs so a typo'd config
    fails fast — not after 10 minutes of filter + summarize work.
    """
    p = Path(vault_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Obsidian vault path does not exist: {p}. "
            "Check vault_path in config.toml."
        )
    if not p.is_dir():
        raise NotADirectoryError(
            f"Obsidian vault path is not a directory: {p}. "
            "Check vault_path in config.toml."
        )
    if not os.access(p, os.W_OK):
        raise PermissionError(
            f"Obsidian vault path is not writable: {p}. "
            "Check filesystem permissions."
        )


__all__ = ["write_brief", "render_brief", "check_vault_writable"]
