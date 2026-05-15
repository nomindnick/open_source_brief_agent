"""Tools wrapping ``hf papers list`` and ``hf papers read``.

Both tools shell out to the ``hf`` CLI rather than calling
``huggingface_hub`` directly — the CLI surface is more stable across
versions and keeps our dependency tree thin. Per the :class:`Tool`
contract, neither tool ever raises out of ``run``: subprocess failures
are folded into an error string the agent can react to.
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import date as date_cls
from typing import Any

from agent.tools.base import Tool

# Defaults used when callers don't override via the constructors.
# Production wiring sources these from config.toml's [paper_survey] table;
# tests and ad-hoc scripts get reasonable behavior without configuration.
DEFAULT_SUBPROCESS_TIMEOUT_S = 60
DEFAULT_LIST_ABSTRACT_CHARS = 500


def _run_hf(args: list[str], timeout_s: int = DEFAULT_SUBPROCESS_TIMEOUT_S) -> tuple[bool, str]:
    """Invoke ``hf`` with the given args. Returns (success, output_or_error).

    Output is captured stdout on success; stderr (or a synthesized message)
    on failure. ``check=False`` so non-zero exits don't raise — the agent
    surface handles errors as observations.
    """
    try:
        result = subprocess.run(
            ["hf", *args],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError:
        return False, "ERROR: `hf` CLI is not installed or not on PATH."
    except subprocess.TimeoutExpired:
        return False, (
            f"ERROR: `hf {' '.join(args)}` timed out after {timeout_s}s."
        )

    if result.returncode != 0:
        # Surface a concise error from stderr (or stdout if stderr is empty).
        msg = (result.stderr or result.stdout or "<no output>").strip().splitlines()
        first_line = msg[0] if msg else "<no output>"
        return False, f"ERROR: `hf {' '.join(args)}` failed: {first_line}"

    return True, result.stdout


def _format_list_as_markdown(
    papers: list[dict[str, Any]],
    abstract_chars: int = DEFAULT_LIST_ABSTRACT_CHARS,
) -> str:
    """Compress a JSON paper list into a scannable markdown summary."""
    if not papers:
        return "No papers returned for that date."
    sections: list[str] = [f"# HuggingFace Papers — {len(papers)} result(s)\n"]
    for p in papers:
        pid = p.get("id", "<unknown>")
        title = (p.get("title") or "<untitled>").strip()
        upvotes = p.get("upvotes")
        summary = (p.get("summary") or "").strip()
        if len(summary) > abstract_chars:
            summary = summary[:abstract_chars].rstrip() + "…"
        # Normalize whitespace inside the summary so it doesn't visually
        # blow up the list view.
        summary = re.sub(r"\s+", " ", summary)

        upvote_str = f"{upvotes} upvotes" if upvotes is not None else "no upvotes"
        sections.append(
            f"## {pid} ({upvote_str})\n"
            f"**{title}**\n\n"
            f"{summary}\n"
        )
    return "\n".join(sections)


# Pre-compiled patterns for `hf papers read` output cleanup.
_PAPER_PREAMBLE_RE = re.compile(
    r"^Title:\s*(?P<title>.+?)\nURL Source:\s*\S+\n+Markdown Content:\s*\n",
    re.MULTILINE,
)
_LONE_IMAGE_RE = re.compile(r"^\s*\[image[^\]]*\]\(.*?\)\s*$", re.MULTILINE)
_MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")


def _clean_paper_text(raw: str) -> str:
    """Strip the ``hf papers read`` preamble and minor noise.

    Replaces the ``Title: ... / URL Source: ... / Markdown Content:`` header
    with a single ``# <title>`` heading so the result reads as a normal
    markdown document.
    """
    raw = raw.strip()
    m = _PAPER_PREAMBLE_RE.match(raw)
    if m:
        title = m.group("title").strip()
        body = raw[m.end():]
        raw = f"# {title}\n\n{body}"

    raw = _LONE_IMAGE_RE.sub("", raw)
    raw = _MULTI_BLANK_LINE_RE.sub("\n\n", raw)
    return raw.strip() + "\n"


class HfPapersListTool(Tool):
    name = "hf_papers_list"
    description = (
        "Lists papers HuggingFace has surfaced as 'Daily Papers' for a given date. "
        "Returns a markdown summary with one section per paper (id, title, "
        "truncated abstract, upvote count). Use this to see what's available "
        "before picking specific papers to read in full."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "date": {
                "type": "string",
                "description": (
                    "Date in YYYY-MM-DD format. Optional — defaults to today "
                    "(system local date)."
                ),
            }
        },
    }

    def __init__(
        self,
        *,
        timeout_s: int = DEFAULT_SUBPROCESS_TIMEOUT_S,
        abstract_chars: int = DEFAULT_LIST_ABSTRACT_CHARS,
    ) -> None:
        self._timeout_s = timeout_s
        self._abstract_chars = abstract_chars

    def run(self, input: dict[str, Any]) -> str:
        date = input.get("date") or date_cls.today().isoformat()
        if not isinstance(date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
            return (
                f"ERROR: 'date' must be a YYYY-MM-DD string. Got {date!r}."
            )

        ok, output = _run_hf(
            ["papers", "list", "--date", date, "--format", "json"],
            timeout_s=self._timeout_s,
        )
        if not ok:
            return output

        try:
            papers = json.loads(output)
        except json.JSONDecodeError as e:
            return f"ERROR: hf papers list returned unparseable JSON: {e}"

        if not isinstance(papers, list):
            return f"ERROR: hf papers list returned unexpected shape: {type(papers).__name__}"

        return _format_list_as_markdown(papers, abstract_chars=self._abstract_chars)


class HfPapersReadTool(Tool):
    name = "hf_papers_read"
    description = (
        "Fetches the full text of a HuggingFace-surfaced paper by arxiv id. "
        "Returns the paper as markdown — title, abstract, body, references. "
        "Use after `hf_papers_list` to read specific papers in detail."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "arxiv-style paper id, e.g. '2509.05591' or '2605.13647'.",
            }
        },
        "required": ["id"],
    }

    def __init__(self, *, timeout_s: int = DEFAULT_SUBPROCESS_TIMEOUT_S) -> None:
        self._timeout_s = timeout_s

    def run(self, input: dict[str, Any]) -> str:
        pid = input.get("id")
        if not isinstance(pid, str) or not pid.strip():
            return f"ERROR: 'id' must be a non-empty string. Got {pid!r}."
        pid = pid.strip()

        ok, output = _run_hf(["papers", "read", pid], timeout_s=self._timeout_s)
        if not ok:
            return output

        return _clean_paper_text(output)


__all__ = ["HfPapersListTool", "HfPapersReadTool"]
