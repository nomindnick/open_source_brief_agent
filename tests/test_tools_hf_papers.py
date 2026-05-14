"""Unit tests for the HF papers tools — subprocess is mocked.

Live behavior is exercised by the smoke run in Sprint 3.1; these tests
cover pure formatting, error handling, and bad-input recovery so the
agent's observation contracts hold even when the CLI misbehaves.
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from agent.tools.hf_papers import (
    HfPapersListTool,
    HfPapersReadTool,
    _clean_paper_text,
    _format_list_as_markdown,
)


# ── pure functions ──────────────────────────────────────────────────────


def test_format_list_includes_id_title_upvotes_and_abstract():
    md = _format_list_as_markdown(
        [
            {"id": "2509.05591", "title": "Test paper", "upvotes": 42, "summary": "Some abstract."},
        ]
    )
    assert "2509.05591" in md
    assert "Test paper" in md
    assert "42 upvotes" in md
    assert "Some abstract." in md


def test_format_list_truncates_long_abstracts():
    long = "x" * 2000
    md = _format_list_as_markdown([{"id": "x", "title": "t", "upvotes": 0, "summary": long}])
    assert "…" in md
    assert len(md) < 2000


def test_format_list_handles_missing_fields():
    md = _format_list_as_markdown([{"id": "x"}])
    assert "x" in md
    assert "<untitled>" in md
    assert "no upvotes" in md


def test_format_list_handles_empty_list():
    md = _format_list_as_markdown([])
    assert "No papers" in md


def test_clean_paper_text_strips_preamble_and_uses_title_as_heading():
    raw = (
        "Title: My Paper\n"
        "URL Source: https://arxiv.org/html/2509.05591\n\n"
        "Markdown Content:\n"
        "Some author\n\n"
        "###### Abstract\n\n"
        "We propose ...\n"
    )
    cleaned = _clean_paper_text(raw)
    assert cleaned.startswith("# My Paper")
    assert "URL Source" not in cleaned
    assert "Markdown Content" not in cleaned
    assert "We propose" in cleaned


def test_clean_paper_text_is_idempotent_on_already_clean_input():
    raw = "# Already a heading\n\nBody.\n"
    cleaned = _clean_paper_text(raw)
    assert "Already a heading" in cleaned
    assert "Body." in cleaned


# ── HfPapersListTool ───────────────────────────────────────────────────


def _mock_run(returncode=0, stdout="", stderr=""):
    def fake(args, **kwargs):  # noqa: ARG001
        return subprocess.CompletedProcess(
            args=args, returncode=returncode, stdout=stdout, stderr=stderr
        )

    return fake


def test_list_happy_path():
    payload = json.dumps(
        [{"id": "2509.05591", "title": "T", "upvotes": 1, "summary": "S"}]
    )
    with patch("agent.tools.hf_papers.subprocess.run", side_effect=_mock_run(stdout=payload)):
        out = HfPapersListTool().run({"date": "2026-05-14"})
    assert "2509.05591" in out
    assert "T" in out


def test_list_validates_date_format():
    out = HfPapersListTool().run({"date": "yesterday"})
    assert out.startswith("ERROR")
    assert "YYYY-MM-DD" in out


def test_list_defaults_to_today_when_no_date():
    payload = json.dumps([])
    captured_args: list[list[str]] = []

    def fake(args, **kwargs):  # noqa: ARG001
        captured_args.append(args)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=payload, stderr="")

    with patch("agent.tools.hf_papers.subprocess.run", side_effect=fake):
        HfPapersListTool().run({})
    # No date passed → tool should have supplied today's date.
    assert "--date" in captured_args[0]
    # ISO date format (YYYY-MM-DD) immediately after --date.
    idx = captured_args[0].index("--date")
    assert len(captured_args[0][idx + 1]) == 10


def test_list_surfaces_cli_error_as_observation():
    with patch(
        "agent.tools.hf_papers.subprocess.run",
        side_effect=_mock_run(returncode=1, stderr="Network unreachable\n"),
    ):
        out = HfPapersListTool().run({"date": "2026-05-14"})
    assert out.startswith("ERROR")
    assert "Network unreachable" in out


def test_list_handles_unparseable_json():
    with patch(
        "agent.tools.hf_papers.subprocess.run",
        side_effect=_mock_run(stdout="not json"),
    ):
        out = HfPapersListTool().run({"date": "2026-05-14"})
    assert out.startswith("ERROR")
    assert "JSON" in out or "json" in out.lower()


def test_list_returns_error_when_hf_cli_missing():
    def fake(args, **kwargs):  # noqa: ARG001
        raise FileNotFoundError

    with patch("agent.tools.hf_papers.subprocess.run", side_effect=fake):
        out = HfPapersListTool().run({})
    assert "ERROR" in out
    assert "hf" in out


def test_list_returns_error_on_timeout():
    def fake(args, **kwargs):  # noqa: ARG001
        raise subprocess.TimeoutExpired(cmd=args, timeout=60)

    with patch("agent.tools.hf_papers.subprocess.run", side_effect=fake):
        out = HfPapersListTool().run({})
    assert out.startswith("ERROR")
    assert "timed out" in out


# ── HfPapersReadTool ───────────────────────────────────────────────────


def test_read_validates_id_required():
    out = HfPapersReadTool().run({})
    assert out.startswith("ERROR")
    assert "id" in out


def test_read_validates_id_is_string():
    out = HfPapersReadTool().run({"id": 123})
    assert out.startswith("ERROR")


def test_read_happy_path_strips_preamble():
    raw = (
        "Title: Sample\nURL Source: https://x\n\nMarkdown Content:\n"
        "###### Abstract\n\nText here.\n"
    )
    with patch("agent.tools.hf_papers.subprocess.run", side_effect=_mock_run(stdout=raw)):
        out = HfPapersReadTool().run({"id": "2509.05591"})
    assert out.startswith("# Sample")
    assert "URL Source" not in out


def test_read_surfaces_bad_id_as_observation():
    """The agent must self-correct on bad ids — not see an exception."""
    with patch(
        "agent.tools.hf_papers.subprocess.run",
        side_effect=_mock_run(
            returncode=1,
            stderr="Error: Paper '9999.99999' not found on the Hub.\n",
        ),
    ):
        out = HfPapersReadTool().run({"id": "9999.99999"})
    assert out.startswith("ERROR")
    assert "not found" in out
