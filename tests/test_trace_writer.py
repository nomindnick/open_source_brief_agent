"""TraceWriter unit tests.

Covers: file creation, append-mode line buffering, JSONL validity, MD
structure (reasoning blockquote, code-fenced content, truncation marker),
and same-day rerun directory allocation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.trace.writer import TRACE_SCHEMA_VERSION, TraceWriter


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def test_creates_files_under_expected_path(tmp_path: Path):
    with TraceWriter(tmp_path, "2026-05-14", "test") as w:
        pass
    assert (tmp_path / "2026-05-14" / "test" / "trace.jsonl").is_file()
    assert (tmp_path / "2026-05-14" / "test" / "trace.md").is_file()


def test_jsonl_has_meta_header_and_close_record(tmp_path: Path):
    with TraceWriter(tmp_path, "2026-05-14", "test", model_profile="x", max_iter=7):
        pass
    events = _read_jsonl(tmp_path / "2026-05-14" / "test" / "trace.jsonl")
    assert events[0]["event"] == "meta"
    assert events[0]["trace_schema_version"] == TRACE_SCHEMA_VERSION
    assert events[0]["model_profile"] == "x"
    assert events[0]["max_iter"] == 7
    assert events[-1]["event"] == "close"


def test_every_jsonl_line_is_valid_json(tmp_path: Path):
    with TraceWriter(tmp_path, "2026-05-14", "test") as w:
        w.log_model_turn(content="<final_answer>hi</final_answer>", reasoning="thinking")
        w.log_tool_call("echo", {"message": "x"})
        w.log_tool_result("echo", "x")
        w.log_parse_error("bad json")
        w.log_final_answer("hi")
    raw = (tmp_path / "2026-05-14" / "test" / "trace.jsonl").read_text()
    # Each non-empty line parses individually.
    for line in raw.splitlines():
        if line.strip():
            json.loads(line)


def test_md_has_header_and_turn_sections(tmp_path: Path):
    with TraceWriter(tmp_path, "2026-05-14", "test", model_profile="qwen") as w:
        w.log_model_turn(
            content="<tool_use><name>echo</name><input>{}</input></tool_use>",
            reasoning="step 1\nstep 2",
        )
        w.log_tool_call("echo", {})
        w.log_tool_result("echo", "ok")
        w.log_model_turn(content="<final_answer>done</final_answer>", reasoning=None)
        w.log_final_answer("done")
    md = (tmp_path / "2026-05-14" / "test" / "trace.md").read_text()
    assert "# Trace: test — 2026-05-14" in md
    assert "**Model profile:** qwen" in md
    assert "## Turn 1" in md
    assert "## Turn 2" in md
    # Reasoning rendered as blockquote.
    assert "> **Reasoning**" in md
    assert "> step 1" in md
    assert "> step 2" in md
    # Model content in xml code fence.
    assert "```xml" in md
    assert "## Final answer" in md


def test_long_tool_result_is_truncated_in_md_but_full_in_jsonl(tmp_path: Path):
    big = "x" * 2000
    with TraceWriter(tmp_path, "2026-05-14", "test") as w:
        w.log_tool_result("hf_papers_read", big)
    md = (tmp_path / "2026-05-14" / "test" / "trace.md").read_text()
    jsonl = _read_jsonl(tmp_path / "2026-05-14" / "test" / "trace.jsonl")
    # MD shows truncation marker.
    assert "truncated" in md.lower()
    assert big not in md  # full content not in MD
    # JSONL has the full string.
    result_event = next(e for e in jsonl if e["event"] == "tool_result")
    assert result_event["result"] == big


def test_same_day_rerun_gets_suffixed_dir(tmp_path: Path):
    a = TraceWriter(tmp_path, "2026-05-14", "test")
    a.close()
    b = TraceWriter(tmp_path, "2026-05-14", "test")
    b.close()
    c = TraceWriter(tmp_path, "2026-05-14", "test")
    c.close()
    assert a.run_dir.name == "test"
    assert b.run_dir.name == "test-1"
    assert c.run_dir.name == "test-2"


def test_close_is_idempotent(tmp_path: Path):
    w = TraceWriter(tmp_path, "2026-05-14", "test")
    w.close()
    w.close()  # must not raise


def test_partial_trace_survives_unclosed_writer(tmp_path: Path):
    """Line-buffered append: events written before a crash remain on disk."""
    w = TraceWriter(tmp_path, "2026-05-14", "test")
    w.log_model_turn(content="x", reasoning=None)
    # Don't call close(). Simulate a crash by just dropping the reference.
    jsonl = (tmp_path / "2026-05-14" / "test" / "trace.jsonl").read_text()
    md = (tmp_path / "2026-05-14" / "test" / "trace.md").read_text()
    assert '"model_turn"' in jsonl
    assert "## Turn 1" in md
