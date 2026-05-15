"""Tests for summarize parser + summarize_keepers orchestration.

Live behavior is verified by the Sprint 3.3 smoke run. These tests cover
parse leniency, structural validation, and graceful-degradation in the
orchestrator (a bad read or unparseable summary skips that paper, not
the whole run).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from agent.filter import Keeper
from agent.loop import NoOpTrace
from agent.model.base import ModelResponse
from agent.summarize import (
    PaperSummary,
    SummarizeError,
    parse_summary_response,
    summarize_keeper,
    summarize_keepers,
)


# ── parse_summary_response ─────────────────────────────────────────────


def test_parse_clean_json_object():
    parsed = parse_summary_response(
        '{"title": "X", "tldr": "y", "why_it_matters": "z", "quote": "q"}'
    )
    assert parsed["title"] == "X"


def test_parse_strips_code_fence():
    parsed = parse_summary_response(
        '```json\n{"title": "X", "tldr": "y", "why_it_matters": "z", "quote": null}\n```'
    )
    assert parsed["quote"] is None


def test_parse_tolerates_prose_preamble():
    parsed = parse_summary_response(
        'Here is the summary:\n\n{"title": "X", "tldr": "y", "why_it_matters": "z"}'
    )
    assert parsed["title"] == "X"


def test_parse_no_object_raises():
    with pytest.raises(SummarizeError, match="did not contain a JSON object"):
        parse_summary_response("just prose, no json")


def test_parse_malformed_json_raises():
    with pytest.raises(SummarizeError, match="not valid JSON"):
        parse_summary_response("{this is not json}")


def test_parse_array_root_raises():
    with pytest.raises(SummarizeError, match="JSON object"):
        parse_summary_response('[{"title": "X"}]')


# ── summarize_keeper (single-paper happy path + validation) ───────────


@dataclass
class _StubModel:
    response_content: str
    response_reasoning: str | None = None

    def complete(self, messages, **kwargs):  # noqa: ARG002
        return ModelResponse(
            content=self.response_content,
            reasoning=self.response_reasoning,
            raw=self.response_content,
            usage={},
        )


def _seed_prompts(tmp_path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "_tool_calling_format.md").write_text("(unused)")
    (prompts_dir / "system_summarize.md").write_text(
        "{{paper_id}}\n{{filter_reason}}\n{{paper_text}}"
    )
    return prompts_dir


def test_summarize_keeper_happy_path(tmp_path, monkeypatch):
    _seed_prompts(tmp_path)
    monkeypatch.chdir(tmp_path)

    model = _StubModel(
        response_content=json.dumps(
            {
                "title": "Test Paper",
                "tldr": "Short summary.",
                "why_it_matters": "Two sentences. Refinement of filter reason.",
                "quote": "A direct verbatim sentence.",
            }
        )
    )
    keeper = Keeper(id="2509.05591", reason="pre-read reason")
    summary = summarize_keeper(model, keeper, paper_text="full paper text" * 100)

    assert summary == PaperSummary(
        id="2509.05591",
        title="Test Paper",
        tldr="Short summary.",
        why_it_matters="Two sentences. Refinement of filter reason.",
        quote="A direct verbatim sentence.",
        link="https://arxiv.org/abs/2509.05591",
        filter_reason="pre-read reason",
    )


def test_summarize_keeper_optional_quote_can_be_null(tmp_path, monkeypatch):
    _seed_prompts(tmp_path)
    monkeypatch.chdir(tmp_path)

    model = _StubModel(
        response_content=json.dumps(
            {
                "title": "T",
                "tldr": "t",
                "why_it_matters": "w",
                "quote": None,
            }
        )
    )
    summary = summarize_keeper(
        model, Keeper(id="x", reason="r"), paper_text="paper"
    )
    assert summary.quote is None


def test_summarize_keeper_missing_required_field_raises(tmp_path, monkeypatch):
    _seed_prompts(tmp_path)
    monkeypatch.chdir(tmp_path)

    model = _StubModel(
        response_content=json.dumps({"title": "T", "tldr": "t"})  # missing why_it_matters
    )
    with pytest.raises(SummarizeError, match="why_it_matters"):
        summarize_keeper(model, Keeper(id="x", reason="r"), paper_text="p")


def test_summarize_keeper_truncates_paper_text(tmp_path, monkeypatch):
    _seed_prompts(tmp_path)
    monkeypatch.chdir(tmp_path)

    captured: list[str] = []

    @dataclass
    class _CapturingModel:
        def complete(self, messages, **kwargs):  # noqa: ARG002
            captured.append(messages[0]["content"])
            return ModelResponse(
                content='{"title":"T","tldr":"t","why_it_matters":"w","quote":null}',
                reasoning=None,
                raw="",
                usage={},
            )

    big = "x" * 100_000
    summarize_keeper(
        _CapturingModel(),
        Keeper(id="x", reason="r"),
        paper_text=big,
        max_chars=1000,
    )
    # The prompt sent to the model should contain a truncated version.
    assert "x" * 1000 in captured[0]
    assert "x" * 2000 not in captured[0]


# ── summarize_keepers (orchestrator + graceful degradation) ──────────


class _FakeReadTool:
    def __init__(self, results: dict[str, str]) -> None:
        self.results = results
        self.calls: list[str] = []

    def run(self, input: dict[str, Any]) -> str:
        pid = input["id"]
        self.calls.append(pid)
        return self.results.get(pid, "ERROR: no such paper")


def test_summarize_keepers_skips_paper_with_read_error(tmp_path, monkeypatch):
    _seed_prompts(tmp_path)
    monkeypatch.chdir(tmp_path)

    keepers = [
        Keeper(id="good", reason="r1"),
        Keeper(id="bad", reason="r2"),
    ]
    read_tool = _FakeReadTool(
        {
            "good": "paper text",
            # "bad" returns "ERROR: ..." by default
        }
    )
    model = _StubModel(
        response_content=json.dumps(
            {"title": "T", "tldr": "t", "why_it_matters": "w", "quote": None}
        )
    )

    summaries = summarize_keepers(
        model, keepers, trace=NoOpTrace(), read_tool=read_tool
    )
    assert len(summaries) == 1
    assert summaries[0].id == "good"
    # Both papers were attempted.
    assert read_tool.calls == ["good", "bad"]


def test_summarize_keepers_skips_paper_with_unparseable_summary(tmp_path, monkeypatch):
    _seed_prompts(tmp_path)
    monkeypatch.chdir(tmp_path)

    keepers = [
        Keeper(id="good", reason="r"),
        Keeper(id="ugly", reason="r"),
    ]
    read_tool = _FakeReadTool({"good": "text", "ugly": "text"})

    # First call returns valid JSON, second returns garbage.
    responses = iter(
        [
            json.dumps(
                {"title": "T", "tldr": "t", "why_it_matters": "w", "quote": None}
            ),
            "this is not json and has no object",
        ]
    )

    @dataclass
    class _Sequential:
        def complete(self, messages, **kwargs):  # noqa: ARG002
            return ModelResponse(
                content=next(responses), reasoning=None, raw="", usage={}
            )

    summaries = summarize_keepers(
        _Sequential(), keepers, trace=NoOpTrace(), read_tool=read_tool
    )
    assert [s.id for s in summaries] == ["good"]


def test_summarize_keepers_contains_model_exceptions(tmp_path, monkeypatch):
    """If the model adapter raises, we skip the paper rather than crashing."""
    _seed_prompts(tmp_path)
    monkeypatch.chdir(tmp_path)

    keepers = [
        Keeper(id="good", reason="r"),
        Keeper(id="boom", reason="r"),
    ]
    read_tool = _FakeReadTool({"good": "text", "boom": "text"})

    responses = iter(
        [
            json.dumps(
                {"title": "T", "tldr": "t", "why_it_matters": "w", "quote": None}
            )
        ]
    )

    @dataclass
    class _Flaky:
        def complete(self, messages, **kwargs):  # noqa: ARG002
            try:
                return ModelResponse(
                    content=next(responses), reasoning=None, raw="", usage={}
                )
            except StopIteration:
                raise RuntimeError("backend exploded") from None

    summaries = summarize_keepers(
        _Flaky(), keepers, trace=NoOpTrace(), read_tool=read_tool
    )
    assert [s.id for s in summaries] == ["good"]


def test_summarize_keepers_empty_input_returns_empty():
    summaries = summarize_keepers(
        _StubModel(response_content="ignored"),
        keepers=[],
        trace=NoOpTrace(),
    )
    assert summaries == []
