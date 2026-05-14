"""Tests for the filter parser and the filter_papers orchestrator.

The live behavior (does the model pick good papers?) is verified by the
Sprint 3.2 smoke run. These tests cover parser leniency, structural
error reporting, and the orchestration contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent.filter import (
    FilterError,
    Keeper,
    filter_papers,
    parse_filter_response,
)
from agent.model.base import ModelResponse


# ── parse_filter_response ──────────────────────────────────────────────


def test_parse_clean_json_array():
    keepers = parse_filter_response(
        '[{"id":"2509.05591","reason":"local inference"},'
        '{"id":"2509.04122","reason":"RL for LLMs"}]'
    )
    assert len(keepers) == 2
    assert keepers[0].id == "2509.05591"
    assert keepers[0].reason == "local inference"


def test_parse_strips_code_fence():
    raw = '```json\n[{"id": "2509.05591", "reason": "ok"}]\n```'
    keepers = parse_filter_response(raw)
    assert keepers[0].id == "2509.05591"


def test_parse_tolerates_prose_preamble():
    raw = 'Here are my picks:\n\n[{"id": "x", "reason": "matches"}]'
    keepers = parse_filter_response(raw)
    assert keepers[0].id == "x"


def test_parse_tolerates_trailing_comma():
    keepers = parse_filter_response('[{"id":"x","reason":"y"},]')
    assert keepers[0].id == "x"


def test_parse_empty_array_is_valid():
    keepers = parse_filter_response("[]")
    assert keepers == []


def test_parse_no_array_at_all_raises():
    with pytest.raises(FilterError, match="did not contain a JSON array"):
        parse_filter_response("I don't know what to pick.")


def test_parse_missing_id_raises_with_index():
    with pytest.raises(FilterError, match="#1"):
        parse_filter_response('[{"reason": "no id"}]')


def test_parse_empty_id_raises():
    with pytest.raises(FilterError, match="empty 'id'"):
        parse_filter_response('[{"id": "", "reason": "x"}]')


def test_parse_missing_reason_raises_with_id_in_message():
    with pytest.raises(FilterError, match="2509.05591"):
        parse_filter_response('[{"id": "2509.05591"}]')


def test_parse_non_object_entry_raises():
    with pytest.raises(FilterError, match="not an object"):
        parse_filter_response('["just a string"]')


def test_parse_non_array_root_raises():
    with pytest.raises(FilterError, match="JSON array"):
        parse_filter_response('{"id": "x", "reason": "y"}')


def test_parse_malformed_json_raises():
    with pytest.raises(FilterError, match="not valid JSON"):
        parse_filter_response("[{this is not json}]")


def test_parse_trims_whitespace_from_fields():
    keepers = parse_filter_response('[{"id": "  x  ", "reason": "  y  "}]')
    assert keepers[0].id == "x"
    assert keepers[0].reason == "y"


# ── filter_papers (orchestration) ──────────────────────────────────────


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


def test_filter_papers_happy_path(tmp_path, monkeypatch):
    # Need prompts/ on disk for load_prompt to work; point at a tmp dir
    # with our prompt file copied in.
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "_tool_calling_format.md").write_text("(unused by filter)")
    (prompts_dir / "system_filter.md").write_text(
        "{{interests}}\n\n{{papers}}\n\nReturn JSON."
    )
    monkeypatch.chdir(tmp_path)

    model = _StubModel(
        response_content='[{"id": "abc", "reason": "matches your interests"}]',
        response_reasoning="thinking step",
    )
    result = filter_papers(model, list_markdown="# papers", interests="interests")

    assert len(result.keepers) == 1
    assert result.keepers[0] == Keeper(id="abc", reason="matches your interests")
    assert result.response.reasoning == "thinking step"


def test_filter_papers_propagates_filter_error(tmp_path, monkeypatch):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "system_filter.md").write_text("{{interests}}\n{{papers}}")
    monkeypatch.chdir(tmp_path)

    model = _StubModel(response_content="no JSON here, just prose")
    with pytest.raises(FilterError):
        filter_papers(model, list_markdown="# papers", interests="interests")
