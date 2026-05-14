"""llama.cpp adapter.

Talks to ``llama-server``'s OpenAI-compatible ``/v1/chat/completions``
endpoint. Reasoning models (Qwen3, etc.) emit ``<think>…</think>`` blocks
*inline* in the assistant's content; this adapter strips them out into
:attr:`ModelResponse.reasoning` so the agent loop's parser never sees them.

If the model didn't emit any think blocks, ``reasoning`` is None and
``content`` is returned unchanged.
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from agent.config import ModelProfile
from agent.model.base import ModelBackendError, ModelResponse

# Tolerate whitespace inside the tag boundary and DOTALL so think blocks
# spanning multiple lines are captured.
_THINK_RE = re.compile(r"<think\b[^>]*>(.*?)</think\s*>", re.DOTALL | re.IGNORECASE)


def _split_thinking(text: str) -> tuple[str, str | None]:
    """Extract <think>…</think> blocks from a model response.

    Returns (content_without_think_blocks, joined_reasoning_or_None).
    Multiple blocks are concatenated with a blank line between them.
    Content has leading/trailing whitespace stripped after removal so
    the parser doesn't see stray newlines where the think block used
    to be.
    """
    matches = _THINK_RE.findall(text)
    if not matches:
        return text, None
    cleaned = _THINK_RE.sub("", text).strip()
    reasoning = "\n\n".join(m.strip() for m in matches)
    return cleaned, reasoning


class LlamaCppModel:
    """``ModelInterface`` implementation for llama.cpp's HTTP server."""

    def __init__(
        self,
        profile: ModelProfile,
        timeout: float = 600.0,
        client: httpx.Client | None = None,
    ) -> None:
        if profile.backend != "llamacpp":
            raise ValueError(
                f"LlamaCppModel given profile with backend={profile.backend!r}"
            )
        self._profile = profile
        # Strip trailing slash so we can append /v1/... cleanly.
        self._base_url = str(profile.base_url).rstrip("/")
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def complete(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self._profile.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self._profile.temperature),
            "max_tokens": kwargs.get("max_tokens", self._profile.max_tokens),
            "stream": False,
        }

        url = f"{self._base_url}/v1/chat/completions"
        try:
            resp = self._client.post(url, json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise ModelBackendError(f"llama.cpp request failed: {e}") from e

        try:
            data = resp.json()
            message = data["choices"][0]["message"]
            raw_content = message.get("content") or ""
        except (KeyError, IndexError, ValueError) as e:
            raise ModelBackendError(
                f"llama.cpp returned an unexpected payload: {resp.text[:500]}"
            ) from e

        if self._profile.supports_thinking:
            content, reasoning = _split_thinking(raw_content)
        else:
            content, reasoning = raw_content, None

        return ModelResponse(
            content=content,
            reasoning=reasoning,
            raw=resp.text,
            usage=data.get("usage", {}) or {},
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "LlamaCppModel":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
