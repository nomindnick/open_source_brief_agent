"""Ollama adapter.

Talks to Ollama's ``/api/chat`` endpoint. Unlike llama.cpp, Ollama exposes
thinking as a structured field: when the request body sets ``"think": true``,
the response's ``message.thinking`` carries reasoning and ``message.content``
carries the final answer. No regex stripping needed for the happy path —
they arrive already separated.

For profiles with ``supports_thinking=False``, we omit the ``think`` key
entirely (sending ``"think": false`` works too but is unnecessary).
"""

from __future__ import annotations

from typing import Any

import httpx

from agent.config import ModelProfile
from agent.model.base import ModelBackendError, ModelResponse


class OllamaModel:
    """``ModelInterface`` implementation for the Ollama daemon."""

    def __init__(
        self,
        profile: ModelProfile,
        timeout: float = 600.0,
        client: httpx.Client | None = None,
    ) -> None:
        if profile.backend != "ollama":
            raise ValueError(
                f"OllamaModel given profile with backend={profile.backend!r}"
            )
        self._profile = profile
        self._base_url = str(profile.base_url).rstrip("/")
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def complete(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> ModelResponse:
        # Ollama puts inference params under "options". num_ctx mirrors
        # the profile's context_length so KV cache is sized to match what
        # the agent expects to be able to use.
        options: dict[str, Any] = {
            "temperature": kwargs.get("temperature", self._profile.temperature),
            "num_predict": kwargs.get("max_tokens", self._profile.max_tokens),
            "num_ctx": self._profile.context_length,
        }
        payload: dict[str, Any] = {
            "model": self._profile.model_name,
            "messages": messages,
            "options": options,
            "stream": False,
        }
        if self._profile.supports_thinking:
            payload["think"] = True

        url = f"{self._base_url}/api/chat"
        try:
            resp = self._client.post(url, json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise ModelBackendError(f"Ollama request failed: {e}") from e

        try:
            data = resp.json()
            message = data["message"]
            content = message.get("content") or ""
        except (KeyError, ValueError) as e:
            raise ModelBackendError(
                f"Ollama returned an unexpected payload: {resp.text[:500]}"
            ) from e

        # message.thinking is only present when think=true was honored.
        # Empty string from the server is treated as no reasoning.
        thinking = message.get("thinking") if self._profile.supports_thinking else None
        reasoning = thinking if thinking else None

        # Normalize usage to the same keys llama.cpp uses, so trace/UI code
        # doesn't have to switch on backend.
        usage: dict[str, Any] = {}
        if "prompt_eval_count" in data:
            usage["prompt_tokens"] = data["prompt_eval_count"]
        if "eval_count" in data:
            usage["completion_tokens"] = data["eval_count"]
        if "total_duration" in data:
            usage["total_duration_ns"] = data["total_duration"]

        return ModelResponse(
            content=content,
            reasoning=reasoning,
            raw=resp.text,
            usage=usage,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "OllamaModel":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
