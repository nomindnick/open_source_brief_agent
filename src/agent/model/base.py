"""Model backend contract.

The whole point of this module: the agent loop, parser, and tools must not
know which backend is generating tokens. They call ``model.complete(messages)``
and get back a structured :class:`ModelResponse` with thinking already
separated from content. Per-backend mechanics (HTTP shape, how thinking is
exposed) live in the concrete adapters.

V1 is sync. See Sprint 1.2 notes for the migration path to async if/when
parallel calls across multiple model servers become a goal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ModelResponse:
    """A single completion result, with reasoning separated from content.

    The agent loop's parser only ever consumes :attr:`content`; thinking
    tokens go into :attr:`reasoning` for the trace writer to render
    alongside (but visually subordinate to) the action.

    Attributes:
        content: Post-thinking output. What the XML parser parses for
            tool calls and final answers. Never contains ``<think>`` blocks
            even when the backend emits them inline.
        reasoning: The model's thinking text, or None for non-reasoning
            models / profiles with ``supports_thinking=False``. Logged
            to the trace but never parsed.
        raw: The full, unmodified response body the backend returned.
            For debugging — diff this against ``content`` to confirm
            the adapter's thinking-extraction is doing what you expect.
        usage: Backend-reported token counts. Shape is best-effort
            (e.g. ``{"prompt_tokens": int, "completion_tokens": int}``);
            empty dict when the backend doesn't report.
    """

    content: str
    reasoning: str | None = None
    raw: str = ""
    usage: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ModelInterface(Protocol):
    """Sync chat-completion interface.

    Implementations live in ``agent.model.llamacpp`` and ``agent.model.ollama``.
    Future frontier adapters (Claude, etc.) implement the same Protocol.

    The Protocol is intentionally tiny — one method, one return type — so
    that swapping backends, mocking in tests, and any future sync→async
    migration are all small changes.
    """

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> ModelResponse:
        """Generate a completion for the given chat messages.

        Args:
            messages: OpenAI-style chat messages, e.g.
                ``[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]``.
            **kwargs: Per-call overrides for profile defaults (e.g.
                ``temperature``, ``max_tokens``). Implementations may
                ignore keys they don't recognize.

        Returns:
            A :class:`ModelResponse`. ``content`` is always a string
            (possibly empty); ``reasoning`` is None when the profile
            doesn't support thinking or the model produced none.

        Raises:
            ModelBackendError: On HTTP/transport failure, non-2xx
                responses, or malformed backend output. Adapters wrap
                their backend-specific exceptions in this type so the
                agent loop has a single thing to catch.
        """
        ...


class ModelBackendError(RuntimeError):
    """A model backend failed to produce a usable response.

    Wraps connection errors, timeouts, non-2xx HTTP responses, and
    malformed backend output. The agent loop catches this to decide
    whether to retry or abort the run.
    """
