"""Ollama adapter — talks to ``/api/chat``.

Sends ``"think": true`` for profiles flagged ``supports_thinking`` and reads
``message.thinking`` into ``ModelResponse.reasoning``.

Filled in by Sprint 1.2.
"""
