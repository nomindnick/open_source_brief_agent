"""Model backend abstraction (llama.cpp, Ollama, future frontier adapters).

The factory :func:`get_model` is the only entry point most callers need —
it resolves a profile name to a configured :class:`ModelInterface` instance.
"""

from __future__ import annotations

from agent.config import Config
from agent.model.base import ModelBackendError, ModelInterface, ModelResponse
from agent.model.llamacpp import LlamaCppModel
from agent.model.ollama import OllamaModel

__all__ = [
    "ModelInterface",
    "ModelResponse",
    "ModelBackendError",
    "LlamaCppModel",
    "OllamaModel",
    "get_model",
]


def get_model(config: Config, profile_name: str | None = None) -> ModelInterface:
    """Build a :class:`ModelInterface` for the named profile.

    Args:
        config: A loaded :class:`Config` (use :func:`agent.config.load_config`).
        profile_name: Profile name as it appears in ``config.toml``
            (e.g. ``"qwen3-30b-llamacpp"``). When None, falls back to
            ``config.default_model``.

    Returns:
        A configured model adapter implementing :class:`ModelInterface`.

    Raises:
        ValueError: If the profile doesn't exist, or its ``backend`` is
            not one of the known backends.
    """
    profile = config.profile(profile_name)
    match profile.backend:
        case "llamacpp":
            return LlamaCppModel(profile)
        case "ollama":
            return OllamaModel(profile)
        case other:  # pragma: no cover — pydantic Literal blocks this at load time
            raise ValueError(f"Unknown backend: {other!r}")
