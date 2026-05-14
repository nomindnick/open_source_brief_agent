"""Prompt loading + ``{{placeholder}}`` interpolation.

Mission prompts live as markdown files under ``prompts/``. They reference
runtime values (the registered tools, the user's interests, etc.) via
``{{key}}`` placeholders. This module reads the markdown and does the
substitution.

The shared tool-calling format doc (``prompts/_tool_calling_format.md``)
is auto-included as the ``{{tool_calling_format}}`` placeholder so every
mission gets the same schema documentation without copy-pasting.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_PROMPTS_DIR = Path("prompts")
TOOL_CALLING_FORMAT_PARTIAL = "_tool_calling_format.md"


def load_prompt(
    name: str,
    prompts_dir: Path | str = DEFAULT_PROMPTS_DIR,
    **placeholders: str,
) -> str:
    """Load ``prompts/<name>.md`` and substitute ``{{key}}`` placeholders.

    The ``tool_calling_format`` placeholder is filled in automatically
    from ``_tool_calling_format.md`` unless the caller overrides it.

    Args:
        name: Mission/prompt name (without ``.md``). e.g. ``"system_test"``.
        prompts_dir: Directory containing the prompt files. Default
            ``prompts/`` relative to CWD.
        **placeholders: Values to substitute for ``{{key}}`` occurrences.

    Returns:
        The fully-substituted prompt string.

    Raises:
        FileNotFoundError: If the named prompt file does not exist.
        ValueError: If the rendered prompt still contains ``{{...}}``
            placeholders that were not provided.
    """
    prompts_dir = Path(prompts_dir)
    prompt_path = prompts_dir / f"{name}.md"
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    text = prompt_path.read_text(encoding="utf-8")

    # Auto-inject the tool-calling format unless caller provided their own.
    if "tool_calling_format" not in placeholders:
        partial_path = prompts_dir / TOOL_CALLING_FORMAT_PARTIAL
        if partial_path.is_file():
            placeholders["tool_calling_format"] = partial_path.read_text(encoding="utf-8")

    for key, value in placeholders.items():
        text = text.replace(f"{{{{{key}}}}}", value)

    # Catch typos: if a {{...}} marker survived substitution, that's almost
    # always a placeholder the caller forgot. Fail loudly so a half-rendered
    # prompt doesn't get silently shipped to the model.
    leftover = _find_unfilled_placeholders(text)
    if leftover:
        raise ValueError(
            f"Prompt {name!r} has unfilled placeholders: {sorted(leftover)}. "
            f"Pass them as kwargs to load_prompt()."
        )

    return text


def _find_unfilled_placeholders(text: str) -> set[str]:
    """Return any ``{{key}}`` markers that remain in ``text``."""
    import re

    return set(re.findall(r"\{\{(\w+)\}\}", text))
