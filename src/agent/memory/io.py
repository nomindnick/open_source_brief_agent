"""Memory file I/O.

Read/write helpers for the markdown files under the project's top-level
``memory/`` directory. Sprint 3.2 ships ``read_interests``; Sprint 4.2
fills in seen-id tracking and reflection write-back.

The ``memory/`` directory itself is at the *project root*, not under
``src/agent``. See SPEC § "Data Model".
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_MEMORY_DIR = Path("memory")


def read_interests(memory_dir: Path | str = DEFAULT_MEMORY_DIR) -> str:
    """Return the contents of ``memory/Interests.md``.

    Raises ``FileNotFoundError`` if missing — Interests.md is user-curated
    and is checked into the repo with seed content, so this should always
    succeed in a properly configured project.
    """
    path = Path(memory_dir) / "Interests.md"
    return path.read_text(encoding="utf-8")
