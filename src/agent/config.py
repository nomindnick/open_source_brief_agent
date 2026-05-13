"""Configuration loading.

Reads ``config.toml`` and ``.env`` into a typed pydantic-settings model.
Supports multiple ``[models.<profile>]`` tables; the CLI selects one per run.

Filled in by Sprint 1.2.
"""
