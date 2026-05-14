"""Configuration loading.

Reads ``config.toml`` into a typed pydantic model. Loads ``.env`` so
subprocess tools (e.g. ``hf``) inherit secrets like ``HF_TOKEN`` from
the shell environment.

Multiple ``[models.<profile>]`` tables are supported; the CLI picks
one with ``--model <profile>``, falling back to ``default_model`` when
unspecified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

DEFAULT_CONFIG_PATH = Path("config.toml")
DEFAULT_ENV_PATH = Path(".env")


class ModelProfile(BaseModel):
    """One swappable backend+model+params bundle.

    Each profile maps to a ``[models.<name>]`` table in config.toml. The
    profile name (the dict key in ``Config.models``) is what the CLI
    references via ``--model <name>``.
    """

    backend: Literal["llamacpp", "ollama"]
    model_name: str
    base_url: HttpUrl
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(gt=0)
    context_length: int = Field(gt=0)
    supports_thinking: bool = False


class Config(BaseSettings):
    """Top-level runtime config loaded from config.toml.

    Field-level validation catches typos and bad URLs at startup. A
    cross-field validator confirms ``default_model`` names a real profile.
    """

    model_config = SettingsConfigDict(
        toml_file=str(DEFAULT_CONFIG_PATH),
        extra="forbid",
    )

    default_model: str
    vault_path: Path
    iteration_cap: int = Field(default=25, gt=0)
    models: dict[str, ModelProfile]

    @field_validator("vault_path", mode="after")
    @classmethod
    def _expand_vault_path(cls, v: Path) -> Path:
        # Allow ~ in config.toml without forcing the user to spell out $HOME.
        return v.expanduser()

    @model_validator(mode="after")
    def _check_default_model_exists(self) -> "Config":
        if self.default_model not in self.models:
            known = ", ".join(sorted(self.models)) or "<none>"
            raise ValueError(
                f"default_model={self.default_model!r} is not a defined profile. "
                f"Known profiles: {known}"
            )
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # TOML is the only source of config values. Env vars are reserved
        # for secrets consumed by subprocess tools, not for overriding
        # config values (keeps "what's set" trivially auditable).
        return (init_settings, TomlConfigSettingsSource(settings_cls))

    def profile(self, name: str | None = None) -> ModelProfile:
        """Look up a model profile by name, defaulting to ``default_model``."""
        chosen = name or self.default_model
        try:
            return self.models[chosen]
        except KeyError:
            known = ", ".join(sorted(self.models)) or "<none>"
            raise ValueError(
                f"Unknown model profile {chosen!r}. Known profiles: {known}"
            ) from None


def load_config(
    path: Path | str = DEFAULT_CONFIG_PATH,
    env_path: Path | str = DEFAULT_ENV_PATH,
) -> Config:
    """Load config from a TOML file. Also loads ``.env`` if present.

    Args:
        path: Path to the TOML config file. Default: ``config.toml`` in CWD.
        env_path: Path to the .env file (loaded into ``os.environ`` for
            subprocess tools). Missing file is fine — silently skipped.

    Raises:
        FileNotFoundError: With a hint pointing at ``config.toml.example``
            when the config file is missing.
        pydantic.ValidationError: When config values fail validation.
            The message identifies the offending field.
    """
    config_path = Path(path)
    env_path = Path(env_path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Copy config.toml.example to config.toml and edit it for your machine."
        )

    if env_path.is_file():
        load_dotenv(env_path)

    # Temporarily override the class-level toml_file so callers can pass
    # arbitrary paths (e.g. in tests) without monkeypatching the class.
    class _Loader(Config):
        model_config = SettingsConfigDict(
            toml_file=str(config_path),
            extra="forbid",
        )

    return _Loader()  # type: ignore[call-arg]
