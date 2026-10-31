"""Application configuration.

Single source of truth for all environment-driven settings, per
docs/architecture/13-configuration-and-devops.md. Grouped into nested
settings models for readability; env var names match .env.example exactly.

Deliberately has no dependency on app.schemas (that's Module 6 in the
implementation order) — SupportedLanguage here is a local, minimal set of
ISO codes rather than importing the future canonical enum.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_KNOWN_LANGUAGE_CODES = {"en", "hi", "bn", "ta", "te", "mr", "gu", "pa", "kn", "ml", "ur"}


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_user: str = "nyaya"
    postgres_password: SecretStr = SecretStr("change_me")
    postgres_db: str = "nyayaos"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: str | None = None

    @property
    def async_dsn(self) -> str:
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password.get_secret_value()}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_url: str | None = None

    @property
    def dsn(self) -> str:
        return self.redis_url or f"redis://{self.redis_host}:{self.redis_port}/0"


class JWTSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    jwt_secret_key: SecretStr = SecretStr("change_me_access_secret")
    jwt_refresh_secret: SecretStr = SecretStr("change_me_refresh_secret")
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # "gemini" = Gemma 4 via Google GenAI / Gemini API (hackday demo path).
    llm_provider: Literal["ollama", "vllm", "gemini"] = "ollama"
    # Default: Gemma 4 26B A4B (faster). Fallback: gemma-4-31b-it.
    gemma_model_name: str = "gemma-4-26b-a4b-it"
    # Accept GEMMA_FALLBACK_MODEL or legacy GEMMA_FALLBACK_MODEL_NAME.
    gemma_fallback_model_name: str = Field(
        default="gemma-4-31b-it",
        validation_alias=AliasChoices(
            "gemma_fallback_model_name",
            "GEMMA_FALLBACK_MODEL_NAME",
            "gemma_fallback_model",
            "GEMMA_FALLBACK_MODEL",
        ),
    )
    ollama_base_url: str = "http://localhost:11434"
    vllm_base_url: str = "http://localhost:8001"
    embedding_model: str = "nomic-embed-text"
    # Read only from .env / environment — never hardcode. Owner inserts the key.
    google_api_key: SecretStr | None = None

    @property
    def gemma_fallback_model(self) -> str:
        """Preferred alias for the fallback model id (Me 1 / Me 2 contract)."""
        return self.gemma_fallback_model_name


class VectorDBSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    vector_db: Literal["chroma", "pgvector", "qdrant"] = "chroma"


class ObservabilitySettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "nyayaos"
    log_level: str = "INFO"


class I18nSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    default_language: str = "en"
    supported_languages: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: sorted(_KNOWN_LANGUAGE_CODES)
    )

    @field_validator("supported_languages", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return value

    @field_validator("supported_languages")
    @classmethod
    def _validate_known_codes(cls, value: list[str]) -> list[str]:
        unknown = set(value) - _KNOWN_LANGUAGE_CODES
        if unknown:
            raise ValueError(f"Unsupported language code(s): {sorted(unknown)}")
        return value

    @field_validator("default_language")
    @classmethod
    def _validate_default(cls, value: str) -> str:
        if value not in _KNOWN_LANGUAGE_CODES:
            raise ValueError(f"Unknown default_language: {value!r}")
        return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    environment: Literal["dev", "staging", "prod"] = "dev"
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["*"]
    )
    max_upload_size: int = 20_971_520  # bytes
    upload_directory: str = "./uploads"
    ocr_provider: Literal["tesseract", "gemma_vision"] = "tesseract"
    google_maps_api_key: SecretStr | None = None

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    jwt: JWTSettings = Field(default_factory=JWTSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    vector_db: VectorDBSettings = Field(default_factory=VectorDBSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    i18n: I18nSettings = Field(default_factory=I18nSettings)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors(cls, value: object) -> object:
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    """Cached singleton accessor. Use via FastAPI Depends(get_settings), never as a bare import."""
    return Settings()
