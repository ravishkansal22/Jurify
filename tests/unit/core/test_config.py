import pytest
from pydantic import ValidationError

from app.core.config import (
    DatabaseSettings,
    I18nSettings,
    LLMSettings,
    RedisSettings,
    Settings,
    get_settings,
)


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_settings_load_with_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # Nested settings models each read env_file=".env"; pass _env_file=None
    # so a developer's local .env cannot poison default assertions.
    for var in (
        "DATABASE_URL",
        "POSTGRES_USER",
        "REDIS_URL",
        "CORS_ORIGINS",
        "LLM_PROVIDER",
        "GEMMA_MODEL_NAME",
        "GEMMA_FALLBACK_MODEL_NAME",
        "GEMMA_FALLBACK_MODEL",
        "GOOGLE_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    settings = Settings(
        _env_file=None,
        database=DatabaseSettings(_env_file=None),
        redis=RedisSettings(_env_file=None),
        llm=LLMSettings(_env_file=None),
        i18n=I18nSettings(_env_file=None),
    )

    assert settings.environment == "dev"
    assert settings.database.postgres_db == "nyayaos"
    assert settings.redis.redis_host == "localhost"
    assert settings.cors_origins == ["*"]
    assert settings.llm.llm_provider == "ollama"
    assert settings.llm.gemma_model_name == "gemma-4-26b-a4b-it"
    assert settings.llm.gemma_fallback_model_name == "gemma-4-31b-it"
    assert settings.llm.gemma_fallback_model == "gemma-4-31b-it"
    assert settings.llm.google_api_key is None
    assert settings.i18n.default_language == "en"
    assert "hi" in settings.i18n.supported_languages


def test_google_api_key_loaded_as_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "<your-key>")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")

    llm = LLMSettings(_env_file=None)

    assert llm.llm_provider == "gemini"
    assert llm.google_api_key is not None
    assert llm.google_api_key.get_secret_value() == "<your-key>"
    assert "SecretStr" in repr(llm.google_api_key)


def test_get_settings_is_cached_singleton() -> None:
    first = get_settings()
    second = get_settings()
    assert first is second


def test_database_async_dsn_assembled_from_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("POSTGRES_USER", "alice")
    monkeypatch.setenv("POSTGRES_PASSWORD", "s3cret")
    monkeypatch.setenv("POSTGRES_HOST", "db.internal")
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    monkeypatch.setenv("POSTGRES_DB", "casesdb")

    database = DatabaseSettings(_env_file=None)

    assert database.async_dsn == (
        "postgresql+asyncpg://alice:s3cret@db.internal:5433/casesdb"
    )


def test_database_url_override_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://explicit-dsn/whatever")

    settings = Settings()

    assert settings.database.async_dsn == "postgresql+asyncpg://explicit-dsn/whatever"


def test_cors_origins_parsed_from_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "http://a.com, http://b.com,http://c.com")

    settings = Settings()

    assert settings.cors_origins == ["http://a.com", "http://b.com", "http://c.com"]


def test_supported_languages_parsed_from_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPPORTED_LANGUAGES", "en,hi,ta")

    settings = I18nSettings()

    assert settings.supported_languages == ["en", "hi", "ta"]


def test_unknown_language_code_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPPORTED_LANGUAGES", "en,xx")

    with pytest.raises(ValidationError):
        I18nSettings()


def test_unknown_default_language_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEFAULT_LANGUAGE", "fr")

    with pytest.raises(ValidationError):
        I18nSettings()


def test_jwt_secrets_are_secret_str() -> None:
    settings = Settings()

    assert "SecretStr" in repr(settings.jwt.jwt_secret_key)
    assert settings.jwt.jwt_secret_key.get_secret_value() != repr(settings.jwt.jwt_secret_key)
