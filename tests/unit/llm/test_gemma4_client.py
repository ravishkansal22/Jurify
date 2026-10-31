"""Unit tests for Gemma4Client — all Google GenAI calls are mocked."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.config import LLMSettings, Settings
from app.llm.gemma_client import (
    DEFAULT_GEMMA4_MODEL,
    FALLBACK_GEMMA4_MODEL,
    Gemma4Client,
    GemmaAPIError,
    get_chat_model,
    get_gemma4_client,
)


def _settings_with_key(key: str = "<your-key>") -> Settings:
    return Settings(
        _env_file=None,
        llm=LLMSettings(
            _env_file=None,
            llm_provider="gemini",
            gemma_model_name=DEFAULT_GEMMA4_MODEL,
            gemma_fallback_model_name=FALLBACK_GEMMA4_MODEL,
            google_api_key=key,  # type: ignore[arg-type]
        ),
    )


def test_gemma4_client_requires_api_key() -> None:
    settings = Settings(
        _env_file=None,
        llm=LLMSettings(_env_file=None, llm_provider="gemini", google_api_key=None),
    )
    with pytest.raises(GemmaAPIError, match="GOOGLE_API_KEY"):
        Gemma4Client(settings)


def test_get_chat_model_rejects_gemini_provider() -> None:
    settings = _settings_with_key()
    with pytest.raises(ValueError, match="Gemma4Client"):
        get_chat_model(settings)


@pytest.mark.asyncio
async def test_generate_text_uses_google_genai_and_thinking() -> None:
    settings = _settings_with_key()
    mock_response = SimpleNamespace(text="plain summary")

    mock_aio_models = SimpleNamespace(
        generate_content=AsyncMock(return_value=mock_response)
    )
    mock_client = MagicMock()
    mock_client.aio.models = mock_aio_models

    with patch("google.genai.Client", return_value=mock_client) as client_ctor:
        client = get_gemma4_client(settings)
        text = await client.generate_text(
            "Explain this lease",
            system_instruction="You are a legal document explainer.",
            thinking_level="high",
        )

    client_ctor.assert_called_once_with(api_key="<your-key>")
    assert text == "plain summary"
    kwargs = mock_aio_models.generate_content.await_args.kwargs
    assert kwargs["model"] == DEFAULT_GEMMA4_MODEL
    assert kwargs["contents"] == "Explain this lease"
    assert kwargs["config"].system_instruction == "You are a legal document explainer."
    level = kwargs["config"].thinking_config.thinking_level
    assert str(getattr(level, "value", level)).lower() == "high"


@pytest.mark.asyncio
async def test_generate_with_image_uses_inline_bytes() -> None:
    settings = _settings_with_key()
    mock_response = SimpleNamespace(text="image analysis")
    mock_aio_models = SimpleNamespace(
        generate_content=AsyncMock(return_value=mock_response)
    )
    mock_client = MagicMock()
    mock_client.aio.models = mock_aio_models

    fake_part = object()
    with (
        patch("google.genai.Client", return_value=mock_client),
        patch("google.genai.types.Part.from_bytes", return_value=fake_part) as from_bytes,
    ):
        client = Gemma4Client(settings)
        text = await client.generate_with_image(
            "OCR and explain",
            image_bytes=b"fake-png-bytes",
            mime_type="image/png",
            thinking_level="minimal",
        )

    assert text == "image analysis"
    from_bytes.assert_called_once_with(data=b"fake-png-bytes", mime_type="image/png")
    contents = mock_aio_models.generate_content.await_args.kwargs["contents"]
    assert contents == [fake_part, "OCR and explain"]
    level = mock_aio_models.generate_content.await_args.kwargs[
        "config"
    ].thinking_config.thinking_level
    assert str(getattr(level, "value", level)).lower() == "minimal"


@pytest.mark.asyncio
async def test_generate_falls_back_to_31b_on_primary_failure() -> None:
    settings = _settings_with_key()
    mock_aio_models = SimpleNamespace(
        generate_content=AsyncMock(
            side_effect=[
                RuntimeError("primary down"),
                SimpleNamespace(text="fallback ok"),
            ]
        )
    )
    mock_client = MagicMock()
    mock_client.aio.models = mock_aio_models

    with patch("google.genai.Client", return_value=mock_client):
        client = Gemma4Client(settings)
        text = await client.generate_text("hello")

    assert text == "fallback ok"
    models_called = [
        c.kwargs["model"] for c in mock_aio_models.generate_content.await_args_list
    ]
    assert models_called == [DEFAULT_GEMMA4_MODEL, FALLBACK_GEMMA4_MODEL]


@pytest.mark.asyncio
async def test_generate_raises_gemma_api_error_when_both_fail() -> None:
    settings = _settings_with_key()
    mock_aio_models = SimpleNamespace(
        generate_content=AsyncMock(side_effect=RuntimeError("boom"))
    )
    mock_client = MagicMock()
    mock_client.aio.models = mock_aio_models

    with patch("google.genai.Client", return_value=mock_client):
        client = Gemma4Client(settings)
        with pytest.raises(GemmaAPIError, match="Gemma 4 request failed"):
            await client.generate_text("hello")
