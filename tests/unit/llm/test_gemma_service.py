"""Unit tests for GemmaService — Google GenAI SDK is always mocked."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from app.core.config import LLMSettings, Settings
from app.llm.gemma_service import GemmaError, GemmaService, get_gemma_service


class TinySchema(BaseModel):
    title: str
    score: int = Field(ge=0, le=10)


def _settings(
    *,
    primary: str = "gemma-4-26b-a4b-it",
    fallback: str = "gemma-4-31b-it",
    key: str = "test-key",
) -> Settings:
    return Settings(
        _env_file=None,
        llm=LLMSettings(
            _env_file=None,
            llm_provider="gemini",
            gemma_model_name=primary,
            gemma_fallback_model_name=fallback,
            google_api_key=key,  # type: ignore[arg-type]
        ),
    )


@pytest.fixture(autouse=True)
def _clear_service_cache() -> None:
    get_gemma_service.cache_clear()
    yield
    get_gemma_service.cache_clear()


def _mock_client(text: str | None = "hello") -> MagicMock:
    mock_aio_models = SimpleNamespace(
        generate_content=AsyncMock(return_value=SimpleNamespace(text=text))
    )
    mock_client = MagicMock()
    mock_client.aio.models = mock_aio_models
    return mock_client


@pytest.mark.asyncio
async def test_generate_text_success() -> None:
    mock_client = _mock_client("plain text")
    with patch("app.llm.gemma_service.genai.Client", return_value=mock_client):
        svc = GemmaService(_settings())
        out = await svc.generate_text("hi", system="sys", thinking=False)

    assert out == "plain text"
    kwargs = mock_client.aio.models.generate_content.await_args.kwargs
    assert kwargs["model"] == "gemma-4-26b-a4b-it"
    assert kwargs["contents"] == ["hi"]
    assert kwargs["config"].system_instruction == "sys"
    level = kwargs["config"].thinking_config.thinking_level
    assert str(getattr(level, "value", level)).lower() == "minimal"


@pytest.mark.asyncio
async def test_generate_with_image_success() -> None:
    mock_client = _mock_client("seen image")
    fake_part = object()
    with (
        patch("app.llm.gemma_service.genai.Client", return_value=mock_client),
        patch("app.llm.gemma_service.types.Part.from_bytes", return_value=fake_part) as from_bytes,
    ):
        svc = GemmaService(_settings())
        out = await svc.generate_with_image(
            "describe",
            b"\x89PNG",
            "image/png",
            thinking=True,
        )

    assert out == "seen image"
    from_bytes.assert_called_once_with(data=b"\x89PNG", mime_type="image/png")
    kwargs = mock_client.aio.models.generate_content.await_args.kwargs
    assert kwargs["contents"] == [fake_part, "describe"]
    level = kwargs["config"].thinking_config.thinking_level
    assert str(getattr(level, "value", level)).lower() == "high"


@pytest.mark.asyncio
async def test_generate_structured_success_native_json() -> None:
    mock_client = _mock_client('{"title": "Lease", "score": 7}')
    with patch("app.llm.gemma_service.genai.Client", return_value=mock_client):
        svc = GemmaService(_settings())
        result = await svc.generate_structured("extract", TinySchema, thinking=False)

    assert result == TinySchema(title="Lease", score=7)
    kwargs = mock_client.aio.models.generate_content.await_args.kwargs
    assert kwargs["config"].response_mime_type == "application/json"
    assert kwargs["config"].response_schema is TinySchema


@pytest.mark.asyncio
async def test_generate_structured_retry_then_fail() -> None:
    mock_aio_models = SimpleNamespace(
        generate_content=AsyncMock(
            side_effect=[
                SimpleNamespace(text='{"title": "x"}'),  # missing score
                SimpleNamespace(text='{"title": "x"}'),  # still missing score
            ]
        )
    )
    mock_client = MagicMock()
    mock_client.aio.models = mock_aio_models

    with patch("app.llm.gemma_service.genai.Client", return_value=mock_client):
        svc = GemmaService(_settings())
        with pytest.raises(GemmaError, match="validation failed after retry"):
            await svc.generate_structured("extract", TinySchema)

    assert mock_aio_models.generate_content.await_count == 2


@pytest.mark.asyncio
async def test_fallback_model_used_after_primary_exhausts() -> None:
    mock_aio_models = SimpleNamespace(
        generate_content=AsyncMock(
            side_effect=[
                ConnectionError("connection reset"),
                ConnectionError("connection reset"),
                SimpleNamespace(text="from-fallback"),
            ]
        )
    )
    mock_client = MagicMock()
    mock_client.aio.models = mock_aio_models

    with patch("app.llm.gemma_service.genai.Client", return_value=mock_client):
        svc = GemmaService(_settings())
        out = await svc.generate_text("hello")

    assert out == "from-fallback"
    models = [
        call.kwargs["model"] for call in mock_aio_models.generate_content.await_args_list
    ]
    assert models == [
        "gemma-4-26b-a4b-it",
        "gemma-4-26b-a4b-it",
        "gemma-4-31b-it",
    ]


def test_get_gemma_service_is_cached() -> None:
    mock_client = _mock_client("x")
    with patch("app.llm.gemma_service.genai.Client", return_value=mock_client):
        a = get_gemma_service()
        b = get_gemma_service()
    assert a is b
