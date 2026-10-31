"""Gemma LLM clients.

- get_chat_model(): LangChain BaseChatModel for ollama | vllm (existing agents).
- Gemma4Client: Gemma 4 via Google GenAI / Gemini API (Legal Document Intelligence
  demo). Uses google.genai with thinking, system_instruction, and image parts.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app.core.config import Settings

ThinkingLevel = Literal["high", "minimal"]

DEFAULT_GEMMA4_MODEL = "gemma-4-26b-a4b-it"
FALLBACK_GEMMA4_MODEL = "gemma-4-31b-it"


class GemmaAPIError(Exception):
    """Raised when the Google GenAI / Gemma 4 call fails (maps to HTTP 502)."""


class Gemma4Client:
    """Thin async wrapper around google.genai for Gemma 4.

    Reads GOOGLE_API_KEY from Settings.llm.google_api_key only — never hardcode.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.model = settings.llm.gemma_model_name or DEFAULT_GEMMA4_MODEL
        self.fallback_model = (
            settings.llm.gemma_fallback_model_name or FALLBACK_GEMMA4_MODEL
        )
        api_key = settings.llm.google_api_key
        if api_key is None or not api_key.get_secret_value().strip():
            raise GemmaAPIError(
                "GOOGLE_API_KEY is not set. Add it to your local .env "
                "(use placeholder <your-key> in committed files only)."
            )
        # Imported lazily so ollama/vllm unit tests need not import google.genai.
        from google import genai

        self._client = genai.Client(api_key=api_key.get_secret_value())

    async def generate(
        self,
        contents: Any,
        *,
        system_instruction: str | None = None,
        thinking_level: ThinkingLevel = "high",
        temperature: float = 0.2,
        response_mime_type: str | None = None,
        use_fallback: bool = False,
    ) -> str:
        """Call Gemma 4 generate_content; optionally retry on fallback model."""
        from google.genai import types

        model = self.fallback_model if use_fallback else self.model
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
            response_mime_type=response_mime_type,
        )
        try:
            response = await self._client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as exc:  # noqa: BLE001 — surface as GemmaAPIError for 502
            if not use_fallback and self.fallback_model != model:
                try:
                    return await self.generate(
                        contents,
                        system_instruction=system_instruction,
                        thinking_level=thinking_level,
                        temperature=temperature,
                        response_mime_type=response_mime_type,
                        use_fallback=True,
                    )
                except GemmaAPIError:
                    raise GemmaAPIError(f"Gemma 4 request failed: {exc}") from exc
            raise GemmaAPIError(f"Gemma 4 request failed: {exc}") from exc

        text = getattr(response, "text", None)
        if not text:
            raise GemmaAPIError("Gemma 4 returned an empty response")
        return text

    async def generate_text(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        thinking_level: ThinkingLevel = "high",
        response_mime_type: str | None = None,
    ) -> str:
        return await self.generate(
            prompt,
            system_instruction=system_instruction,
            thinking_level=thinking_level,
            response_mime_type=response_mime_type,
        )

    async def generate_with_image(
        self,
        prompt: str,
        *,
        image_bytes: bytes,
        mime_type: str,
        system_instruction: str | None = None,
        thinking_level: ThinkingLevel = "high",
        response_mime_type: str | None = None,
    ) -> str:
        """Multimodal call: inline image bytes + text prompt (Gemma 4 vision)."""
        from google.genai import types

        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        return await self.generate(
            [image_part, prompt],
            system_instruction=system_instruction,
            thinking_level=thinking_level,
            response_mime_type=response_mime_type,
        )


def get_gemma4_client(settings: Settings) -> Gemma4Client:
    """Factory for the Gemma 4 Google GenAI client used by the demo API."""
    return Gemma4Client(settings)


def get_chat_model(settings: Settings, *, temperature: float = 0.0) -> BaseChatModel:
    """LangChain chat model for ollama | vllm (Module 10 / existing agents)."""
    llm_settings = settings.llm

    if llm_settings.llm_provider == "ollama":
        return ChatOllama(
            model=llm_settings.gemma_model_name,
            base_url=llm_settings.ollama_base_url,
            temperature=temperature,
        )

    if llm_settings.llm_provider == "vllm":
        # vLLM exposes an OpenAI-compatible endpoint — ChatOpenAI pointed
        # at it works without a real OpenAI account; api_key is required
        # by the client but ignored by vLLM's own server.
        return ChatOpenAI(
            model=llm_settings.gemma_model_name,
            base_url=f"{llm_settings.vllm_base_url}/v1",
            api_key="not-needed",
            temperature=temperature,
        )

    if llm_settings.llm_provider == "gemini":
        raise ValueError(
            "LLM_PROVIDER=gemini uses Gemma4Client (Google GenAI), not "
            "get_chat_model(). Call get_gemma4_client(settings) instead."
        )

    raise ValueError(f"Unknown LLM provider: {llm_settings.llm_provider!r}")
