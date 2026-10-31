"""Gemma 4 service (Me 1) via the Google GenAI SDK.

Contract for Me 2: ``generate_text``, ``generate_with_image``,
``generate_structured``, ``GemmaError``, and ``get_gemma_service``.
"""

from __future__ import annotations

import asyncio
import logging
import re
from functools import lru_cache
from typing import Any, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)

# Module-level note filled by the last successful structured call (for live checks).
LAST_JSON_METHOD: str | None = None


class GemmaError(Exception):
    """Raised on any API failure or unparseable output."""


def _is_transient(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    markers = (
        "timeout",
        "timed out",
        "temporarily",
        "unavailable",
        "429",
        "500",
        "502",
        "503",
        "504",
        "resourceexhausted",
        "internal",
        "connection",
        "reset",
    )
    return any(m in name or m in text for m in markers)


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if "```" not in cleaned:
        return cleaned
    # Prefer fenced JSON block when present.
    if "```json" in cleaned.lower():
        lower = cleaned.lower()
        start = lower.index("```json") + len("```json")
        rest = cleaned[start:]
        end = rest.find("```")
        if end != -1:
            return rest[:end].strip()
    parts = cleaned.split("```")
    if len(parts) >= 3:
        return parts[1].strip()
    return _FENCE_RE.sub("", cleaned).strip()


class GemmaService:
    """Async Gemma 4 client used by the Legal Document Intelligence API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        api_key = None
        if self.settings.llm.google_api_key is not None:
            api_key = self.settings.llm.google_api_key.get_secret_value()
        if not api_key or not str(api_key).strip():
            raise GemmaError(
                "GOOGLE_API_KEY is not set. Add it to your local .env "
                "(placeholders only in .env.example)."
            )
        self._client = genai.Client(api_key=api_key)
        self.primary_model = self.settings.llm.gemma_model_name
        self.fallback_model = self.settings.llm.gemma_fallback_model

    def _build_config(
        self,
        *,
        system: str | None,
        thinking: bool,
        response_mime_type: str | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> types.GenerateContentConfig:
        thinking_level = "high" if thinking else "minimal"
        kwargs: dict[str, Any] = {
            "temperature": 0.2,
            "thinking_config": types.ThinkingConfig(thinking_level=thinking_level),
        }
        if system:
            kwargs["system_instruction"] = system
        if response_mime_type:
            kwargs["response_mime_type"] = response_mime_type
        if response_schema is not None:
            kwargs["response_schema"] = response_schema
        return types.GenerateContentConfig(**kwargs)

    async def _generate_once(
        self,
        model: str,
        contents: Any,
        config: types.GenerateContentConfig,
    ) -> str:
        response = await self._client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        text = getattr(response, "text", None)
        if not text:
            raise GemmaError("Gemma returned an empty response")
        return text

    async def _call_with_retry_and_fallback(
        self,
        contents: Any,
        *,
        system: str | None,
        thinking: bool,
        response_mime_type: str | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> str:
        """Retry once on transient errors, then try the fallback model once."""
        config = self._build_config(
            system=system,
            thinking=thinking,
            response_mime_type=response_mime_type,
            response_schema=response_schema,
        )
        models = [self.primary_model]
        if self.fallback_model and self.fallback_model != self.primary_model:
            models.append(self.fallback_model)

        last_exc: BaseException | None = None
        for model_index, model in enumerate(models):
            attempts = 2 if model_index == 0 else 1
            for attempt in range(attempts):
                try:
                    logger.debug(
                        "gemma_generate model=%s attempt=%s",
                        model,
                        attempt + 1,
                    )
                    return await self._generate_once(model, contents, config)
                except GemmaError as exc:
                    last_exc = exc
                    # Empty / hard failure: do not treat as transient retry fuel
                    # unless the message looks transient.
                    if attempt == 0 and model_index == 0 and _is_transient(exc):
                        await asyncio.sleep(0.25)
                        continue
                    break
                except Exception as exc:  # noqa: BLE001 — map to GemmaError
                    last_exc = exc
                    logger.warning(
                        "gemma_generate_failed model=%s attempt=%s err_type=%s",
                        model,
                        attempt + 1,
                        type(exc).__name__,
                    )
                    if attempt == 0 and model_index == 0 and _is_transient(exc):
                        await asyncio.sleep(0.25)
                        continue
                    break

        raise GemmaError(f"Gemma generation failed: {last_exc}") from last_exc

    async def generate_text(
        self,
        prompt: str,
        *,
        system: str | None = None,
        thinking: bool = False,
    ) -> str:
        try:
            return await self._call_with_retry_and_fallback(
                [prompt],
                system=system,
                thinking=thinking,
            )
        except GemmaError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise GemmaError(str(exc)) from exc

    async def generate_with_image(
        self,
        prompt: str,
        image_bytes: bytes,
        mime_type: str,
        *,
        system: str | None = None,
        thinking: bool = False,
    ) -> str:
        try:
            image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            return await self._call_with_retry_and_fallback(
                [image_part, prompt],
                system=system,
                thinking=thinking,
            )
        except GemmaError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise GemmaError(str(exc)) from exc

    async def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        image: tuple[bytes, str] | None = None,
        thinking: bool = False,
    ) -> T:
        global LAST_JSON_METHOD

        contents: list[Any] = []
        if image is not None:
            img_bytes, mime = image
            contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))

        json_prompt = (
            f"{prompt}\n\n"
            "Respond with valid JSON only that matches the required schema. "
            "No markdown, no commentary."
        )
        contents.append(json_prompt)
        json_system = ((system or "").rstrip() + "\nOutput valid JSON only.").strip()

        text: str | None = None
        json_method: str | None = None

        # Method A: native JSON mime + response_schema (preferred).
        # Live Gemma 4 currently 500s on response_schema; we then fall back.
        try:
            text = await self._generate_once(
                self.primary_model,
                contents,
                self._build_config(
                    system=json_system,
                    thinking=thinking,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            json_method = "response_mime_type+response_schema"
        except Exception as exc:  # noqa: BLE001 — feature may be unsupported on Gemma 4
            logger.warning(
                "structured_native_json_rejected err_type=%s; falling back to prompt JSON",
                type(exc).__name__,
            )
            # Method B: prompt-only JSON + fence stripping (works on Gemma 4).
            text = await self._call_with_retry_and_fallback(
                contents,
                system=json_system,
                thinking=thinking,
            )
            json_method = "prompt_json+fence_strip"

        cleaned = _strip_code_fences(text)
        try:
            result = schema.model_validate_json(cleaned)
            LAST_JSON_METHOD = json_method
            logger.info("structured_json_method=%s", json_method)
            return result
        except ValidationError as val_err:
            logger.warning(
                "structured_validation_failed; retrying once with error feedback"
            )
            retry_prompt = (
                f"{json_prompt}\n\n"
                f"Previous JSON failed validation: {val_err}\n"
                f"Invalid output:\n{cleaned}\n"
                "Correct it and output valid JSON only."
            )
            retry_contents = list(contents[:-1]) + [retry_prompt]
            try:
                if json_method == "response_mime_type+response_schema":
                    retry_text = await self._call_with_retry_and_fallback(
                        retry_contents,
                        system=json_system,
                        thinking=thinking,
                        response_mime_type="application/json",
                        response_schema=schema,
                    )
                else:
                    retry_text = await self._call_with_retry_and_fallback(
                        retry_contents,
                        system=json_system,
                        thinking=thinking,
                    )
                cleaned_retry = _strip_code_fences(retry_text)
                result = schema.model_validate_json(cleaned_retry)
                LAST_JSON_METHOD = json_method
                logger.info("structured_json_method=%s (after_validation_retry)", json_method)
                return result
            except Exception as retry_err:  # noqa: BLE001
                raise GemmaError(
                    f"Structured validation failed after retry: {retry_err}"
                ) from retry_err


@lru_cache
def get_gemma_service() -> GemmaService:
    """FastAPI dependency — cached singleton GemmaService."""
    return GemmaService()
