"""POST /api/v1/documents/analyze and /ask (Me 2)."""

from __future__ import annotations

import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.v1.settings_local import analyze_thinking_enabled
from app.api.v1.store import (
    AnalysisStore,
    build_qa_context_from_analysis,
    filter_verified_quotes,
    filter_verified_risk_flags,
    get_analysis_store,
)
from app.core.config import get_settings
from app.llm.gemma_service import GemmaError, GemmaService, get_gemma_service
from app.prompts.doc_analyze import SYSTEM_PROMPT as ANALYZE_SYSTEM
from app.prompts.doc_analyze import build_analyze_prompt
from app.prompts.doc_ask import SYSTEM_PROMPT as ASK_SYSTEM
from app.prompts.doc_ask import build_ask_prompt
from app.schemas.document_analysis import (
    NOT_LEGAL_ADVICE_DISCLAIMER,
    AnalyzeLLMResult,
    AnalyzeResponse,
    AskLLMResult,
    AskRequest,
    AskResponse,
)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

MAX_TEXT_CHARS = 20_000
MAX_IMAGE_BYTES = 5 * 1024 * 1024
_UNPROCESSABLE = getattr(
    status, "HTTP_422_UNPROCESSABLE_CONTENT", status.HTTP_422_UNPROCESSABLE_ENTITY
)
_TOO_LARGE = getattr(
    status, "HTTP_413_CONTENT_TOO_LARGE", status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
)
ALLOWED_IMAGE_TYPES = {
    "image/png": "image/png",
    "image/jpeg": "image/jpeg",
    "image/jpg": "image/jpeg",
    "image/webp": "image/webp",
}


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_document(
    gemma: Annotated[GemmaService, Depends(get_gemma_service)],
    store: Annotated[AnalysisStore, Depends(get_analysis_store)],
    text: Annotated[str | None, Form()] = None,
    language: Annotated[Literal["en", "hi"], Form()] = "en",
    file: Annotated[UploadFile | None, File()] = None,
) -> AnalyzeResponse:
    if (text is None or not text.strip()) and file is None:
        raise HTTPException(
            status_code=_UNPROCESSABLE,
            detail="At least one of text or file is required",
        )

    cleaned_text: str | None = None
    if text is not None and text.strip():
        cleaned_text = text.strip()
        if len(cleaned_text) > MAX_TEXT_CHARS:
            raise HTTPException(
                status_code=_UNPROCESSABLE,
                detail=f"text must be at most {MAX_TEXT_CHARS} characters",
            )

    image_bytes: bytes | None = None
    image_mime: str | None = None
    if file is not None:
        content_type = (file.content_type or "").lower()
        if content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=_UNPROCESSABLE,
                detail="file must be an image of type png, jpg, or webp",
            )
        image_mime = ALLOWED_IMAGE_TYPES[content_type]
        image_bytes = await file.read()
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=_TOO_LARGE,
                detail="file exceeds maximum size of 5 MB",
            )
        if not image_bytes:
            raise HTTPException(status_code=_UNPROCESSABLE, detail="file is empty")

    prompt = build_analyze_prompt(
        language=language,
        has_image=image_bytes is not None,
        text=cleaned_text,
    )
    image_arg = (
        (image_bytes, image_mime) if image_bytes is not None and image_mime else None
    )

    try:
        llm_result = await gemma.generate_structured(
            prompt,
            AnalyzeLLMResult,
            system=ANALYZE_SYSTEM,
            image=image_arg,
            thinking=analyze_thinking_enabled(),
        )
    except GemmaError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    verified_flags = filter_verified_risk_flags(llm_result.risk_flags, cleaned_text)
    settings = get_settings()
    # Assign the real id before store.write so the returned/stored ids never diverge
    # (never expose a "pending" placeholder to clients).
    analysis_id = str(uuid.uuid4())
    draft = AnalyzeResponse(
        analysis_id=analysis_id,
        document_type=llm_result.document_type,
        plain_summary=llm_result.plain_summary,
        key_clauses=llm_result.key_clauses,
        risk_flags=verified_flags,
        questions_to_ask=llm_result.questions_to_ask[:5],
        language=language,
        model=settings.llm.gemma_model_name,
        disclaimer=NOT_LEGAL_ADVICE_DISCLAIMER,
    )

    if cleaned_text:
        source_for_qa = cleaned_text
    else:
        source_for_qa = build_qa_context_from_analysis(draft)

    return store.create(
        source_text=source_for_qa, analysis=draft, analysis_id=analysis_id
    )


@router.post("/ask", response_model=AskResponse)
async def ask_about_document(
    body: AskRequest,
    gemma: Annotated[GemmaService, Depends(get_gemma_service)],
    store: Annotated[AnalysisStore, Depends(get_analysis_store)],
) -> AskResponse:
    stored = store.get(body.analysis_id)
    if stored is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown analysis_id: {body.analysis_id}",
        )

    prompt = build_ask_prompt(
        language=body.language,
        question=body.question,
        document_context=stored.source_text,
    )
    try:
        llm_result = await gemma.generate_structured(
            prompt,
            AskLLMResult,
            system=ASK_SYSTEM,
            thinking=False,
        )
    except GemmaError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    quotes = filter_verified_quotes(llm_result.evidence_quotes, stored.source_text)
    answer = llm_result.answer
    confidence = llm_result.confidence
    if not quotes and "not stated" not in answer.casefold():
        # Prefer honesty when supporting quotes failed verification.
        if not filter_verified_quotes([answer], stored.source_text):
            # keep model answer but lower confidence if quotes were dropped
            confidence = "low" if confidence == "high" else confidence

    return AskResponse(
        answer=answer,
        evidence_quotes=quotes,
        confidence=confidence,
        disclaimer=NOT_LEGAL_ADVICE_DISCLAIMER,
    )
