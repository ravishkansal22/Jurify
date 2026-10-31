"""Document analyze / ask routes (frozen Gemma 4 demo API contract)."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.deps import get_document_intel_service
from app.llm.gemma_client import GemmaAPIError
from app.schemas.legal_document_intel import (
    DocumentAnalyzeResponse,
    DocumentAskRequest,
    DocumentAskResponse,
)
from app.services.legal_document_intel_service import (
    AnalysisNotFoundError,
    LegalDocumentIntelService,
)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

MAX_TEXT_CHARS = 20_000
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB per API contract
# Starlette renamed UNPROCESSABLE_ENTITY → UNPROCESSABLE_CONTENT.
_UNPROCESSABLE = getattr(
    status, "HTTP_422_UNPROCESSABLE_CONTENT", status.HTTP_422_UNPROCESSABLE_ENTITY
)
ALLOWED_IMAGE_TYPES = {
    "image/png": "image/png",
    "image/jpeg": "image/jpeg",
    "image/jpg": "image/jpeg",
    "image/webp": "image/webp",
}


@router.post(
    "/analyze",
    response_model=DocumentAnalyzeResponse,
    status_code=status.HTTP_200_OK,
)
async def analyze_document(
    service: Annotated[LegalDocumentIntelService, Depends(get_document_intel_service)],
    text: Annotated[str | None, Form()] = None,
    language: Annotated[Literal["en", "hi"], Form()] = "en",
    file: Annotated[UploadFile | None, File()] = None,
) -> DocumentAnalyzeResponse:
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
                status_code=getattr(
                    status,
                    "HTTP_413_CONTENT_TOO_LARGE",
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                ),
                detail="file exceeds maximum size of 5 MB",
            )
        if not image_bytes:
            raise HTTPException(
                status_code=_UNPROCESSABLE,
                detail="file is empty",
            )

    try:
        return await service.analyze(
            text=cleaned_text,
            language=language,
            image_bytes=image_bytes,
            image_mime=image_mime,
        )
    except GemmaAPIError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc


@router.post(
    "/ask",
    response_model=DocumentAskResponse,
    status_code=status.HTTP_200_OK,
)
async def ask_about_document(
    body: DocumentAskRequest,
    service: Annotated[LegalDocumentIntelService, Depends(get_document_intel_service)],
) -> DocumentAskResponse:
    try:
        return await service.ask(
            analysis_id=body.analysis_id,
            question=body.question,
            language=body.language,
        )
    except AnalysisNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown analysis_id: {body.analysis_id}",
        ) from exc
    except GemmaAPIError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
