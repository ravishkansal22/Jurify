# Jurify — Legal Document Intelligence (Gemma 4)

Paste text or upload a photo/scan of a legal document (rental agreement, notice, contract). **Gemma 4** explains it in plain English or Hindi, extracts key clauses, flags risky clauses with the exact quoted passage and a reason, and answers questions grounded in the document.

> **Not legal advice.** Jurify is informational only. Consult a qualified lawyer for advice about your situation.

## Hackday demo note — in-memory store

**Analyses are kept in an in-memory Python `dict` for the demo.**  
No Postgres is required for `GET /health` or `POST /api/v1/documents/*`. Data is lost when the process restarts.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY=<your-key>  (never commit the real key)
```

Required `.env` keys for the demo:

```
LLM_PROVIDER=gemini
GEMMA_MODEL_NAME=gemma-4-26b-a4b-it
GEMMA_FALLBACK_MODEL_NAME=gemma-4-31b-it
GOOGLE_API_KEY=<your-key>
```

`.env` is gitignored (`git check-ignore .env` should match).

## Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Health: `GET http://localhost:8000/health` → `{"status":"ok","model":"gemma-4-26b-a4b-it"}`
- Docs: `http://localhost:8000/docs`

## API (frozen)

### `POST /api/v1/documents/analyze` (multipart/form-data)

| Field | Notes |
| --- | --- |
| `text` | optional string, max 20 000 chars |
| `file` | optional image `png` / `jpg` / `webp`, max 5 MB |
| `language` | `en` \| `hi` (default `en`) |

At least one of `text` or `file` is required.

### `POST /api/v1/documents/ask` (JSON)

```json
{"analysis_id": "<id from analyze>", "question": "...", "language": "en"}
```

Errors: `422` validation, `413` file too large, `404` unknown `analysis_id`, `502` when Gemma fails (`{"detail": "..."}`).

## Tests

All LLM calls are mocked in tests:

```bash
pytest -q
```

## Gemma 4

Uses the Google GenAI SDK (`google-genai`) against model ids:

- `gemma-4-26b-a4b-it` (default, faster)
- `gemma-4-31b-it` (automatic fallback)

Thinking is enabled (`thinking_level="high"`) for analyze/ask. Image uploads use inline bytes via `types.Part.from_bytes`.
