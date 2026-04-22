# LLM Middleware

Backend middleware for the LLM project. A FastAPI service that proxies chat and image-recognition requests to multiple LLM providers (Mistral, Anthropic, OpenRouter) behind a single authenticated API.

## Features

- **Multi-provider chat** via `/completion` and `/v1/chat` (Mistral, Anthropic, OpenRouter)
- **Image recognition** via `/image-recognition` with per-provider message conversion (base64 for Anthropic, URL for OpenRouter/Mistral)
- **OpenRouter model catalog** via `/providers/openrouter/models`, cached for 1 hour
- **Bearer-token auth** on all data endpoints (optional — enabled when `ACCESS_TOKEN` is set)
- **CORS** restricted to configured origins

## Project layout

```
main.py       # FastAPI app, CORS, lifespan, global exception handler
config.py     # Settings loaded from .env via pydantic-settings
models.py     # Pydantic request models
routes.py     # Endpoints and auth dependency
services.py   # Shared httpx client, message conversion, provider proxy
```

## Installation

1. Clone the repository
2. Create and activate a virtual environment: `python -m venv .venv && source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Create a `.env` file (see below)
5. Run the server: `uvicorn main:app --reload`

## Configuration

Set these in `.env`:

| Variable              | Required | Default                                                  | Purpose                                |
|-----------------------|----------|----------------------------------------------------------|----------------------------------------|
| `ALLOWED_ORIGINS`     | no       | `http://localhost:5173,http://127.0.0.1:5173`            | Comma-separated CORS origins           |
| `ACCESS_TOKEN`        | no       | `""` (auth disabled when empty)                          | Bearer token required on endpoints     |
| `MISTRAL_API_KEY`     | no       | —                                                        | Enables Mistral provider               |
| `ANTHROPIC_API_KEY`   | no       | —                                                        | Enables Anthropic provider             |
| `OPENROUTER_API_KEY`  | no       | —                                                        | Enables OpenRouter provider            |

A provider is considered unconfigured if its key is missing; requests targeting it return `400`.

## Endpoints

| Method | Path                             | Purpose                                                             |
|--------|----------------------------------|---------------------------------------------------------------------|
| GET    | `/`                              | Health check                                                        |
| GET    | `/providers/openrouter/models`   | Normalized, cached OpenRouter model catalog                         |
| POST   | `/completion`                    | Text chat completion (Mistral / Anthropic / OpenRouter)             |
| POST   | `/image-recognition`             | Multimodal request with text + image                                |
| POST   | `/v1/chat`                       | Generic multi-provider chat proxy with optional `system` prompt     |

All data endpoints require `Authorization: Bearer <ACCESS_TOKEN>` when `ACCESS_TOKEN` is set.

## Error handling

- `400` — invalid input (empty messages, unknown/unconfigured provider)
- `401` / `403` — missing or invalid bearer token
- `500` — Mistral SDK failure
- `502` — upstream HTTP failure from Anthropic / OpenRouter
- `503` — Mistral provider selected but `MISTRAL_API_KEY` is not set
