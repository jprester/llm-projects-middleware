# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

A FastAPI middleware that proxies chat and image-recognition requests to Mistral, Anthropic, and OpenRouter behind a single authenticated API. It exists so the frontend never holds provider API keys.

## Run / verify

```bash
source .venv/bin/activate
uvicorn main:app --reload
```

Quick import smoke test (catches most refactor regressions):

```bash
.venv/bin/python -c "import main; print([r.path for r in main.app.routes])"
```

There is no test suite. After non-trivial changes run the smoke test above and, when possible, exercise the affected endpoint with `curl`.

## Module boundaries

- `config.py` — all env vars via `pydantic-settings`. Add new settings here, never read `os.getenv` elsewhere.
- `models.py` — Pydantic request bodies. Response shapes are inline dicts in routes.
- `services.py` — shared `httpx.AsyncClient`, `convert_messages_for_provider`, and `proxy_request`. Anything provider-specific lives here.
- `routes.py` — endpoint handlers + `verify_token` dependency + OpenRouter models cache. Keep handlers thin; push provider logic into `services`.
- `main.py` — app construction only: lifespan, CORS, global exception handler, router inclusion. Don't add endpoints here.

## Provider conventions

- Provider API keys are resolved by name: `services.proxy_request` looks up `settings.<PROVIDER>_API_KEY` via `getattr`. New providers need a matching `<NAME>_API_KEY` field in `config.Settings` and a URL entry in `services.PROVIDER_URLS`.
- Image payloads differ per provider — Anthropic takes base64 via `source`, OpenRouter takes `image_url: {url}`, Mistral takes `image_url` as a plain string. `convert_messages_for_provider` is the single place that knows this.
- The shared `services.http_client` is closed in the FastAPI lifespan; don't create per-request `AsyncClient` instances.

## Error handling convention

- Input validation errors → `HTTPException(400)`.
- Missing/invalid bearer token → `401` / `403` from `verify_token`.
- Mistral SDK failures → caught in the route, logged, re-raised as `500` with a user-safe message.
- Anthropic / OpenRouter failures → `services.proxy_request` raises `HTTPException` with the upstream status; routes also wrap non-HTTP exceptions as `502 "Upstream API error: ..."`.
- The global handler in `main.py` is a last-resort `500`; prefer raising a specific `HTTPException` at the boundary.

## Things to avoid

- Do not read env vars outside `config.py`.
- Do not register routes in `main.py`; add them to `routes.py`.
- Do not add a second `CORSMiddleware` — it's easy to duplicate when editing `main.py`.
- Do not instantiate new `httpx.AsyncClient` objects in handlers; reuse `services.http_client`.
- When adding a dependency, update `requirements.txt` in the same change (e.g. `pydantic-settings` was missed during an earlier refactor).
