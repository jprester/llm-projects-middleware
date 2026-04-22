import os
import logging
import re
import time
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from mistralai import Mistral
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from the .env file
load_dotenv()

app = FastAPI()

# CORS configuration - restrict origins in production
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Initialize Mistral client once at startup
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    logger.warning("MISTRAL_API_KEY not found in environment variables")
    mistral_client = None
else:
    mistral_client = Mistral(api_key=api_key)

# Access token for protected endpoints
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", "")

# Provider API keys (server-side only)
PROVIDER_KEYS = {
    "anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
    "openrouter": os.getenv("OPENROUTER_API_KEY", ""),
}

PROVIDER_URLS = {
    "anthropic": "https://api.anthropic.com/v1/messages",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# In-memory cache for OpenRouter model catalog
_OR_MODELS_CACHE: dict = {"data": None, "ts": 0.0}
_OR_MODELS_TTL = 60 * 60  # 1 hour

# --- Request models ---

class MessageContent(BaseModel):
    role: str
    content: str
    type: Optional[str] = "text"


class CompletionRequest(BaseModel):
    content: List[MessageContent]
    model: Optional[str] = "mistral-tiny"
    provider: Optional[str] = "mistral"


class ImageMessage(BaseModel):
    role: str
    type: Optional[str] = "text"
    content: str


class ImageCompletionRequest(BaseModel):
    messages: List[ImageMessage]
    model: Optional[str] = "pixtral-12b-2409"
    provider: Optional[str] = "mistral"


# --- New models for v1/chat ---

class V1ChatRequest(BaseModel):
    provider: str
    model: str
    system: Optional[str] = None
    messages: list
    max_tokens: Optional[int] = 1024


# --- Auth dependency ---

async def verify_token(authorization: Optional[str] = Header(None)):
    if not ACCESS_TOKEN:
        return  # No token configured, skip auth
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.removeprefix("Bearer ")
    if token != ACCESS_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid access token")


# --- Helpers ---

def _parse_data_url(data_url: str) -> tuple[str, str]:
    match = re.match(r"data:([^;]+);base64,(.+)", data_url)
    if not match:
        raise ValueError("Invalid data URL")
    return match.group(1), match.group(2)


def _convert_messages_for_provider(messages: list, provider: str) -> list:
    """
    Convert frontend message format [{role, type, content}] to provider-native format.
    Groups consecutive user text+image messages into single messages with content arrays.
    """
    result = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "user")
        msg_type = msg.get("type", "text")

        if role == "user" and msg_type == "text":
            next_msg = messages[i + 1] if i + 1 < len(messages) else None
            if next_msg and next_msg.get("role") == "user" and next_msg.get("type") == "image":
                text_content = msg.get("content", "")
                image_content = next_msg.get("content", "")

                if provider == "anthropic":
                    media_type, base64_data = _parse_data_url(image_content)
                    content_blocks = [
                        {"type": "text", "text": text_content},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_data,
                            },
                        },
                    ]
                elif provider == "openrouter":
                    content_blocks = [
                        {"type": "text", "text": text_content},
                        {"type": "image_url", "image_url": {"url": image_content}},
                    ]
                else:  # mistral
                    content_blocks = [
                        {"type": "text", "text": text_content},
                        {"type": "image_url", "image_url": image_content},
                    ]
                result.append({"role": "user", "content": content_blocks})
                i += 2
                continue

        # Pass through non-grouped messages
        if msg_type == "image":
            image_content = msg.get("content", "")
            if provider == "anthropic":
                media_type, base64_data = _parse_data_url(image_content)
                content_blocks = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_data,
                        },
                    }
                ]
            elif provider == "openrouter":
                content_blocks = [
                    {"type": "image_url", "image_url": {"url": image_content}}
                ]
            else:  # mistral
                content_blocks = [
                    {"type": "image_url", "image_url": image_content}
                ]
            result.append({"role": "user", "content": content_blocks})
        else:
            result.append({"role": role, "content": msg.get("content", "")})
        i += 1

    return result


async def _proxy_anthropic_legacy(messages: list, model: str, api_key: str, max_tokens: int = 1024):
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(PROVIDER_URLS["anthropic"], headers=headers, json=body)

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    data = resp.json()
    text = "".join(b.get("text", "") for b in data.get("content", []))
    return {"response": text}


async def _proxy_openrouter_legacy(messages: list, model: str, api_key: str, max_tokens: int = 1024):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(PROVIDER_URLS["openrouter"], headers=headers, json=body)

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    return {"response": text}


# --- Endpoints ---

@app.get("/")
async def check():
    return {"message": "API is working!"}


@app.get("/providers/openrouter/models")
async def list_openrouter_models(_=Depends(verify_token)):
    """Return a normalized catalog of OpenRouter models. Cached for 1 hour."""
    now = time.time()
    if _OR_MODELS_CACHE["data"] and now - _OR_MODELS_CACHE["ts"] < _OR_MODELS_TTL:
        return _OR_MODELS_CACHE["data"]

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(OPENROUTER_MODELS_URL)
    except Exception as e:
        logger.error(f"OpenRouter models fetch failed: {e}")
        if _OR_MODELS_CACHE["data"]:
            return _OR_MODELS_CACHE["data"]
        raise HTTPException(status_code=502, detail="Failed to reach OpenRouter")

    if resp.status_code != 200:
        if _OR_MODELS_CACHE["data"]:
            return _OR_MODELS_CACHE["data"]
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    raw = resp.json().get("data", [])
    normalized = []
    for m in raw:
        mid = m.get("id")
        if not mid:
            continue
        pricing = m.get("pricing") or {}
        prompt_price = str(pricing.get("prompt") or "0")
        completion_price = str(pricing.get("completion") or "0")
        is_free = float(prompt_price) == 0.0 and float(completion_price) == 0.0
        architecture = m.get("architecture") or {}
        modalities = architecture.get("input_modalities") or []
        normalized.append({
            "id": mid,
            "name": m.get("name") or mid,
            "context_length": m.get("context_length"),
            "pricing": {"prompt": prompt_price, "completion": completion_price},
            "is_free": is_free,
            "supports_images": "image" in modalities,
        })

    payload = {"data": normalized, "fetched_at": int(now)}
    _OR_MODELS_CACHE["data"] = payload
    _OR_MODELS_CACHE["ts"] = now
    return payload


@app.post("/completion")
async def get_completion(request: CompletionRequest, _=Depends(verify_token)):
    if not request.content or len(request.content) == 0:
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    provider = request.provider or "mistral"
    messages = [{"role": msg.role, "content": msg.content} for msg in request.content]

    if provider == "mistral":
        if not mistral_client:
            raise HTTPException(status_code=503, detail="Mistral API not configured")
        try:
            chat_response = mistral_client.chat.complete(
                model=request.model,
                messages=messages,
            )
            return {"response": chat_response.choices[0].message.content}
        except Exception as e:
            logger.error(f"Completion error: {e}")
            raise HTTPException(status_code=500, detail="Failed to get completion from Mistral API")

    provider_key = PROVIDER_KEYS.get(provider)
    if not provider_key:
        raise HTTPException(status_code=400, detail=f"Provider '{provider}' is not configured")

    try:
        if provider == "anthropic":
            return await _proxy_anthropic_legacy(messages, request.model, provider_key)
        elif provider == "openrouter":
            return await _proxy_openrouter_legacy(messages, request.model, provider_key)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Completion error ({provider}): {e}")
        raise HTTPException(status_code=502, detail=f"Upstream API error: {str(e)}")


@app.post("/image-recognition")
async def image_recognition(data: ImageCompletionRequest, _=Depends(verify_token)):
    provider = data.provider or "mistral"

    if not data.messages or len(data.messages) == 0:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    raw_messages = [{"role": msg.role, "type": msg.type, "content": msg.content} for msg in data.messages]

    if provider == "mistral":
        if not mistral_client:
            raise HTTPException(status_code=503, detail="Mistral API not configured")

        converted = _convert_messages_for_provider(raw_messages, "mistral")
        try:
            chat_response = mistral_client.chat.complete(
                model=data.model,
                messages=converted
            )
            return {"response": chat_response.choices[0].message.content}
        except Exception as e:
            logger.error(f"Image recognition error: {e}")
            raise HTTPException(status_code=500, detail="Failed to process image recognition request")

    provider_key = PROVIDER_KEYS.get(provider)
    if not provider_key:
        raise HTTPException(status_code=400, detail=f"Provider '{provider}' is not configured")

    converted = _convert_messages_for_provider(raw_messages, provider)

    try:
        if provider == "anthropic":
            return await _proxy_anthropic_legacy(converted, data.model, provider_key)
        elif provider == "openrouter":
            return await _proxy_openrouter_legacy(converted, data.model, provider_key)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image recognition error ({provider}): {e}")
        raise HTTPException(status_code=502, detail=f"Upstream API error: {str(e)}")


# --- New multi-provider proxy endpoint ---

@app.post("/v1/chat")
async def v1_chat(request: V1ChatRequest, _=Depends(verify_token)):
    provider = request.provider
    provider_key = PROVIDER_KEYS.get(provider)

    if not provider_key:
        raise HTTPException(status_code=400, detail=f"Provider '{provider}' is not configured")

    try:
        if provider == "anthropic":
            return await _proxy_anthropic(request, provider_key)
        elif provider == "openrouter":
            return await _proxy_openrouter(request, provider_key)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"v1/chat error ({provider}): {e}")
        raise HTTPException(status_code=502, detail=f"Upstream API error: {str(e)}")


async def _proxy_anthropic(request: V1ChatRequest, api_key: str):
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    body = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "messages": request.messages,
    }
    if request.system:
        body["system"] = request.system

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(PROVIDER_URLS["anthropic"], headers=headers, json=body)

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    data = resp.json()
    text = "".join(b.get("text", "") for b in data.get("content", []))
    return {"response": text}


async def _proxy_openrouter(request: V1ChatRequest, api_key: str):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    messages = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    messages.extend(request.messages)

    body = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "messages": messages,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(PROVIDER_URLS["openrouter"], headers=headers, json=body)

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    return {"response": text}
