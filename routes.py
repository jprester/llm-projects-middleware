import time
import logging
from fastapi import APIRouter, HTTPException, Depends, Header
from typing import List, Optional
from mistralai import Mistral
from config import settings
from models import CompletionRequest, ImageCompletionRequest, V1ChatRequest
import services

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize Mistral client
mistral_client = Mistral(api_key=settings.MISTRAL_API_KEY) if settings.MISTRAL_API_KEY else None

# In-memory cache for OpenRouter model catalog
_OR_MODELS_CACHE: dict = {"data": None, "ts": 0.0}
_OR_MODELS_TTL = 60 * 60  # 1 hour
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

async def verify_token(authorization: Optional[str] = Header(None)):
    if not settings.ACCESS_TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.removeprefix("Bearer ")
    if token != settings.ACCESS_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid access token")

@router.get("/")
async def check():
    return {"message": "API is working!"}

@router.get("/providers/openrouter/models")
async def list_openrouter_models(_=Depends(verify_token)):
    now = time.time()
    if _OR_MODELS_CACHE["data"] and now - _OR_MODELS_CACHE["ts"] < _OR_MODELS_TTL:
        return _OR_MODELS_CACHE["data"]
    
    try:
        resp = await services.http_client.get(OPENROUTER_MODELS_URL)
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
        if not mid: continue
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

@router.post("/completion")
async def get_completion(request: CompletionRequest, _=Depends(verify_token)):
    if not request.content:
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    provider = request.provider or "mistral"
    messages = [{"role": msg.role, "content": msg.content} for msg in request.content]

    if provider == "mistral":
        if not mistral_client:
            raise HTTPException(status_code=503, detail="Mistral API not configured")
        try:
            chat_response = mistral_client.chat.complete(model=request.model, messages=messages)
            return {"response": chat_response.choices[0].message.content}
        except Exception as e:
            logger.error(f"Completion error: {e}")
            raise HTTPException(status_code=500, detail="Failed to get completion from Mistral API")

    try:
        return await services.proxy_request(provider, request.model, messages)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Completion error ({provider}): {e}")
        raise HTTPException(status_code=502, detail=f"Upstream API error: {str(e)}")

@router.post("/image-recognition")
async def image_recognition(data: ImageCompletionRequest, _=Depends(verify_token)):
    provider = data.provider or "mistral"

    if not data.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    raw_messages = [{"role": msg.role, "type": msg.type, "content": msg.content} for msg in data.messages]

    if provider == "mistral":
        if not mistral_client:
            raise HTTPException(status_code=503, detail="Mistral API not configured")
        converted = services.convert_messages_for_provider(raw_messages, "mistral")
        try:
            chat_response = mistral_client.chat.complete(model=data.model, messages=converted)
            return {"response": chat_response.choices[0].message.content}
        except Exception as e:
            logger.error(f"Image recognition error: {e}")
            raise HTTPException(status_code=500, detail="Failed to process image recognition request")

    converted = services.convert_messages_for_provider(raw_messages, provider)
    try:
        return await services.proxy_request(provider, data.model, converted)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image recognition error ({provider}): {e}")
        raise HTTPException(status_code=502, detail=f"Upstream API error: {str(e)}")

@router.post("/v1/chat")
async def v1_chat(request: V1ChatRequest, _=Depends(verify_token)):
    try:
        return await services.proxy_request(request.provider, request.model, request.messages, request.max_tokens, request.system)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"v1/chat error ({request.provider}): {e}")
        raise HTTPException(status_code=502, detail=f"Upstream API error: {str(e)}")
