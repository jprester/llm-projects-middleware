import os
import logging
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

# --- Existing models (unchanged) ---

class MessageContent(BaseModel):
    role: str
    content: str
    type: Optional[str] = "text"


class CompletionRequest(BaseModel):
    content: List[MessageContent]
    model: Optional[str] = "mistral-tiny"


class ImageMessage(BaseModel):
    content: str


class ImageCompletionRequest(BaseModel):
    messages: List[ImageMessage]
    model: Optional[str] = "pixtral-12b-2409"


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


# --- Existing endpoints (unchanged) ---

@app.get("/")
async def check():
    return {"message": "API is working!"}


@app.post("/completion")
async def get_completion(request: CompletionRequest):
    if not request.content or len(request.content) == 0:
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    if not mistral_client:
        raise HTTPException(status_code=503, detail="Mistral API not configured")

    # Convert Pydantic models to dicts for the API
    messages = [{"role": msg.role, "content": msg.content} for msg in request.content]

    try:
        chat_response = mistral_client.chat.complete(
            model=request.model,
            messages=messages,
        )
        return {"response": chat_response.choices[0].message.content}
    except Exception as e:
        logger.error(f"Completion error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get completion from Mistral API")


@app.post("/image-recognition")
async def image_recognition(data: ImageCompletionRequest):
    if not mistral_client:
        raise HTTPException(status_code=503, detail="Mistral API not configured")

    if not data.messages or len(data.messages) < 2:
        raise HTTPException(status_code=400, detail="At least two content items are required (text prompt and image)")

    if not data.messages[1].content:
        raise HTTPException(status_code=400, detail="Image data not found")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": data.messages[0].content
                },
                {
                    "type": "image_url",
                    "image_url": data.messages[1].content
                }
            ]
        }
    ]

    try:
        chat_response = mistral_client.chat.complete(
            model=data.model,
            messages=messages
        )
        return {"response": chat_response.choices[0].message.content}
    except Exception as e:
        logger.error(f"Image recognition error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process image recognition request")


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
