import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mistralai import Mistral
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional

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
