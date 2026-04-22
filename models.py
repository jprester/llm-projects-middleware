from pydantic import BaseModel
from typing import List, Optional

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


class V1ChatRequest(BaseModel):
    provider: str
    model: str
    system: Optional[str] = None
    messages: list
    max_tokens: Optional[int] = 1024
