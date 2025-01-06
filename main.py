import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mistralai import Mistral
from dotenv import load_dotenv
from pydantic import BaseModel

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load environment variables from the .env file
load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

@app.get("/")
async def check():
    return {"message": "API is working!"}

class CompletionRequest(BaseModel):
    content: list
    model: str = None

@app.post("/completion")
async def get_completion(request: CompletionRequest):
    if not request.content:
        return {"error": "Content cannot be empty"}
    if not api_key:
        return {"error": "API key not found"}

    model = request.model if request.model else "mistral-tiny"
    client = Mistral(api_key=api_key)

    try:
        chat_response = client.chat.complete(
            model=model,
            messages=request.content,
        )
        return {"response": chat_response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}