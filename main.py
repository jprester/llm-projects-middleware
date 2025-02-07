import os
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
    if not request.content or len(request.content) == 0:
        raise HTTPException(status_code=400, detail="Content cannot be empty")
    if not api_key:
        return {"error": "API key not found"}
    if not request.model:
        request.model = "mistral-tiny"

    client = Mistral(api_key=api_key)

    try:
        chat_response = client.chat.complete(
            model=request.model,
            messages=request.content,
        )
        return {"response": chat_response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}


async def encode_image(image_path):
    """Encode the image to base64."""
    print(f"Image path: {image_path}")
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:  # Added general exception handling
        print(f"Error: {e}")
        return None

class ImageCompletionRequest(BaseModel):
    messages: list
    model: str = None

@app.post("/image-recognition")
async def image_recognition(data: ImageCompletionRequest):
    if not data.model:
        data.model = "pixtral-12b-2409"
    if not data.messages or len(data.messages) < 2:
        raise HTTPException(status_code=400, detail="At least two content items are required")

    if hasattr(data, 'messages') and isinstance(data.messages, list) and len(data.messages) > 0:
        print("messages[0]: ", data.messages[0]["content"])
    else:
        print("The required property does not exist or is not accessible.")

    # Specify model
    model = data.model

    # Initialize the Mistral client 
    client = Mistral(api_key=api_key)

    if not data.messages[1]["content"]:
        raise HTTPException(status_code=400, detail="Image not found")
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": data.messages[0]["content"]
                    },
                    {
                        "type": "image_url",
                        "image_url": f"{data.messages[1]["content"]}"
                    }
                ]
            }
        ]

        # Get the chat response
        chat_response = client.chat.complete(
            model=model,
            messages=messages
        )

        # Print the content of the response
        print(chat_response.choices[0].message.content)

        return {"response": chat_response.choices[0].message.content}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}