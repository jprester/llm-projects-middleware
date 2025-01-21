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

@app.post("/image-recognition")
async def image_recognition():
    # Path to your image
    image_path = "./janko.jpg"
    print(f"Image path: {image_path}")

    # Getting the base64 string
    base64_image = await encode_image(image_path)

    # Specify model
    model = "pixtral-12b-2409"

    # Initialize the Mistral client
    client = Mistral(api_key=api_key)

    if not base64_image:
        raise HTTPException(status_code=400, detail="Image not found")
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
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