from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference import generate_response


# Initialize FastAPI
app = FastAPI(
    title="My Custom GPT API",
    description="My own GPT model API",
    version="1.0"
)


# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request Schema
class ChatRequest(BaseModel):
    message: str


# Home Route
@app.get("/")
def home():
    return {
        "message": "My Custom GPT API is running!"
    }


# Health Check
@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# Chat API
@app.post("/chat")
def chat(request: ChatRequest):

    response = generate_response(
        request.message,
        max_new_tokens=200
    )

    return {
        "question": request.message,
        "answer": response
    }
