from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI
from pydantic import BaseModel
from agent import run_agent


app = FastAPI()

class ChatRequest(BaseModel):
    history: list[dict]

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


@app.get("/health")
def health():
    return {"status": "OK"}


@app.post("/chat",response_model = ChatResponse)
def chat(request: ChatRequest):
    result = run_agent(request.history)
    return result