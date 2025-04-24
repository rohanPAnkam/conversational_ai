from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict
import logging
from conversational_ai.models.llm_config import LLMConfig, LLMProvider
from conversational_ai.models.llm_service import LLMService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Dict]
    provider: str = "openai"
    model: str = "gpt-4"
    stream: bool = False

class ChatResponse(BaseModel):
    message: Message
    provider: str
    model: str

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    logger.info("Starting FastAPI application...")
    app.state.llm_config = LLMConfig()
    logger.info("FastAPI application startup complete.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down FastAPI application...")
    app.state.llm_config = None

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint called.")
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    llm_service = LLMService(app.state.llm_config)
    provider_map = {"openai": "OPENAI", "llama": "LLAMA", "gemini": "GEMINI"}
    provider = getattr(LLMProvider, provider_map.get(request.provider.upper(), "OPENAI"))

    if request.stream:
        async def stream_response():
            # Await the generate_response coroutine to get the async iterator
            async for chunk in await llm_service.generate_response(
                request.messages, provider, request.model, request.stream
            ):
                yield chunk
        
        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream"
        )
    
    response_content = ""
    # Await the generate_response coroutine to get the async iterator
    async for chunk in await llm_service.generate_response(
        request.messages, provider, request.model, request.stream
    ):
        response_content += chunk
    
    return ChatResponse(
        message=Message(role="assistant", content=response_content),
        provider=request.provider,
        model=request.model
    )