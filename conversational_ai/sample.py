import os
import uuid
import asyncio
import time
import logging
from enum import Enum
from typing import List, Dict, AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import gradio as gr
import openai
from anthropic import Anthropic
from google.generativeai import GenerativeModel
import google.generativeai as genai
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"

# Pydantic models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4"
    stream: bool = False

class ChatResponse(BaseModel):
    message: Message
    provider: LLMProvider
    model: str

# LLM Configuration
class LLMConfig:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # Initialize clients
        openai.api_key = self.openai_api_key
        genai.configure(api_key=self.gemini_api_key)
        self.anthropic_client = Anthropic(api_key=self.anthropic_api_key)

# FastAPI Application
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting FastAPI application...")
    app.state.llm_config = LLMConfig()
    yield
    # Shutdown
    logger.info("Shutting down FastAPI application...")
    app.state.llm_config = None

app = FastAPI(lifespan=lifespan)

# LLM Service
class LLMService:
    def __init__(self, config: LLMConfig):
        self.config = config

    async def generate_response(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        if request.provider == LLMProvider.OPENAI:
            return await self._openai_generate(request)
        elif request.provider == LLMProvider.ANTHROPIC:
            return await self._anthropic_generate(request)
        elif request.provider == LLMProvider.GEMINI:
            return await self._gemini_generate(request)
        else:
            raise HTTPException(status_code=400, detail="Invalid LLM provider")

    async def _openai_generate(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        try:
            if request.stream:
                async for chunk in await openai.ChatCompletion.acreate(
                    model=request.model,
                    messages=[msg.dict() for msg in request.messages],
                    stream=True
                ):
                    content = chunk.choices[0].delta.get("content", "")
                    if content:
                        yield content
            else:
                response = await openai.ChatCompletion.acreate(
                    model=request.model,
                    messages=[msg.dict() for msg in request.messages]
                )
                yield response.choices[0].message.content
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")

    async def _anthropic_generate(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        try:
            messages = [
                msg.dict() for msg in request.messages 
                if msg.role in ["user", "assistant"]
            ]
            if request.stream:
                with self.config.anthropic_client.messages.stream(
                    model=request.model,
                    messages=messages,
                    max_tokens=1024
                ) as stream:
                    for text in stream.text_stream:
                        yield text
            else:
                response = self.config.anthropic_client.messages.create(
                    model=request.model,
                    messages=messages,
                    max_tokens=1024
                )
                yield response.content[0].text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Anthropic error: {str(e)}")

    async def _gemini_generate(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        try:
            model = GenerativeModel(request.model)
            chat = model.start_chat(history=[
                {"role": msg.role, "parts": [msg.content]} 
                for msg in request.messages
            ])
            
            if request.stream:
                response = chat.send_message(request.messages[-1].content, stream=True)
                for chunk in response:
                    yield chunk.text
            else:
                response = chat.send_message(request.messages[-1].content)
                yield response.text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")

# FastAPI Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    llm_service = LLMService(app.state.llm_config)
    
    if request.stream:
        async def stream_response():
            async for chunk in llm_service.generate_response(request):
                yield chunk
        
        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream"
        )
    
    # Non-streaming response
    response_content = ""
    async for chunk in llm_service.generate_response(request):
        response_content += chunk
    
    return ChatResponse(
        message=Message(role="assistant", content=response_content),
        provider=request.provider,
        model=request.model
    )

# Gradio Interface
def create_gradio_interface():
    async def chat_with_ai(message: str, history: List[Dict], provider: str, model: str):
        provider_map = {
            "OpenAI": LLMProvider.OPENAI,
            "Anthropic": LLMProvider.ANTHROPIC,
            "Gemini": LLMProvider.GEMINI
        }
        
        # Convert history to messages
        messages = []
        for msg in history:
            messages.append(Message(role=msg["role"], content=msg["content"]))
        messages.append(Message(role="user", content=message))

        # Make API call
        import httpx
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    "http://localhost:8000/chat",
                    json={
                        "messages": [msg.dict() for msg in messages],
                        "provider": provider_map[provider].value,
                        "model": model,
                        "stream": True
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    content = ""
                    async for line in response.aiter_text():
                        content += line
                    logger.info(f"Received response: {content}")
                    return {"role": "assistant", "content": content}
                else:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    return {"role": "assistant", "content": f"Error: {response.status_code}"}
            except Exception as e:
                logger.error(f"Error connecting to FastAPI: {str(e)}")
                return {"role": "assistant", "content": f"Error connecting to server: {str(e)}"}

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Conversational AI Chat")
        
        with gr.Row():
            provider = gr.Dropdown(
                choices=["OpenAI", "Anthropic", "Gemini"],
                label="LLM Provider",
                value="OpenAI"
            )
            model = gr.Dropdown(
                choices=["gpt-4", "claude-3-opus", "gemini-pro"],
                label="Model",
                value="gpt-4"
            )
        
        chatbot = gr.Chatbot(type="messages")
        msg = gr.Textbox(placeholder="Type your message here...")
        
        with gr.Row():
            submit = gr.Button("Send")
            clear = gr.Button("Clear")
        
        submit.click(
            fn=chat_with_ai,
            inputs=[msg, chatbot, provider, model],
            outputs=chatbot
        )
        
        clear.click(
            fn=lambda: [],
            inputs=None,
            outputs=chatbot
        )
    
    return demo

# Main execution
if __name__ == "__main__":
    import uvicorn
    
    # Start FastAPI server in a separate thread
    def run_fastapi():
        logger.info("Starting FastAPI server on port 8000...")
        try:
            uvicorn.run(app, host="0.0.0.0", port=8000)
        except Exception as e:
            logger.error(f"FastAPI server failed to start: {str(e)}")
    
    import threading
    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.daemon = True
    fastapi_thread.start()
    
    # Wait a few seconds to ensure FastAPI server is up
    logger.info("Waiting for FastAPI server to start...")
    time.sleep(5)
    
    # Launch Gradio interface
    logger.info("Starting Gradio interface on port 7860...")
    try:
        demo = create_gradio_interface()
        demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    except Exception as e:
        logger.error(f"Gradio interface failed to start: {str(e)}")