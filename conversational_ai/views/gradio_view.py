import gradio as gr
import asyncio
import logging
import httpx
from typing import List, Dict
from conversational_ai.models.llm_config import LLMProvider

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

async def check_fastapi_health():
    """Check if the FastAPI server is running."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8000/health", timeout=5.0)
                if response.status_code == 200:
                    logger.info("FastAPI server is running and healthy.")
                    return True
        except Exception as e:
            logger.warning(f"FastAPI health check failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            await asyncio.sleep(2)
    logger.error("FastAPI server is not responding after multiple attempts.")
    return False

async def chat_with_ai(message: str, history: List[Dict], provider: str, model: str):
    provider_map = {
        "OpenAI": LLMProvider.OPENAI,
        "Anthropic": LLMProvider.ANTHROPIC,
        "Gemini": LLMProvider.GEMINI
    }
    
    # Create a new list with the existing history and the new user message
    messages = history + [{"role": "user", "content": message}]

    logger.info(f"Sending chat request to FastAPI with provider: {provider}, model: {model}")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://localhost:8000/chat",
                json={
                    "messages": messages,
                    "provider": provider.lower(),  # Ensure provider is lowercase
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
                # Append the assistant response to the history
                updated_history = messages + [{"role": "assistant", "content": content}]
                return updated_history
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                updated_history = messages + [{"role": "assistant", "content": f"Error: {response.status_code}"}]
                return updated_history
        except Exception as e:
            logger.error(f"Error connecting to FastAPI: {str(e)}")
            updated_history = messages + [{"role": "assistant", "content": f"Error connecting to server: {str(e)}"}]
            return updated_history

def create_gradio_interface():
    # Check if FastAPI server is running before launching Gradio
    if not asyncio.run(check_fastapi_health()):
        logger.error("Cannot start Gradio interface: FastAPI server is not responding.")
        return None

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Conversational AI Chat")
        
        with gr.Row():
            provider = gr.Dropdown(
                choices=["OpenAI", "Anthropic", "Gemini"],
                label="LLM Provider",
                value="OpenAI"
            )
            model = gr.Dropdown(
                choices=["gpt-4", "claude-3-7-sonnet-20250219", "gemini-2.0-flash"],
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
            outputs=chatbot,
            queue=False
        )
        
        clear.click(
            fn=lambda: [],
            inputs=None,
            outputs=chatbot,
            queue=False
        )
    
    return demo