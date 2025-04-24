import logging
import os
from typing import List, Dict
from dotenv import load_dotenv
import httpx
import json
from conversational_ai.models.llm_config import LLMConfig, LLMProvider

# Load environment variables from .env file
load_dotenv()

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

class LLMService:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            logger.error("GEMINI_API_KEY not found in .env file")
    
    async def generate_response(self, messages: List[Dict], provider: LLMProvider, model: str, stream: bool):
        logger.info(f"Generating response with provider: {provider}, model: {model}, stream: {stream}")
        if provider == LLMProvider.OPENAI:
            async for chunk in self._openai_generate(messages, model, stream):
                logger.info(f"OpenAI chunk: {chunk}")
                yield chunk
        elif provider == LLMProvider.LLAMA:
            async for chunk in self._llama_generate(messages, model, stream):
                logger.info(f"LLaMA chunk: {chunk}")
                yield chunk
        elif provider == LLMProvider.GEMINI:
            async for chunk in self._gemini_generate(messages, model, stream):
                logger.info(f"Gemini chunk: {chunk}")
                yield chunk
    
    async def _openai_generate(self, messages: List[Dict], model: str, stream: bool):
        # Placeholder for OpenAI API call
        if stream:
            for i in range(3):
                yield f"OpenAI response chunk {i}\n"
        else:
            yield "OpenAI full response\n"
    
    async def _llama_generate(self, messages: List[Dict], model: str, stream: bool):
        # Placeholder for LLaMA API call
        full_response = "LLaMA full response: FastAPI is a high-performance Python framework for building APIs, leveraging Python 3.6+ type hints for automatic validation and documentation."
        if stream:
            yield full_response + "\n"
        else:
            yield full_response + "\n"
    
    async def _gemini_generate(self, messages: List[Dict], model: str, stream: bool):
        # Correct Gemini API endpoint
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.gemini_api_key}"
        headers = {
            "Content-Type": "application/json"
        }
        # Convert messages to a format suitable for the Gemini API
        payload = {
            "contents": [{"parts": [{"text": message["content"]} for message in messages]}]
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:  # Increased timeout to 10 seconds
            try:
                logger.info(f"Sending request to {api_url} with payload: {json.dumps(payload)}")
                response = await client.post(api_url, json=payload, headers=headers)
                response.raise_for_status()
                logger.info(f"Received response: {response.text}")
                if stream:
                    # Simulate streaming by splitting the response into chunks
                    data = response.json()
                    full_response = data["candidates"][0]["content"]["parts"][0]["text"]
                    chunks = full_response.split("\n")
                    for chunk in chunks:
                        if chunk.strip():
                            yield chunk + "\n"
                else:
                    data = response.json()
                    full_response = data["candidates"][0]["content"]["parts"][0]["text"]
                    if full_response.strip():
                        yield full_response + "\n"
            except httpx.HTTPStatusError as e:
                logger.error(f"Gemini API error: {e.response.status_code} - {e.response.text}")
                yield f"Error: Gemini API request failed with status {e.response.status_code} - {e.response.text}"
            except httpx.RequestError as e:
                logger.error(f"Network error calling Gemini API: {str(e)}")
                yield f"Error: Network issue - {str(e)}"
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Gemini API response: {str(e)}")
                yield f"Error: Invalid JSON response - {str(e)}"
            except Exception as e:
                logger.error(f"Unexpected error calling Gemini API: {str(e)}", exc_info=True)
                yield f"Error: Unexpected error - {str(e)}"