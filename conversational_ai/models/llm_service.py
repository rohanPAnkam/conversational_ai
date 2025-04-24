import logging
import os
from typing import List, Dict
from dotenv import load_dotenv
import httpx
import json
from conversational_ai.models.llm_config import LLMConfig, LLMProvider
import anthropic

load_dotenv()


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
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.gemini_api_key:
            logger.error("GEMINI_API_KEY not found in .env file")
        if not self.openai_api_key:
            logger.error("OPENAI_API_KEY not found in .env file")
        if not self.anthropic_api_key:
            logger.error("ANTHROPIC_API_KEY not found in .env file")
        self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
    
    async def generate_response(self, messages: List[Dict], provider: LLMProvider, model: str, stream: bool):
        logger.info(f"Generating response with provider: {provider}, model: {model}, stream: {stream}")
        if provider == LLMProvider.OPENAI:
            async for chunk in self._openai_generate(messages, model, stream):
                logger.info(f"OpenAI chunk: {chunk}")
                yield chunk
        elif provider == LLMProvider.ANTHROPIC:
            async for chunk in self._anthropic_generate(messages, model, stream):
                logger.info(f"Anthropic chunk: {chunk}")
                yield chunk
        elif provider == LLMProvider.GEMINI:
            async for chunk in self._gemini_generate(messages, model, stream):
                logger.info(f"Gemini chunk: {chunk}")
                yield chunk
    
    async def _openai_generate(self, messages: List[Dict], model: str, stream: bool):
        api_url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                logger.info(f"Sending request to {api_url} with payload: {json.dumps(payload)}")
                response = await client.post(api_url, json=payload, headers=headers)
                response.raise_for_status()
                logger.info(f"Received response: {response.text}")
                if stream:
                    async for chunk in response.aiter_text():
                        if chunk.strip():
                            data = json.loads(chunk)
                            if "choices" in data and data["choices"]:
                                content = data["choices"][0].get("delta", {}).get("content", "")
                                if content:
                                    yield content
                else:
                    data = response.json()
                    full_response = data["choices"][0]["message"]["content"]
                    if full_response.strip():
                        yield full_response + "\n"
            except httpx.HTTPStatusError as e:
                logger.error(f"OpenAI API error: {e.response.status_code} - {e.response.text}")
                yield f"Error: OpenAI API request failed with status {e.response.status_code} - {e.response.text}"
            except httpx.RequestError as e:
                logger.error(f"Network error calling OpenAI API: {str(e)}")
                yield f"Error: Network issue - {str(e)}"
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse OpenAI API response: {str(e)}")
                yield f"Error: Invalid JSON response - {str(e)}"
            except Exception as e:
                logger.error(f"Unexpected error calling OpenAI API: {str(e)}", exc_info=True)
                yield f"Error: Unexpected error - {str(e)}"
    
    async def _anthropic_generate(self, messages: List[Dict], model: str, stream: bool):
        try:
            logger.info(f"Generating response with Anthropic model: {model}")
            # Extract the last user message as the prompt
            user_message = messages[-1]["content"] if messages else ""
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=1000,
                temperature=1,
                system="You are a helpful assistant.",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_message
                            }
                        ]
                    }
                ]
            )
            logger.info(f"Received response from Anthropic: {response.content}")
            full_response = response.content[0].text if response.content and isinstance(response.content, list) else ""
            if stream:
                chunks = full_response.split("\n")
                for chunk in chunks:
                    if chunk.strip():
                        yield chunk + "\n"
            else:
                if full_response.strip():
                    yield full_response + "\n"
        except Exception as e:
            logger.error(f"Unexpected error calling Anthropic API: {str(e)}", exc_info=True)
            yield f"Error: Unexpected error - {str(e)}"
    
    async def _gemini_generate(self, messages: List[Dict], model: str, stream: bool):
        # Gemini API endpoint
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.gemini_api_key}"
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "contents": [{"parts": [{"text": message["content"]} for message in messages]}]
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                logger.info(f"Sending request to {api_url} with payload: {json.dumps(payload)}")
                response = await client.post(api_url, json=payload, headers=headers)
                response.raise_for_status()
                logger.info(f"Received response: {response.text}")
                if stream:
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