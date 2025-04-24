from typing import AsyncGenerator
from fastapi import HTTPException
from conversational_ai.models.llm_config import LLMProvider, LLMConfig
import openai
from google.generativeai import GenerativeModel
import asyncio

class LLMService:
    def __init__(self, config: LLMConfig):
        self.config = config

    async def generate_response(self, messages, provider: LLMProvider, model: str, stream: bool) -> AsyncGenerator[str, None]:
        if provider == LLMProvider.OPENAI:
            return await self._openai_generate(messages, model, stream)
        elif provider == LLMProvider.LLAMA:
            return await self._llama_generate(messages, model, stream)
        elif provider == LLMProvider.GEMINI:
            return await self._gemini_generate(messages, model, stream)
        else:
            raise HTTPException(status_code=400, detail="Invalid LLM provider")

    async def _openai_generate(self, messages, model: str, stream: bool) -> AsyncGenerator[str, None]:
        try:
            if stream:
                async for chunk in await openai.ChatCompletion.acreate(
                    model=model,
                    messages=messages,
                    stream=True
                ):
                    content = chunk.choices[0].delta.get("content", "")
                    if content:
                        yield content
            else:
                response = await openai.ChatCompletion.acreate(
                    model=model,
                    messages=messages
                )
                yield response.choices[0].message.content
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")

    async def _llama_generate(self, messages, model: str, stream: bool) -> AsyncGenerator[str, None]:
        # Placeholder for LLaMA API (replace with actual LLaMA client implementation)
        try:
            if not self.config.llama_client:
                raise HTTPException(status_code=500, detail="LLaMA client not initialized")
            
            if stream:
                # Simulate streaming response
                content = "This is a simulated LLaMA response (streaming not fully implemented yet)."
                for part in content.split():
                    yield part + " "
                    await asyncio.sleep(0.1)  # Simulate streaming delay
            else:
                yield "This is a simulated LLaMA response."
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLaMA error: {str(e)}")

    async def _gemini_generate(self, messages, model: str, stream: bool) -> AsyncGenerator[str, None]:
        try:
            model_instance = GenerativeModel(model)
            chat = model_instance.start_chat(history=messages)
            
            if stream:
                response = chat.send_message(messages[-1]["content"], stream=True)
                for chunk in response:
                    yield chunk.text
            else:
                response = chat.send_message(messages[-1]["content"])
                yield response.text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")