from enum import Enum
import os
from dotenv import load_dotenv

load_dotenv()

class LLMProvider(Enum):
    OPENAI = "openai"
    LLAMA = "llama"
    GEMINI = "gemini"

class LLMConfig:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.llama_api_key = os.getenv("LLAMA_API_KEY")  # Hypothetical LLaMA API key
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # Initialize clients
        import openai
        import google.generativeai as genai
        openai.api_key = self.openai_api_key
        genai.configure(api_key=self.gemini_api_key)
        self.llama_client = None  # Placeholder for LLaMA client