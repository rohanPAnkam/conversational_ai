from enum import Enum
import os
from dotenv import load_dotenv
import anthropic
load_dotenv()

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "antropic"
    GEMINI = "gemini"

class LLMConfig:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # Initialize clients
        import openai
        import google.generativeai as genai
        openai.api_key = self.openai_api_key
        genai.configure(api_key=self.gemini_api_key)
        anthropic.Anthropic(api_key=self.anthropic_api_key)