# Conversational AI Chat

Conversational AI platform built with FastAPI and Gradio, supporting OpenAI, Anthropic, and Gemini LLMs.

## Setup

### Prerequisites
- Python 3.7+
- FastAPI
- openai
- anthropic
- asyncio

### LLM Switching Guide
- Supported Providers: OpenAI, Anthropic, Gemini.
- Models: gpt-4, claude-3-7-sonnet-20250219, gemini-pro.
- Switching:
  1. Launch the interface at http://localhost:7860.
  2. Use the "LLM Provider" and "Model" dropdowns to select (e.g., Gemini with gemini-pro).
  3. The backend handles switching via LLMProvider enums in llm_config.py and provider-specific logic in llm_service.py.
- Customization:
  1. Add new providers by updating LLMProvider and implementing _provider_generate methods in llm_service.py.
  2. Configure API keys or credentials in llm_config.py if required.
  
