"""Centralized LLM utilities for Paper2Code"""

import os
from abc import ABC, abstractmethod
from openai import OpenAI, AsyncOpenAI
from anthropic import Anthropic, AsyncAnthropic


# Default model - can be changed to switch providers
DEFAULT_MODEL = "gpt-4.1-mini"  # or "claude-3-5-sonnet"


class LLMBackend(ABC):
    """Base class for LLM providers"""
    
    @abstractmethod
    def call(self, prompt: str, max_tokens: int, temperature: float) -> str:
        pass
    
    @abstractmethod
    async def call_async(self, prompt: str, max_tokens: int, temperature: float) -> str:
        pass


class OpenAIBackend(LLMBackend):
    """OpenAI API backend"""
    
    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.async_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def call(self, prompt: str, max_tokens: int, temperature: float) -> str:
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    
    async def call_async(self, prompt: str, max_tokens: int, temperature: float) -> str:
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        response = await self.async_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


class AnthropicBackend(LLMBackend):
    """Anthropic Claude API backend"""
    
    def __init__(self, model: str):
        self.model = model
        self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.async_client = AsyncAnthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    
    def call(self, prompt: str, max_tokens: int, temperature: float) -> str:
        response = self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens or 4096
        )
        return response.content[0].text
    
    async def call_async(self, prompt: str, max_tokens: int, temperature: float) -> str:
        response = await self.async_client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens or 4096
        )
        return response.content[0].text


def _get_backend(model: str) -> LLMBackend:
    """Get appropriate backend based on model name"""
    if model.startswith(("gpt", "o1")):
        return OpenAIBackend(model)
    elif model.startswith("claude"):
        return AnthropicBackend(model)
    else:
        raise ValueError(f"Unknown model: {model}")


def call_llm(prompt: str, model: str = None, max_tokens: int = None, temperature: float = 0) -> str:
    """Make LLM API call and return response text."""
    model = model or DEFAULT_MODEL
    backend = _get_backend(model)
    return backend.call(prompt, max_tokens, temperature)


async def call_llm_async(prompt: str, model: str = None, max_tokens: int = None, temperature: float = 0) -> str:
    """Make async LLM API call for parallel execution in AsyncParallelBatchNode."""
    model = model or DEFAULT_MODEL
    backend = _get_backend(model)
    return await backend.call_async(prompt, max_tokens, temperature)


# Legacy function names for compatibility
call_openai = call_llm
call_openai_async = call_llm_async