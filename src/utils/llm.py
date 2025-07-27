"""Centralized LLM utilities for Paper2Code"""

import os
import sys
from pathlib import Path
from openai import OpenAI, AsyncOpenAI

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def _build_kwargs(prompt: str, max_tokens: int = None, temperature: float = 0) -> dict:
    """Build kwargs dict for OpenAI API calls."""
    kwargs = {
        "model": config.OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return kwargs


def call_openai(prompt: str, max_tokens: int = None, temperature: float = 0) -> str:
    """Make OpenAI API call and return response text."""
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = client.chat.completions.create(**_build_kwargs(prompt, max_tokens, temperature))
    return response.choices[0].message.content


async def call_openai_async(prompt: str, max_tokens: int = None, temperature: float = 0) -> str:
    """Make async OpenAI API call for parallel execution in AsyncParallelBatchNode."""
    client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = await client.chat.completions.create(**_build_kwargs(prompt, max_tokens, temperature))
    return response.choices[0].message.content