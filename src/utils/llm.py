"""Centralized LLM utilities for Paper2Code"""

import os
import sys
import asyncio
from pathlib import Path
from openai import OpenAI, AsyncOpenAI

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def call_openai(prompt: str, max_tokens: int = None, temperature: float = 0) -> str:
    """Make OpenAI API call and return response text
    
    Args:
        prompt: The prompt to send to OpenAI
        max_tokens: Optional maximum tokens for response
        temperature: Temperature for response generation (default: 0)
        
    Returns:
        The response text from OpenAI
    """
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    kwargs = {
        "model": config.OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
        
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


async def call_openai_async(prompt: str, max_tokens: int = None, temperature: float = 0) -> str:
    """Make async OpenAI API call and return response text
    
    This enables parallel execution when used in AsyncParallelBatchNode.
    
    Args:
        prompt: The prompt to send to OpenAI
        max_tokens: Optional maximum tokens for response
        temperature: Temperature for response generation (default: 0)
        
    Returns:
        The response text from OpenAI
    """
    client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    kwargs = {
        "model": config.OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
        
    response = await client.chat.completions.create(**kwargs)
    return response.choices[0].message.content