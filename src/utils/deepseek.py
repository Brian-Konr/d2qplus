import asyncio
from typing import List
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI

load_dotenv("/home/guest/r12922050/GitHub/d2qplus/.env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

async def get_deepseek_response(messages: List, model: str = "deepseek-chat"):
    """
    Asynchronously sends a prompt to the DeepSeek API and returns the response content.
    """
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred while processing prompt: {e}")
        return "" # Or handle error as appropriate

async def get_deepseek_response_batch(messages: List[List], model: str = "deepseek-chat"):
    """
    Asynchronously sends a batch of prompts to the DeepSeek API and returns the responses.
    """
    tasks = [get_deepseek_response(msg, model) for msg in messages]
    print("Sending batch requests to DeepSeek API...")
    responses = await asyncio.gather(*tasks)