from dotenv import load_dotenv
import os
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class LLM:
    def __init__(self, agent):
        self.agent = agent
    def generate_response(self, prompt: str) -> str:
        response = client.chat.completions.create(
            model="gpt-5-nano-2025-08-07", #for faster inference
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


