import os
import json
import requests
from dotenv import load_dotenv
load_dotenv()

class LLM:
    def __init__(self, Agent):
        self.agent = agent
        self.provider = os.getenv("LLM_PROVIDER", "openai").lower()
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")

        # Base URLs
        self.openai_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Auth (OpenAI)
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")

    def generate_response(self, prompt: str) -> str:
        if self.provider == "ollama":
            return self._ollama_chat(prompt)
        return self._openai_chat(prompt)

    def _openai_chat(self, prompt: str) -> str:
        url = f"{self.openai_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for a 2D agent simulation."},
                {"role": "user", "content": prompt},
            ],
        }
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _ollama_chat(self, prompt: str) -> str:

        url = f"{self.ollama_base}/api/chat"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for a 2D agent simulation."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }

        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        return str(data)

llm = LLM()
print(llm.generate_response("How are you doing today?"))