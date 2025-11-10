import os
import json
import requests
from dotenv import load_dotenv
load_dotenv()
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import uuid
from constants import system_prompt
class LLM:
    def __init__(self, agent=None):
        self.agent = agent
        self.provider = os.getenv("LLM_PROVIDER", "openai").lower()
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")

        # Base URLs
        self.openai_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Auth (OpenAI)
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")

        global _LLM_EXECUTOR
        if "_LLM_EXECUTOR" not in globals() or _LLM_EXECUTOR is None:
            max_workers = int(os.getenv("LLM_MAX_WORKERS", "10"))
            _LLM_EXECUTOR = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="llm")

        self._executor = _LLM_EXECUTOR
        self._futures: dict[str, Future] = {}

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
                {"role": "system", "content": system_prompt},
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
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "max_tokens": 150,
        }

        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        return str(data)

    def submit(self, prompt: str) -> str:
        task_id = uuid.uuid4().hex
        if self.provider == "ollama":
            future = self._executor.submit(self._ollama_chat, prompt)
        else:
            future = self._executor.submit(self._openai_chat, prompt)
        self._futures[task_id] = future
        return task_id

    def poll(self) -> list[tuple[str, str]]:
        completed: list[tuple[str, str]] = []
        to_remove: list[str] = []
        for task_id, fut in list(self._futures.items()):
            if fut.done():
                try:
                    result = fut.result()
                except Exception as exc: 
                    result = f"LLM error: {exc}"
                completed.append((task_id, result))
                to_remove.append(task_id)
        for task_id in to_remove:
            self._futures.pop(task_id, None)
        return completed

    def cancel_all(self) -> None:
        for fut in self._futures.values():
            fut.cancel()
        self._futures.clear()
