import os
import json
import requests
from dotenv import load_dotenv
load_dotenv()
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import uuid
from constants import system_prompt


class LLMRequestError(RuntimeError):
    """Raised when the LLM provider returns an unrecoverable response."""
    pass

try:
    from google import genai
except ImportError:
    genai = None


class LLM:
    def __init__(self, agent=None):
        self.agent = agent
        self.provider = os.getenv("LLM_PROVIDER", "openai").lower()
        self.model = os.getenv("LLM_MODEL")
        if not self.model:
            if self.provider == "gemini":
                self.model = "gemini-2.5-flash"
            else:
                self.model = "gpt-4o-mini"

        # Base URLs
        self.openai_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Auth (OpenAI)
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.gemini_api_key = (
            os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
        )

        self._gemini_client = None

        global _LLM_EXECUTOR
        if "_LLM_EXECUTOR" not in globals() or _LLM_EXECUTOR is None:
            max_workers = int(os.getenv("LLM_MAX_WORKERS", "200"))
            _LLM_EXECUTOR = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="llm")

        self._executor = _LLM_EXECUTOR
        self._futures: dict[str, Future] = {}

    def generate_response(self, prompt: str) -> str:
        if self.provider == "gemini":
            return self._gemini_chat(prompt)
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
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=200)
        except requests.exceptions.Timeout as exc:
            raise LLMRequestError("OpenAI chat request timed out after 200 seconds") from exc
        except requests.exceptions.RequestException as exc:
            raise LLMRequestError(f"OpenAI chat request failed: {exc}") from exc

        if resp.status_code >= 400:
            detail = resp.text.strip()
            if len(detail) > 500:
                detail = f"{detail[:500]}…"
            raise LLMRequestError(
                f"OpenAI chat request returned HTTP {resp.status_code}: {detail or 'no response body'}"
            )

        try:
            data = resp.json()
        except ValueError as exc:
            truncated = resp.text[:500] + ("…" if len(resp.text) > 500 else "")
            raise LLMRequestError(
                f"OpenAI chat response is not valid JSON (status {resp.status_code}): {truncated}"
            ) from exc

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMRequestError(
                f"OpenAI chat response missing expected payload: {json.dumps(data)[:500]}"
            ) from exc

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

    def _gemini_chat(self, prompt: str) -> str:
        if genai is None:
            raise RuntimeError("google-genai package is not installed")
        if not self.gemini_api_key:
            raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY environment variable")

        if self._gemini_client is None:
            self._gemini_client = genai.Client(api_key=self.gemini_api_key)

        contents = prompt
        if system_prompt:
            contents = f"{system_prompt}\n\n{prompt}"

        response = self._gemini_client.models.generate_content(
            model=self.model,
            contents=contents,
        )
        if hasattr(response, "text") and response.text:
            return response.text
        return str(response)

    def submit(self, prompt: str) -> str:
        task_id = uuid.uuid4().hex
        if self.provider == "gemini":
            future = self._executor.submit(self._gemini_chat, prompt)
        elif self.provider == "ollama":
            future = self._executor.submit(self._ollama_chat, prompt)
        else:
            future = self._executor.submit(self._openai_chat, prompt)
        self._futures[task_id] = future
        return task_id

    def poll(self) -> list[tuple[str, str]]:
        completed: list[tuple[str, str]] = []
        to_remove: list[str] = []
        fatal_exc: Exception | None = None
        for task_id, fut in list(self._futures.items()):
            if fut.done():
                try:
                    result = fut.result()
                except LLMRequestError as exc:
                    fatal_exc = exc
                    to_remove.append(task_id)
                except Exception as exc: 
                    result = f"LLM error: {exc}"
                    completed.append((task_id, result))
                    to_remove.append(task_id)
                else:
                    completed.append((task_id, result))
                    to_remove.append(task_id)
        for task_id in to_remove:
            self._futures.pop(task_id, None)
        if fatal_exc is not None:
            raise fatal_exc
        return completed

    def cancel_all(self) -> None:
        for fut in self._futures.values():
            fut.cancel()
        self._futures.clear()
