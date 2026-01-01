import os
import uuid
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, Future
from openai import OpenAI
import ollama
from constants import system_prompt

load_dotenv()


# Prompt template for fusing summary with information received from other agents
INTERACTION_PROMPT = """Merge these two information sources into a unified summary. Return the summary only.

[Info 1]:
{summary}

[Info 2]:
{received_info}

"""

# Prompt template for fusing summary with privately discovered information from sites
PRIVATE_INFO_PROMPT = """Merge these two information sources into a unified summary. Return the summary only.

[Info 1]:
{summary}

[Info 2]:
{private_info}

"""


class LLMRequestError(RuntimeError):
    """Raised when the LLM provider returns an unrecoverable response."""
    pass


# Global executor shared across all LLM instances
_LLM_EXECUTOR: ThreadPoolExecutor | None = None


class LLM:
    def __init__(self, agent=None):
        self.agent = agent
        
        # Provider selection: "openai" or "ollama"
        self.provider = os.getenv("LLM_PROVIDER", "openai").lower()
        
        # Default models per provider
        if self.provider == "ollama":
            self.model = os.getenv("LLM_MODEL", "gemma3")
        else:
            self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        
        # OpenAI client (only needed for openai provider)
        if self.provider == "openai":
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY", ""),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            )

        # Initialize global executor if needed
        global _LLM_EXECUTOR
        if _LLM_EXECUTOR is None:
            max_workers = int(os.getenv("LLM_MAX_WORKERS", "200"))
            _LLM_EXECUTOR = ThreadPoolExecutor(
                max_workers=max_workers, 
                thread_name_prefix="llm"
            )

        self._executor = _LLM_EXECUTOR
        self._futures: dict[str, Future] = {}
        
        print(f"🤖 LLM initialized: provider={self.provider}, model={self.model}")

    def chat(self, prompt: str) -> str:
        """Make a chat completion request."""
        if self.provider == "ollama":
            return self._ollama_chat(prompt)
        else:
            return self._openai_chat(prompt)

    def _openai_chat(self, prompt: str) -> str:
        """Make a chat request using OpenAI SDK."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                timeout=200,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise LLMRequestError(f"OpenAI chat request failed: {exc}") from exc

    def _ollama_chat(self, prompt: str) -> str:
        """Make a chat request using native Ollama library."""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            return response["message"]["content"] or ""
        except Exception as exc:
            raise LLMRequestError(f"Ollama chat request failed: {exc}") from exc

    def submit_interaction(self, summary: str, received_info: str) -> str:
        """
        Submit a fusion task for summary + received information from other agents.
        Returns a task_id to poll for results.
        """
        prompt = INTERACTION_PROMPT.format(
            summary=summary or "(no prior knowledge)",
            received_info=received_info or "(no received information)"
        )
        task_id = uuid.uuid4().hex
        future = self._executor.submit(self.chat, prompt)
        self._futures[task_id] = future
        return task_id

    def submit_private_info(self, summary: str, private_info: str) -> str:
        """
        Submit a fusion task for summary + privately discovered information from sites.
        Returns a task_id to poll for results.
        """
        prompt = PRIVATE_INFO_PROMPT.format(
            summary=summary or "(no prior knowledge)",
            private_info=private_info or "(no new information)"
        )
        task_id = uuid.uuid4().hex
        future = self._executor.submit(self.chat, prompt)
        self._futures[task_id] = future
        return task_id

    def poll(self) -> list[tuple[str, str]]:
        """Check for completed tasks and return their results."""
        completed: list[tuple[str, str]] = []
        to_remove: list[str] = []
        fatal_exc: Exception | None = None

        for task_id, fut in list(self._futures.items()):
            if fut.done():
                to_remove.append(task_id)
                try:
                    result = fut.result()
                    completed.append((task_id, result))
                except LLMRequestError as exc:
                    fatal_exc = exc
                except Exception as exc:
                    completed.append((task_id, f"LLM error: {exc}"))

        for task_id in to_remove:
            self._futures.pop(task_id, None)

        if fatal_exc is not None:
            raise fatal_exc

        return completed

    def cancel_all(self) -> None:
        """Cancel all pending tasks."""
        for fut in self._futures.values():
            fut.cancel()
        self._futures.clear()
