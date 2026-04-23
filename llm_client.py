import json
import logging
from typing import Any, Optional

import requests

log = logging.getLogger(__name__)


class ChatCompletionClient:
    """Minimal OpenAI-compatible /v1/chat/completions client.

    Works with any endpoint that speaks the OpenAI chat-completions protocol:
    OpenAI, vLLM, TGI's OpenAI route, HF Router, Anthropic's OpenAI shim,
    LM Studio, llama.cpp server, etc.
    """

    def __init__(self, api_base: str, api_key: str, model: Optional[str] = None):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _discover_model(self) -> str:
        models_url = f"{self.api_base}/models"
        response = requests.get(
            models_url,
            headers=self._headers(),
            timeout=30,
        )
        if not response.ok:
            raise RuntimeError(
                f"Failed to discover model from {models_url} (status {response.status_code}). "
                "Provide llm_model explicitly."
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"Failed to parse {models_url} response as JSON.") from exc

        data = payload.get("data")
        first_model = data[0] if isinstance(data, list) and data else None
        model_id = first_model.get("id") if isinstance(first_model, dict) else None
        if not isinstance(model_id, str) or not model_id:
            raise RuntimeError(
                f"No models found at {models_url}. Check the endpoint URL or provide llm_model explicitly."
            )
        log.info("Discovered LLM model %s from %s", model_id, models_url)
        self.model = model_id
        return model_id

    def _resolve_model(self) -> str:
        if self.model:
            return self.model
        return self._discover_model()

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: Optional[dict[str, Any]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        extra: Optional[dict[str, Any]] = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self._resolve_model(),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if extra:
            payload.update(extra)

        r = requests.post(
            f"{self.api_base}/chat/completions",
            headers=self._headers(),
            data=json.dumps(payload),
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
