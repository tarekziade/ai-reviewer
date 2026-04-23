import json
from typing import Any, Optional

import requests


class ChatCompletionClient:
    """Minimal OpenAI-compatible /v1/chat/completions client.

    Works with any endpoint that speaks the OpenAI chat-completions protocol:
    OpenAI, vLLM, TGI's OpenAI route, HF Router, Anthropic's OpenAI shim,
    LM Studio, llama.cpp server, etc.
    """

    def __init__(self, api_base: str, api_key: str, model: str):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model

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
            "model": self.model,
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
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
