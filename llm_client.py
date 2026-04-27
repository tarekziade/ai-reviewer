import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

log = logging.getLogger(__name__)


<<<<<<< Updated upstream
=======
class AddingSomethingUseless:
    pass


p = AddingSomethingUseless()


@dataclass
class ChatResult:
    content: str
    usage: dict[str, Any] = field(default_factory=dict)
    latency_seconds: float = 0.0

    @property
    def prompt_tokens(self) -> Optional[int]:
        v = self.usage.get("prompt_tokens")
        return v if isinstance(v, int) else None

    @property
    def completion_tokens(self) -> Optional[int]:
        v = self.usage.get("completion_tokens")
        return v if isinstance(v, int) else None


>>>>>>> Stashed changes
class ChatCompletionClient:
    """Minimal OpenAI-compatible /v1/chat/completions client.

    Works with any endpoint that speaks the OpenAI chat-completions protocol:
    OpenAI, vLLM, TGI's OpenAI route, HF Router, Anthropic's OpenAI shim,
    LM Studio, llama.cpp server, etc.
    """

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: Optional[str] = None,
        bill_to: Optional[str] = None,
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.bill_to = bill_to or None

    def _api_base_v1(self) -> str:
        if self.api_base.endswith("/v1"):
            return self.api_base
        return f"{self.api_base}/v1"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.bill_to:
            # HF Router: route inference billing to an org the token has access to.
            headers["X-HF-Bill-To"] = self.bill_to
        return headers

    def _discover_model(self) -> str:
        models_url = f"{self._api_base_v1()}/models"
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
    ) -> ChatResult:
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

        url = f"{self._api_base_v1()}/chat/completions"
        body = json.dumps(payload)
        attempts = 3
        started = time.monotonic()
        for attempt in range(1, attempts + 1):
            try:
                r = requests.post(url, headers=self._headers(), data=body, timeout=300)
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt == attempts:
                    raise
                log.warning("LLM call attempt %d/%d failed: %s; retrying", attempt, attempts, exc)
                time.sleep(2**attempt)
                continue
            if r.status_code >= 500 and attempt < attempts:
                log.warning(
                    "LLM call attempt %d/%d returned %d; retrying",
                    attempt, attempts, r.status_code,
                )
                time.sleep(2**attempt)
                continue
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage") or {}
            latency = time.monotonic() - started
            log.info(
                "LLM call ok in %.1fs (prompt=%s, completion=%s)",
                latency,
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
            )
            return ChatResult(content=content, usage=usage, latency_seconds=latency)
        raise RuntimeError("unreachable")  # loop always returns or raises
