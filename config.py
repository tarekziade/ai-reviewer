import os
from dataclasses import dataclass
from typing import Optional


def _int_env(name: str, default: int) -> int:
    """Like int(os.environ[name]) with a default, but also treats an empty
    string as "use default" so unset GitHub Action secrets (which forward as
    "") don't blow up int parsing."""
    raw = (os.environ.get(name) or "").strip()
    return int(raw) if raw else default


def _load_private_key() -> Optional[str]:
    inline = os.environ.get("GITHUB_PRIVATE_KEY")
    if inline:
        return inline.replace("\\n", "\n")
    path = os.environ.get("GITHUB_PRIVATE_KEY_PATH")
    if not path:
        return None
    with open(path, "r") as f:
        return f.read()


@dataclass
class Config:
    # Only used in webhook mode (GitHub App). In Action mode the runner
    # provides GITHUB_TOKEN directly so these may be absent.
    github_app_id: Optional[str]
    github_private_key: Optional[str]
    github_webhook_secret: Optional[str]

    llm_api_base: str
    llm_api_key: str
    llm_model: Optional[str]
    llm_bill_to: Optional[str]
    llm_max_tokens: int
    llm_stream: bool

    mention_trigger: str
    review_event: str
    max_diff_chars: int
    review_rules_path: str
    default_review_rules: str
    allow_approve: bool
    persona_header: str
    context_script_path: str
    context_script_timeout: int

    @classmethod
    def from_env(cls, *, require_app: bool = True) -> "Config":
        app_id = os.environ.get("GITHUB_APP_ID")
        private_key = _load_private_key()
        webhook_secret = os.environ.get("GITHUB_WEBHOOK_SECRET")

        if require_app:
            missing = [
                name
                for name, val in [
                    ("GITHUB_APP_ID", app_id),
                    ("GITHUB_PRIVATE_KEY / GITHUB_PRIVATE_KEY_PATH", private_key),
                    ("GITHUB_WEBHOOK_SECRET", webhook_secret),
                ]
                if not val
            ]
            if missing:
                raise RuntimeError(
                    "Missing required env vars for webhook mode: " + ", ".join(missing)
                )

        return cls(
            github_app_id=app_id,
            github_private_key=private_key,
            github_webhook_secret=webhook_secret,
            llm_api_base=os.environ.get("LLM_API_BASE", "https://api.openai.com/v1").rstrip("/"),
            llm_api_key=os.environ["LLM_API_KEY"],
            llm_model=os.environ.get("LLM_MODEL") or None,
            llm_bill_to=os.environ.get("LLM_BILL_TO") or None,
            llm_max_tokens=_int_env("LLM_MAX_TOKENS", 4096),
            llm_stream=os.environ.get("LLM_STREAM", "").lower() in ("1", "true", "yes"),
            mention_trigger=os.environ.get("MENTION_TRIGGER", "@serge"),
            review_event=os.environ.get("REVIEW_EVENT", "COMMENT"),
            max_diff_chars=_int_env("MAX_DIFF_CHARS", 200000),
            review_rules_path=os.environ.get("REVIEW_RULES_PATH", ".ai/review-rules.md"),
            default_review_rules=os.environ.get(
                "DEFAULT_REVIEW_RULES",
                "Apply general Python correctness and security standards.",
            ),
            allow_approve=os.environ.get("ALLOW_APPROVE", "").lower() in ("1", "true", "yes"),
            persona_header=os.environ.get("PERSONA_HEADER", "🤗 **Serge** says:"),
            context_script_path=os.environ.get("CONTEXT_SCRIPT_PATH", ".ai/context-script"),
            context_script_timeout=_int_env("CONTEXT_SCRIPT_TIMEOUT", 30),
        )
