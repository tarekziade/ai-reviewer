import os
from dataclasses import dataclass
from typing import Optional


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

    mention_trigger: str
    review_event: str
    max_diff_chars: int
    review_rules_path: str
    default_review_rules: str

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
            mention_trigger=os.environ.get("MENTION_TRIGGER", "@serge"),
            review_event=os.environ.get("REVIEW_EVENT", "COMMENT"),
            max_diff_chars=int(os.environ.get("MAX_DIFF_CHARS", "200000")),
            review_rules_path=os.environ.get("REVIEW_RULES_PATH", ".ai/review-rules.md"),
            default_review_rules=os.environ.get(
                "DEFAULT_REVIEW_RULES",
                "Apply general Python correctness and security standards.",
            ),
        )
