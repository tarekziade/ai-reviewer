from typing import Optional

from .reviewer import ReviewRequest


def build_review_request(
    event_name: str,
    payload: dict,
    mention_trigger: str,
) -> Optional[ReviewRequest]:
    """Decide whether an incoming event should trigger a review, and build
    the ReviewRequest if so. Returns None when gating conditions don't match.

    The same logic backs both the Flask webhook (app.py) and the GitHub
    Action entry point (action_runner.py), so behaviour is identical in
    both deployment modes.
    """
    if event_name not in ("issue_comment", "pull_request_review_comment"):
        return None
    if payload.get("action") != "created":
        return None

    comment = payload.get("comment") or {}
    if mention_trigger not in (comment.get("body") or ""):
        return None
    if comment.get("author_association") not in ("MEMBER", "OWNER", "COLLABORATOR"):
        return None

    repo = payload.get("repository") or {}
    full = repo.get("full_name") or ""
    if "/" not in full:
        return None
    owner, name = full.split("/", 1)

    if event_name == "issue_comment":
        issue = payload.get("issue") or {}
        if not issue.get("pull_request"):
            return None
        if issue.get("state") != "open":
            return None
        pr_number = issue.get("number")
    else:
        pr_number = (payload.get("pull_request") or {}).get("number")

    if not isinstance(pr_number, int):
        return None

    return ReviewRequest(
        owner=owner,
        repo=name,
        number=pr_number,
        trigger_comment_id=comment.get("id") or 0,
        trigger_comment_body=comment.get("body") or "",
        commenter=(comment.get("user") or {}).get("login") or "unknown",
    )
