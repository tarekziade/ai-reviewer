import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from config import Config
from github_client import GitHubClient
from llm_client import ChatCompletionClient
from patch import ParsedFile, parse_patch
from prompts import build_system_prompt, build_user_prompt

log = logging.getLogger(__name__)


@dataclass
class ReviewRequest:
    owner: str
    repo: str
    number: int
    trigger_comment_id: int
    trigger_comment_body: str
    commenter: str


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(content: str) -> dict[str, Any]:
    """Be forgiving: accept raw JSON, fenced JSON, or the first {...} block."""
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    m = _JSON_FENCE_RE.search(content)
    if m:
        return json.loads(m.group(1))
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(content[start : end + 1])
    raise ValueError("LLM response did not contain a JSON object")


def _build_annotated_diff(files: list[dict], max_chars: int) -> tuple[str, dict[str, ParsedFile]]:
    parsed_by_path: dict[str, ParsedFile] = {}
    blocks: list[str] = []
    total = 0
    for f in files:
        path = f.get("filename")
        patch = f.get("patch")
        if not path or not patch:
            continue
        parsed = parse_patch(path, patch)
        parsed_by_path[path] = parsed
        block = f"--- a/{path}\n+++ b/{path}\n{parsed.annotated}\n"
        if total + len(block) > max_chars:
            blocks.append(f"\n[diff truncated after {total} chars]\n")
            break
        blocks.append(block)
        total += len(block)
    return "\n".join(blocks), parsed_by_path


def _validate_comments(
    raw_comments: list[dict[str, Any]], parsed_by_path: dict[str, ParsedFile]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for c in raw_comments:
        path = c.get("path")
        side = c.get("side", "RIGHT")
        line = c.get("line")
        body = c.get("body")
        if not (isinstance(path, str) and isinstance(line, int) and isinstance(body, str)):
            rejected.append(c)
            continue
        side = side if side in ("RIGHT", "LEFT") else "RIGHT"
        parsed = parsed_by_path.get(path)
        if not parsed or (side, line) not in parsed.valid_positions:
            rejected.append(c)
            continue
        valid.append({"path": path, "side": side, "line": line, "body": body})
    return valid, rejected


def _load_review_rules(gh: GitHubClient, owner: str, repo: str, pr: dict, cfg: Config) -> str:
    default_branch = pr.get("base", {}).get("repo", {}).get("default_branch") or "main"
    try:
        content = gh.get_file_contents(owner, repo, cfg.review_rules_path, ref=default_branch)
    except Exception:
        log.exception("failed to fetch review rules")
        content = None
    return content or cfg.default_review_rules


def run_review(cfg: Config, gh: GitHubClient, req: ReviewRequest) -> None:
    log.info("Starting review of %s/%s#%d", req.owner, req.repo, req.number)

    try:
        gh.add_reaction_to_issue_comment(req.owner, req.repo, req.trigger_comment_id, "eyes")
    except Exception:
        log.debug("reaction failed (non-fatal)", exc_info=True)

    pr = gh.get_pr(req.owner, req.repo, req.number)
    files = gh.get_pr_files(req.owner, req.repo, req.number)

    diff_text, parsed_by_path = _build_annotated_diff(files, cfg.max_diff_chars)
    if not parsed_by_path:
        gh.post_issue_comment(
            req.owner,
            req.repo,
            req.number,
            "No reviewable diff hunks were found (binary files or empty patches).",
        )
        return

    review_rules = _load_review_rules(gh, req.owner, req.repo, pr, cfg)

    llm = ChatCompletionClient(cfg.llm_api_base, cfg.llm_api_key, cfg.llm_model)
    system_prompt = build_system_prompt(review_rules)
    user_prompt = build_user_prompt(
        repo_full_name=f"{req.owner}/{req.repo}",
        number=req.number,
        title=pr.get("title") or "",
        body=pr.get("body") or "",
        author=(pr.get("user") or {}).get("login") or "unknown",
        commenter=req.commenter,
        trigger_comment=req.trigger_comment_body,
        diff=diff_text,
    )

    raw = llm.complete(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    try:
        result = _extract_json(raw)
    except Exception:
        log.exception("could not parse LLM output as JSON")
        gh.post_issue_comment(
            req.owner,
            req.repo,
            req.number,
            f"Reviewer LLM returned unparseable output:\n\n```\n{raw[:3000]}\n```",
        )
        return

    summary = (result.get("summary") or "").strip()
    event = result.get("event") or cfg.review_event
    if event not in ("COMMENT", "REQUEST_CHANGES", "APPROVE"):
        event = cfg.review_event

    valid, rejected = _validate_comments(result.get("comments") or [], parsed_by_path)
    if rejected:
        log.warning("Dropped %d invalid comment(s): %s", len(rejected), rejected)

    # GitHub requires a body when there are no inline comments and the event is
    # not APPROVE; also REQUEST_CHANGES requires at least one comment or a body.
    body = summary or "(no overall summary provided)"
    if rejected:
        body += f"\n\n_Note: {len(rejected)} suggested inline comment(s) were dropped because they referenced lines not present in the diff._"

    head_sha = (pr.get("head") or {}).get("sha")
    if not head_sha:
        raise RuntimeError("PR payload missing head.sha")

    gh.create_review(
        req.owner,
        req.repo,
        req.number,
        commit_id=head_sha,
        body=body,
        comments=valid,
        event=event,
    )
    log.info(
        "Posted review on %s/%s#%d (%d inline, event=%s)",
        req.owner,
        req.repo,
        req.number,
        len(valid),
        event,
    )
