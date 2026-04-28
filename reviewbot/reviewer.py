import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from .config import Config
from .context_script import run_context_script
from .github_client import GitHubClient
from .llm_client import ChatCompletionClient, ChatResult, ToolCall
from .patch import ParsedFile, parse_patch
from .prompts import build_system_prompt, build_user_prompt
from .tools import TOOL_SPECS, ToolEnv, run_tool

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


def _build_annotated_diff(
    files: list[dict],
    max_chars: int,
    skip_paths: set[str],
) -> tuple[str, dict[str, ParsedFile], list[str]]:
    """Annotate each file's patch and collect addressable line positions.

    `skip_paths` is honored as an exclusion list — entries are reported
    back so the caller can mention them to the LLM, but their patch
    contents are never sent. Repos drive this list through their
    `.ai/context-script` (e.g. to exclude regenerated files).
    """
    parsed_by_path: dict[str, ParsedFile] = {}
    blocks: list[str] = []
    skipped: list[str] = []
    total = 0
    for f in files:
        path = f.get("filename")
        patch = f.get("patch")
        if not path or not patch:
            continue
        if path in skip_paths:
            skipped.append(path)
            continue
        parsed = parse_patch(path, patch)
        parsed_by_path[path] = parsed
        block = f"--- a/{path}\n+++ b/{path}\n{parsed.annotated}\n"
        if total + len(block) > max_chars:
            blocks.append(f"\n[diff truncated after {total} chars]\n")
            break
        blocks.append(block)
        total += len(block)
    return "\n".join(blocks), parsed_by_path, skipped


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


@dataclass
class _AggregateMetrics:
    """Accumulated stats across all LLM turns in one agentic loop."""
    turns: int = 0
    tool_calls: int = 0
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _format_aggregated_metrics(m: "_AggregateMetrics") -> str:
    parts = [
        f"{m.turns} LLM turn{'s' if m.turns != 1 else ''}",
        f"{m.tool_calls} tool call{'s' if m.tool_calls != 1 else ''}",
        f"{m.latency_seconds:.1f}s",
    ]
    if m.prompt_tokens or m.completion_tokens:
        parts.append(f"{m.prompt_tokens or '?'} in / {m.completion_tokens or '?'} out tokens")
    return " · ".join(parts)


def _make_tool_env(cfg: Config) -> Optional[ToolEnv]:
    if not cfg.repo_checkout_path:
        return None
    try:
        env = ToolEnv(repo_root=cfg.repo_checkout_path)
    except Exception:
        log.exception("repo checkout path invalid; running without browse tools")
        return None
    log.info("Browse tools enabled, rooted at %s", env.repo_root)
    return env


def _run_agentic_loop(
    llm: ChatCompletionClient,
    initial_messages: list[dict[str, Any]],
    *,
    cfg: Config,
    tool_env: Optional[ToolEnv],
) -> tuple[ChatResult, _AggregateMetrics]:
    """Run a tool-augmented chat loop until the model emits a final
    (non-tool) response, falling back to a final non-tool turn if the
    iteration budget is exhausted.

    Returns the *last* ChatResult (whose ``content`` carries the JSON
    review) and an aggregate-metrics struct."""
    messages = list(initial_messages)
    metrics = _AggregateMetrics()
    tools_arg = TOOL_SPECS if tool_env is not None else None

    for iteration in range(1, cfg.tool_max_iterations + 1):
        log.info("Agent loop iteration %d/%d", iteration, cfg.tool_max_iterations)
        chat = llm.complete(
            messages,
            response_format={"type": "json_object"},
            max_tokens=cfg.llm_max_tokens,
            tools=tools_arg,
            tool_choice="auto" if tools_arg else None,
        )
        metrics.turns += 1
        metrics.latency_seconds += chat.latency_seconds
        if chat.prompt_tokens is not None:
            metrics.prompt_tokens += chat.prompt_tokens
        if chat.completion_tokens is not None:
            metrics.completion_tokens += chat.completion_tokens

        if not chat.tool_calls:
            return chat, metrics

        if tool_env is None:
            # Model emitted tool calls even though we didn't pass tools —
            # ignore them and treat the textual content as the answer.
            log.warning(
                "Model emitted %d tool_call(s) but tools are disabled; using "
                "content as final answer",
                len(chat.tool_calls),
            )
            return chat, metrics

        # Append the assistant's tool_calls turn so the next request has
        # the full conversation, then execute each call and append the
        # results as `tool` messages.
        messages.append({
            "role": "assistant",
            "content": chat.content or None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in chat.tool_calls
            ],
        })
        for tc in chat.tool_calls:
            metrics.tool_calls += 1
            result = _execute_tool_call(tool_env, tc)
            # Kimi-K2 (and some other engines) require ``name`` on tool
            # replies; OpenAI's spec ignores it. Always sending it is
            # the safer cross-provider choice.
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": result,
            })

    # Iteration budget hit — force a final answer with tools disabled.
    log.warning(
        "Tool budget (%d) exhausted; asking model for a final review without tools",
        cfg.tool_max_iterations,
    )
    messages.append({
        "role": "user",
        "content": (
            "You have used all available tool calls. Based on what you have "
            "already gathered, produce the final JSON review now. Do not "
            "request any more tools."
        ),
    })
    chat = llm.complete(
        messages,
        response_format={"type": "json_object"},
        max_tokens=cfg.llm_max_tokens,
    )
    metrics.turns += 1
    metrics.latency_seconds += chat.latency_seconds
    if chat.prompt_tokens is not None:
        metrics.prompt_tokens += chat.prompt_tokens
    if chat.completion_tokens is not None:
        metrics.completion_tokens += chat.completion_tokens
    return chat, metrics


def _execute_tool_call(env: ToolEnv, tc: ToolCall) -> str:
    """Parse the model's tool arguments and dispatch. Always returns a
    string — errors are surfaced to the model rather than raised."""
    try:
        args = json.loads(tc.arguments) if tc.arguments else {}
    except json.JSONDecodeError as exc:
        log.warning("tool %s emitted unparseable arguments: %s", tc.name, exc)
        return f"error: arguments were not valid JSON: {exc}"
    if not isinstance(args, dict):
        return f"error: arguments must be a JSON object, got {type(args).__name__}"
    log.info("tool call: %s(%s)", tc.name, _summarize_args(args))
    output = run_tool(env, tc.name, args)
    log.info("tool call %s returned %d chars", tc.name, len(output))
    return output


def _summarize_args(args: dict[str, Any], limit: int = 200) -> str:
    s = json.dumps(args, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + "..."


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

    ctx_result = run_context_script(
        cfg.context_script_path,
        title=pr.get("title") or "",
        body=pr.get("body") or "",
        files=files,
        timeout_seconds=cfg.context_script_timeout,
    )
    skip_paths: set[str] = set(ctx_result.skip_files) if ctx_result else set()
    extra_context = ctx_result.context if ctx_result else None
    if ctx_result:
        log.info(
            "context script: context=%d chars, skip_files=%d",
            len(extra_context or ""),
            len(skip_paths),
        )

    diff_text, parsed_by_path, skipped = _build_annotated_diff(
        files, cfg.max_diff_chars, skip_paths
    )
    if skipped:
        log.info("Excluded %d file(s) per .ai/context-script: %s", len(skipped), skipped)
        note = (
            "The following files were excluded from this review by the target "
            "repo's `.ai/context-script`. Do NOT review their contents and do "
            "NOT place inline comments on them. Refer to REPO-PROVIDED "
            "CONTEXT below for the reason and any related guidance:\n  - "
            + "\n  - ".join(skipped)
            + "\n\n"
        )
        diff_text = note + diff_text
    if not parsed_by_path:
        gh.post_issue_comment(
            req.owner,
            req.repo,
            req.number,
            "No reviewable diff hunks were found (binary files, empty patches, or all files excluded by .ai/context-script).",
        )
        return

    review_rules = _load_review_rules(gh, req.owner, req.repo, pr, cfg)

    llm = ChatCompletionClient(
        cfg.llm_api_base,
        cfg.llm_api_key,
        cfg.llm_model,
        bill_to=cfg.llm_bill_to,
        stream=cfg.llm_stream,
    )
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
        extra_context=extra_context,
    )

    tool_env = _make_tool_env(cfg)
    chat, total_metrics = _run_agentic_loop(
        llm,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        cfg=cfg,
        tool_env=tool_env,
    )
    metrics_line = _format_aggregated_metrics(total_metrics)

    try:
        result = _extract_json(chat.content)
    except Exception:
        log.exception("could not parse LLM output as JSON")
        gh.post_issue_comment(
            req.owner,
            req.repo,
            req.number,
            f"Reviewer LLM returned unparseable output ({metrics_line}):\n\n```\n{chat.content[:3000]}\n```",
        )
        return

    summary = (result.get("summary") or "").strip()
    event = result.get("event") or cfg.review_event
    if event not in ("COMMENT", "REQUEST_CHANGES", "APPROVE"):
        event = cfg.review_event

    if event == "APPROVE" and not cfg.allow_approve:
        log.info("Downgrading APPROVE to COMMENT (Actions tokens cannot approve; set ALLOW_APPROVE=1 in App mode to permit)")
        event = "COMMENT"

    valid, rejected = _validate_comments(result.get("comments") or [], parsed_by_path)
    if rejected:
        log.warning("Dropped %d invalid comment(s): %s", len(rejected), rejected)

    # GitHub requires a body when there are no inline comments and the event is
    # not APPROVE; also REQUEST_CHANGES requires at least one comment or a body.
    body = summary or "(no overall summary provided)"
    if cfg.persona_header:
        body = f"{cfg.persona_header}\n\n{body}"
    if rejected:
        body += f"\n\n_Note: {len(rejected)} suggested inline comment(s) were dropped because they referenced lines not present in the diff._"
    body += f"\n\n_{metrics_line}_"

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
