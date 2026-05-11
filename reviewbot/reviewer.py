import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .config import Config
from .context_script import run_context_script
from .github_client import GitHubClient
from .llm_client import ChatCompletionClient, ChatResult, ToolCall
from .patch import DiffSnippetLine, ParsedFile, extract_hunk_snippet, parse_patch
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


@dataclass
class DraftComment:
    """One validated inline comment, with a stable id the web UI can
    address when applying edits (override body, discard). ``diff_hunk``
    is a GitHub-style snippet around the commented line (empty if the
    patch wasn't available)."""
    id: str
    path: str
    side: str
    line: int
    body: str
    diff_hunk: list[DiffSnippetLine] = field(default_factory=list)


@dataclass
class ReviewDraft:
    """The output of prepare_review: everything needed to publish a
    review, but not yet posted. Tweakable from the web UI via edits."""
    owner: str
    repo: str
    number: int
    head_sha: str
    summary: str
    event: str
    comments: list[DraftComment] = field(default_factory=list)
    rejected_count: int = 0
    metrics_line: str = ""


@dataclass
class ReviewEdits:
    """User-supplied tweaks applied at publish time. Only fields that are
    set override the draft; missing keys mean "use the draft value"."""
    summary: Optional[str] = None
    event: Optional[str] = None
    # Map of DraftComment.id -> override body (None = keep original).
    comment_overrides: dict[str, str] = field(default_factory=dict)
    # Set of DraftComment.id to discard (drop from the published review).
    discarded_comment_ids: set[str] = field(default_factory=set)


_FENCED_BLOCK_RE = re.compile(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```")
_TAGGED_DIFF_LINE_RE = re.compile(r"^\[(R|L)\s*(\d+)\] ")
_PARSE_PREVIEW_CHARS = 500


def _content_preview(text: str, limit: int = _PARSE_PREVIEW_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [+{len(text) - limit} chars truncated]"


def _extract_json(content: Optional[str]) -> dict[str, Any]:
    """Forgiving JSON extraction. Tries, in order:

    1. Direct parse of the stripped content.
    2. Each fenced ``` block (with or without a `json` language tag).
    3. ``raw_decode`` starting at every ``{`` position, picking the first
       attempt that yields a JSON object.

    The third pass means trailing prose after the JSON ("Hope this helps!")
    or surrounding chatter ("Sure, here you go: {...}") doesn't break us.
    Raises ValueError with a length-and-preview diagnostic when nothing parses.
    """
    if not content:
        raise ValueError("LLM response was empty")
    text = content.strip()
    if not text:
        raise ValueError("LLM response was whitespace only")

    decoder = json.JSONDecoder()

    try:
        result = decoder.decode(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    for match in _FENCED_BLOCK_RE.finditer(text):
        candidate = match.group(1).strip()
        if not candidate:
            continue
        try:
            result = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict):
            return result

    for idx in (i for i, ch in enumerate(text) if ch == "{"):
        try:
            result, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict):
            return result

    raise ValueError(
        f"LLM response did not contain a JSON object "
        f"(length={len(content)} chars, preview={_content_preview(text)!r})"
    )


@dataclass
class _DiffChunk:
    text: str
    parsed_by_path: dict[str, ParsedFile]
    visible_positions: dict[str, set[tuple[str, int]]]


def _copy_positions_map(
    positions: dict[str, set[tuple[str, int]]]
) -> dict[str, set[tuple[str, int]]]:
    return {path: set(vals) for path, vals in positions.items()}


def _extract_visible_positions(text: str) -> set[tuple[str, int]]:
    visible: set[tuple[str, int]] = set()
    for line in text.splitlines():
        m = _TAGGED_DIFF_LINE_RE.match(line)
        if not m:
            continue
        side = "RIGHT" if m.group(1) == "R" else "LEFT"
        visible.add((side, int(m.group(2))))
    return visible


def _split_annotated_block(
    path: str, parsed: ParsedFile, max_chars: int
) -> list[tuple[str, set[tuple[str, int]]]]:
    """Split a file diff into one or more prompt-sized blocks.

    We prefer hunk-aligned splits; if a single hunk is still too large,
    we fall back to line-based splits inside that hunk and repeat the
    hunk header in each fragment so the model keeps local context.
    """
    header = f"--- a/{path}\n+++ b/{path}\n"
    full = f"{header}{parsed.annotated}\n"
    if len(full) <= max_chars:
        return [(full, _extract_visible_positions(parsed.annotated))]

    budget = max(1, max_chars - len(header) - 1)
    raw_lines = parsed.annotated.splitlines(keepends=True)
    sections: list[list[str]] = []
    current: list[str] = []
    for line in raw_lines:
        if line.startswith("@@") and current:
            sections.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append(current)

    units: list[str] = []
    for section_lines in sections:
        section_text = "".join(section_lines)
        if len(section_text) <= budget:
            units.append(section_text)
            continue

        prefix = section_lines[0] if section_lines and section_lines[0].startswith("@@") else ""
        remainder = section_lines[1:] if prefix else section_lines
        chunk_lines: list[str] = []
        chunk_len = len(prefix)
        for line in remainder:
            if chunk_lines and chunk_len + len(line) > budget:
                units.append(prefix + "".join(chunk_lines))
                chunk_lines = []
                chunk_len = len(prefix)
            chunk_lines.append(line)
            chunk_len += len(line)
        if chunk_lines or prefix:
            units.append(prefix + "".join(chunk_lines))

    blocks: list[tuple[str, set[tuple[str, int]]]] = []
    current_units: list[str] = []
    current_len = 0
    for unit in units:
        if current_units and current_len + len(unit) > budget:
            body = "".join(current_units)
            block = f"{header}{body}"
            if not block.endswith("\n"):
                block += "\n"
            blocks.append((block, _extract_visible_positions(body)))
            current_units = [unit]
            current_len = len(unit)
        else:
            current_units.append(unit)
            current_len += len(unit)
    if current_units:
        body = "".join(current_units)
        block = f"{header}{body}"
        if not block.endswith("\n"):
            block += "\n"
        blocks.append((block, _extract_visible_positions(body)))
    return blocks


def _build_annotated_diff_chunks(
    files: list[dict],
    max_chars: int,
    skip_paths: set[str],
) -> tuple[list[_DiffChunk], list[str]]:
    """Build one or more review chunks without dropping diff content.

    `max_chars` is now a per-chunk budget, not a whole-PR truncation
    threshold. Large PRs are split across multiple prompts so every
    changed hunk remains reviewable.
    """
    chunks: list[_DiffChunk] = []
    skipped: list[str] = []
    current_parts: list[str] = []
    current_parsed: dict[str, ParsedFile] = {}
    current_visible: dict[str, set[tuple[str, int]]] = {}
    current_len = 0

    for f in files:
        path = f.get("filename")
        patch = f.get("patch")
        if not path or not patch:
            continue
        if path in skip_paths:
            skipped.append(path)
            continue

        parsed = parse_patch(path, patch)
        for block, visible in _split_annotated_block(path, parsed, max_chars):
            if current_parts and current_len + len(block) > max_chars:
                chunks.append(
                    _DiffChunk(
                        text="".join(current_parts),
                        parsed_by_path=dict(current_parsed),
                        visible_positions=_copy_positions_map(current_visible),
                    )
                )
                current_parts = []
                current_parsed = {}
                current_visible = {}
                current_len = 0

            current_parts.append(block)
            current_len += len(block)
            current_parsed[path] = parsed
            current_visible.setdefault(path, set()).update(visible)

    if current_parts:
        chunks.append(
            _DiffChunk(
                text="".join(current_parts),
                parsed_by_path=dict(current_parsed),
                visible_positions=_copy_positions_map(current_visible),
            )
        )

    return chunks, skipped


def _validate_comments(
    raw_comments: list[dict[str, Any]],
    visible_positions: dict[str, set[tuple[str, int]]],
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
        positions = visible_positions.get(path)
        if not positions or (side, line) not in positions:
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


def _merge_metrics(total: "_AggregateMetrics", part: "_AggregateMetrics") -> None:
    total.turns += part.turns
    total.tool_calls += part.tool_calls
    total.latency_seconds += part.latency_seconds
    total.prompt_tokens += part.prompt_tokens
    total.completion_tokens += part.completion_tokens


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


def _build_runner_context(
    *,
    all_files: list[dict],
    skipped: list[str],
    chunk_index: int,
    chunk_total: int,
) -> Optional[str]:
    notes: list[str] = []
    if chunk_total > 1:
        notes.append(
            f"This PR diff was split into {chunk_total} chunks because the full diff exceeded "
            f"the per-call budget. You are reviewing chunk {chunk_index} of {chunk_total}."
        )
        notes.append("Only place inline comments on lines shown in this chunk's diff below.")
        changed_paths = [f.get("filename") for f in all_files if f.get("filename")]
        if changed_paths:
            notes.append("Changed files in the full PR:\n- " + "\n- ".join(changed_paths))
    if skipped:
        notes.append(
            "The following files were excluded from this review by the target repo's "
            "`.ai/context-script`. Do NOT review their contents and do NOT place inline "
            "comments on them. Refer to REPO-PROVIDED CONTEXT for the reason and any related "
            "guidance:\n- " + "\n- ".join(skipped)
        )
    return "\n\n".join(notes) if notes else None


def _merge_chunk_summaries(
    summaries: list[tuple[int, str]], chunk_total: int
) -> str:
    clean = [
        (idx, summary.strip())
        for idx, summary in summaries
        if isinstance(summary, str) and summary.strip()
    ]
    if not clean:
        return ""
    if chunk_total <= 1 and len(clean) == 1:
        return clean[0][1]
    parts = [
        f"Review ran in {chunk_total} chunks because the PR exceeded the per-call diff budget."
    ]
    for chunk_idx, summary in clean:
        parts.append(f"Chunk {chunk_idx}:\n{summary}")
    return "\n\n".join(parts)


def _merge_chunk_event(events: list[str], comments_count: int) -> str:
    normalized = [e for e in events if e in ("COMMENT", "REQUEST_CHANGES", "APPROVE")]
    if any(e == "REQUEST_CHANGES" for e in normalized):
        return "REQUEST_CHANGES"
    if normalized and all(e == "APPROVE" for e in normalized) and comments_count == 0:
        return "APPROVE"
    return "COMMENT"


def _run_agentic_loop(
    llm: ChatCompletionClient,
    initial_messages: list[dict[str, Any]],
    *,
    cfg: Config,
    tool_env: Optional[ToolEnv],
    emit: Optional[Callable[[str, str], None]] = None,
) -> tuple[ChatResult, _AggregateMetrics]:
    """Run a tool-augmented chat loop until the model emits a final
    (non-tool) response, falling back to a final non-tool turn if the
    iteration budget is exhausted.

    Returns the *last* ChatResult (whose ``content`` carries the JSON
    review) and an aggregate-metrics struct."""
    messages = list(initial_messages)
    metrics = _AggregateMetrics()
    tools_arg = TOOL_SPECS if tool_env is not None else None

    # llm_client emits ("token", ...) for response content,
    # ("reasoning", ...) for chain-of-thought deltas, and
    # ("stream_metrics", ...) with a char-based estimate of the current
    # turn's output every ~0.75s. We wrap `emit` so stream_metrics is
    # overlaid on prior-turn cumulative totals and surfaced as the
    # regular "metrics" event the UI already consumes.
    chunk_cb: Optional[Callable[[str, str], None]] = _wrap_chunk_cb(emit, metrics)

    # Several inference stacks (Kimi-K2 on HF Router, vLLM with some
    # tool parsers, etc.) reject `response_format` + `tools` in the same
    # request. We omit response_format whenever tools are in play and
    # rely on the system prompt's "output ONLY a single JSON object"
    # instruction plus _extract_json's forgiving parsing for the final
    # answer.
    response_format = None if tools_arg else {"type": "json_object"}

    # ``tool_max_iterations <= 0`` means "no cap" — keep looping as long
    # as the model keeps asking for tools. In practice it stops by
    # emitting a non-tool reply; this branch is only here so a runaway
    # loop is bounded when the user opts in.
    iter_cap: Optional[int] = (
        cfg.tool_max_iterations if cfg.tool_max_iterations > 0 else None
    )
    iteration = 0
    while True:
        iteration += 1
        if iter_cap is not None and iteration > iter_cap:
            break
        label = f"{iteration}/{iter_cap}" if iter_cap is not None else f"{iteration}"
        log.info("Agent loop iteration %s", label)
        if emit is not None:
            emit("step", f"llm:{label}")
            emit("log", f"LLM turn {label}")
        chat = llm.complete(
            messages,
            response_format=response_format,
            max_tokens=cfg.llm_max_tokens,
            tools=tools_arg,
            tool_choice="auto" if tools_arg else None,
            chunk_callback=chunk_cb,
        )
        metrics.turns += 1
        metrics.latency_seconds += chat.latency_seconds
        if chat.prompt_tokens is not None:
            metrics.prompt_tokens += chat.prompt_tokens
        if chat.completion_tokens is not None:
            metrics.completion_tokens += chat.completion_tokens
        _emit_metrics(emit, metrics)

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
            if emit is not None:
                emit("tool", f"{tc.name}({_summarize_args_str(tc.arguments)})")
                _emit_metrics(emit, metrics)
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
    # Only reachable when ``tool_max_iterations > 0``.
    log.warning(
        "Tool budget (%d) exhausted; asking model for a final review without tools",
        cfg.tool_max_iterations,
    )
    if emit is not None:
        emit("log", "Tool budget exhausted; asking for a final review without tools")
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
        chunk_callback=chunk_cb,
    )
    metrics.turns += 1
    metrics.latency_seconds += chat.latency_seconds
    if chat.prompt_tokens is not None:
        metrics.prompt_tokens += chat.prompt_tokens
    if chat.completion_tokens is not None:
        metrics.completion_tokens += chat.completion_tokens
    _emit_metrics(emit, metrics)
    return chat, metrics


def _wrap_chunk_cb(
    emit: Optional[Callable[[str, str], None]],
    metrics: "_AggregateMetrics",
) -> Optional[Callable[[str, str], None]]:
    """Forward token/reasoning chunks unchanged, but turn the
    stream-side ``stream_metrics`` estimates into full ``metrics``
    payloads with prior-turn totals overlaid. This way the UI counter
    grows monotonically across turns instead of resetting each time a
    new stream starts."""
    if emit is None:
        return None

    def cb(kind: str, text: str) -> None:
        if kind != "stream_metrics":
            emit(kind, text)
            return
        try:
            live = json.loads(text)
            this_in = int(live.get("in", 0))
            this_out = int(live.get("out", 0))
            this_seconds = float(live.get("seconds", 0.0))
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        cum_in = metrics.prompt_tokens + this_in
        cum_out = metrics.completion_tokens + this_out
        cum_seconds = metrics.latency_seconds + this_seconds
        rate = cum_out / cum_seconds if cum_seconds > 0 else 0.0
        emit(
            "metrics",
            json.dumps(
                {
                    "in": cum_in,
                    "out": cum_out,
                    "rate": round(rate, 1),
                    "seconds": round(cum_seconds, 1),
                    "turns": metrics.turns + 1,
                    "tools": metrics.tool_calls,
                }
            ),
        )

    return cb


def _emit_metrics(
    emit: Optional[Callable[[str, str], None]], metrics: "_AggregateMetrics"
) -> None:
    if emit is None:
        return
    rate = (
        metrics.completion_tokens / metrics.latency_seconds
        if metrics.latency_seconds > 0
        else 0.0
    )
    payload = json.dumps(
        {
            "in": metrics.prompt_tokens,
            "out": metrics.completion_tokens,
            "rate": round(rate, 1),
            "seconds": round(metrics.latency_seconds, 1),
            "turns": metrics.turns,
            "tools": metrics.tool_calls,
        }
    )
    emit("metrics", payload)


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


def _summarize_args_str(raw: str, limit: int = 200) -> str:
    """Like _summarize_args but takes the raw arguments string from the
    model — handy for emitting a breadcrumb before JSON-parsing."""
    s = (raw or "").replace("\n", " ").strip()
    return s if len(s) <= limit else s[:limit] + "..."


def _summarize_rejected_comments(rejected: list[dict[str, Any]], max_items: int = 5) -> str:
    """Render the first few rejected comments as `path:line` refs so the
    log line stays short even if the model emitted dozens of bogus inline
    comments. Without this, the full payloads bloat the action log."""
    refs: list[str] = []
    for c in rejected[:max_items]:
        path = c.get("path", "?")
        line = c.get("line", "?")
        refs.append(f"{path}:{line}")
    if len(rejected) > max_items:
        refs.append(f"...(+{len(rejected) - max_items} more)")
    return ", ".join(refs)


def _load_review_rules(gh: GitHubClient, owner: str, repo: str, pr: dict, cfg: Config) -> str:
    default_branch = pr.get("base", {}).get("repo", {}).get("default_branch") or "main"
    try:
        content = gh.get_file_contents(owner, repo, cfg.review_rules_path, ref=default_branch)
    except Exception:
        log.exception("failed to fetch review rules")
        content = None
    return content or cfg.default_review_rules


class _UnparseableLLMOutput(Exception):
    """The LLM returned content we couldn't parse as JSON. Carries the raw
    content + finish_reason + metrics line so the caller can render an
    error to whatever surface they own (GitHub comment, web UI, etc.)."""

    def __init__(self, content: str, finish_reason: Optional[str], metrics_line: str):
        super().__init__("unparseable LLM output")
        self.content = content
        self.finish_reason = finish_reason
        self.metrics_line = metrics_line


def prepare_review(
    cfg: Config,
    gh: GitHubClient,
    req: ReviewRequest,
    *,
    chunk_callback: Optional[Callable[[str, str], None]] = None,
) -> Optional[ReviewDraft]:
    """Run the LLM review pipeline and return a ReviewDraft ready to be
    published (or edited then published).

    `chunk_callback(kind, text)` is invoked as work progresses so callers
    (e.g. the web UI) can surface live activity. Kinds:
      - "log": a human-readable status line
      - "token": a slice of the assistant's streamed content
      - "tool": a tool-call breadcrumb

    Returns None when the PR has no reviewable diff (in which case a
    notice is posted to the PR, matching the original behavior).

    Raises _UnparseableLLMOutput when the model's final reply can't be
    JSON-parsed; the caller decides how to surface that.
    """

    def _emit(kind: str, text: str) -> None:
        if chunk_callback is not None:
            try:
                chunk_callback(kind, text)
            except Exception:
                log.debug("chunk_callback raised; suppressing", exc_info=True)

    log.info(
        "Starting review of %s/%s#%d (triggered by @%s)",
        req.owner,
        req.repo,
        req.number,
        req.commenter,
    )
    _emit("log", f"Starting review of {req.owner}/{req.repo}#{req.number}")

    if req.trigger_comment_id:
        try:
            gh.add_reaction_to_issue_comment(
                req.owner, req.repo, req.trigger_comment_id, "eyes"
            )
        except Exception:
            log.debug("reaction failed (non-fatal)", exc_info=True)

    _emit("step", "fetch")
    pr = gh.get_pr(req.owner, req.repo, req.number)
    files = gh.get_pr_files(req.owner, req.repo, req.number)
    _emit("log", f"Fetched PR with {len(files)} changed file(s)")

    _emit("step", "context")
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
        _emit(
            "log",
            f"Context script: {len(extra_context or '')} chars context, {len(skip_paths)} skip(s)",
        )

    diff_chunks, skipped = _build_annotated_diff_chunks(files, cfg.max_diff_chars, skip_paths)
    if skipped:
        log.info("Excluded %d file(s) per .ai/context-script: %s", len(skipped), skipped)
    if not diff_chunks:
        gh.post_issue_comment(
            req.owner,
            req.repo,
            req.number,
            "No reviewable diff hunks were found (binary files, empty patches, or all files excluded by .ai/context-script).",
        )
        _emit("log", "No reviewable diff hunks; posted notice and stopped")
        return None
    if len(diff_chunks) > 1:
        log.info(
            "Split oversized PR diff into %d review chunks (budget=%d chars/chunk)",
            len(diff_chunks),
            cfg.max_diff_chars,
        )
        _emit(
            "log",
            f"Split oversized PR diff into {len(diff_chunks)} review chunk(s)",
        )

    head_sha = (pr.get("head") or {}).get("sha")
    if not head_sha:
        raise RuntimeError("PR payload missing head.sha")

    review_rules = _load_review_rules(gh, req.owner, req.repo, pr, cfg)
    tool_env = _make_tool_env(cfg)

    llm = ChatCompletionClient(
        cfg.llm_api_base,
        cfg.llm_api_key,
        cfg.llm_model,
        bill_to=cfg.llm_bill_to,
        stream=cfg.llm_stream,
    )
    system_prompt = build_system_prompt(review_rules, tools_enabled=tool_env is not None)
    total_metrics = _AggregateMetrics()
    all_valid: list[dict[str, Any]] = []
    all_events: list[str] = []
    all_summaries: list[tuple[int, str]] = []
    rejected_count = 0

    for idx, chunk in enumerate(diff_chunks, start=1):
        runner_context = _build_runner_context(
            all_files=files,
            skipped=skipped,
            chunk_index=idx,
            chunk_total=len(diff_chunks),
        )
        user_prompt = build_user_prompt(
            repo_full_name=f"{req.owner}/{req.repo}",
            number=req.number,
            title=pr.get("title") or "",
            body=pr.get("body") or "",
            author=(pr.get("user") or {}).get("login") or "unknown",
            commenter=req.commenter,
            trigger_comment=req.trigger_comment_body,
            diff=chunk.text,
            extra_context=extra_context,
            runner_context=runner_context,
        )

        _emit("step", "llm")
        if len(diff_chunks) > 1:
            _emit("log", f"Calling LLM for diff chunk {idx}/{len(diff_chunks)}…")
        else:
            _emit("log", "Calling LLM…")
        chat, chunk_metrics = _run_agentic_loop(
            llm,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            cfg=cfg,
            tool_env=tool_env,
            emit=_emit,
        )
        _merge_metrics(total_metrics, chunk_metrics)

        try:
            result = _extract_json(chat.content)
        except ValueError as exc:
            metrics_line = _format_aggregated_metrics(total_metrics)
            log.error(
                "could not parse LLM output as JSON: %s "
                "(content_chars=%d, finish_reason=%s, prompt_tokens=%s, completion_tokens=%s)",
                exc,
                len(chat.content or ""),
                chat.finish_reason,
                chat.prompt_tokens,
                chat.completion_tokens,
            )
            raise _UnparseableLLMOutput(
                content=chat.content or "",
                finish_reason=chat.finish_reason,
                metrics_line=metrics_line,
            ) from exc

        summary = (result.get("summary") or "").strip()
        event = result.get("event") or cfg.review_event
        if event not in ("COMMENT", "REQUEST_CHANGES", "APPROVE"):
            event = cfg.review_event
        if event == "APPROVE" and not cfg.allow_approve:
            log.info(
                "Downgrading APPROVE to COMMENT (Actions tokens cannot approve; set ALLOW_APPROVE=1 in App mode to permit)"
            )
            event = "COMMENT"
        all_events.append(event)
        if summary:
            all_summaries.append((idx, summary))

        valid, rejected = _validate_comments(
            result.get("comments") or [], chunk.visible_positions
        )
        rejected_count += len(rejected)
        if rejected:
            log.warning(
                "Dropped %d invalid comment(s) from chunk %d/%d (referenced lines not in visible diff or malformed): %s",
                len(rejected),
                idx,
                len(diff_chunks),
                _summarize_rejected_comments(rejected),
            )
        for c in valid:
            c["_parsed"] = chunk.parsed_by_path.get(c["path"])
        all_valid.extend(valid)

    metrics_line = _format_aggregated_metrics(total_metrics)
    _emit("log", f"LLM done: {metrics_line}")

    summary = _merge_chunk_summaries(all_summaries, len(diff_chunks))
    event = _merge_chunk_event(all_events, len(all_valid))

    draft_comments: list[DraftComment] = []
    seen_comments: set[tuple[str, str, int, str]] = set()
    for i, c in enumerate(all_valid):
        dedupe_key = (c["path"], c["side"], c["line"], c["body"])
        if dedupe_key in seen_comments:
            continue
        seen_comments.add(dedupe_key)
        parsed = c.get("_parsed")
        hunk = (
            extract_hunk_snippet(parsed.raw_patch, c["side"], c["line"])
            if isinstance(parsed, ParsedFile)
            else []
        )
        draft_comments.append(
            DraftComment(
                id=f"c{i}",
                path=c["path"],
                side=c["side"],
                line=c["line"],
                body=c["body"],
                diff_hunk=hunk,
            )
        )

    _emit("step", "done")
    return ReviewDraft(
        owner=req.owner,
        repo=req.repo,
        number=req.number,
        head_sha=head_sha,
        summary=summary,
        event=event,
        comments=draft_comments,
        rejected_count=rejected_count,
        metrics_line=metrics_line,
    )


def publish_review(
    cfg: Config,
    gh: GitHubClient,
    draft: ReviewDraft,
    *,
    edits: Optional[ReviewEdits] = None,
) -> None:
    """Apply optional user edits to a ReviewDraft and post it via the
    GitHub reviews API. Mirrors the body-formatting rules previously
    inlined in run_review (persona header, dropped-comments note,
    metrics footer)."""
    edits = edits or ReviewEdits()

    summary = edits.summary if edits.summary is not None else draft.summary
    event = edits.event or draft.event
    if event not in ("COMMENT", "REQUEST_CHANGES", "APPROVE"):
        event = draft.event
    if event == "APPROVE" and not cfg.allow_approve:
        log.info(
            "Downgrading APPROVE to COMMENT (Actions tokens cannot approve; set ALLOW_APPROVE=1 in App mode to permit)"
        )
        event = "COMMENT"

    comments_payload: list[dict[str, Any]] = []
    for c in draft.comments:
        if c.id in edits.discarded_comment_ids:
            continue
        body = edits.comment_overrides.get(c.id, c.body)
        if not isinstance(body, str) or not body.strip():
            continue
        comments_payload.append(
            {"path": c.path, "side": c.side, "line": c.line, "body": body}
        )

    body = summary or "(no overall summary provided)"
    if cfg.persona_header:
        body = f"{cfg.persona_header}\n\n{body}"
    if draft.rejected_count:
        body += (
            f"\n\n_Note: {draft.rejected_count} suggested inline comment(s) "
            "were dropped because they referenced lines not present in the diff._"
        )
    if draft.metrics_line:
        body += f"\n\n_{draft.metrics_line}_"

    gh.create_review(
        draft.owner,
        draft.repo,
        draft.number,
        commit_id=draft.head_sha,
        body=body,
        comments=comments_payload,
        event=event,
    )
    log.info(
        "Posted review on %s/%s#%d (%d inline, event=%s, %s)",
        draft.owner,
        draft.repo,
        draft.number,
        len(comments_payload),
        event,
        draft.metrics_line,
    )


def run_review(cfg: Config, gh: GitHubClient, req: ReviewRequest) -> None:
    """Webhook + Action entry point. Unchanged behavior: prepares the
    review, then immediately publishes it. Renders a fallback issue
    comment if the LLM output is unparseable."""
    try:
        draft = prepare_review(cfg, gh, req)
    except _UnparseableLLMOutput as exc:
        gh.post_issue_comment(
            req.owner,
            req.repo,
            req.number,
            f"Reviewer LLM returned unparseable output ({exc.metrics_line}, "
            f"finish_reason={exc.finish_reason}):\n\n```\n{exc.content[:3000]}\n```",
        )
        return
    if draft is None:
        return
    publish_review(cfg, gh, draft)
