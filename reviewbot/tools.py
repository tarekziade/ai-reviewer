"""Sandboxed file-browsing tools exposed to the reviewer LLM.

Each tool operates against a single repository checkout root. All paths
are resolved via realpath and rejected if they escape the root, and a
small denylist of noisy directories (``.git``, ``node_modules``, etc.) is
hidden from the model so it can't accidentally pull in junk.

The module is decoupled from the LLM client: it just exposes
``TOOL_SPECS`` (the OpenAI-style schemas) and ``run_tool``, which takes a
function name + JSON-decoded arguments and returns a string the model
will see as the ``tool`` message content.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


# Per-call output cap: tool messages re-bill the prompt on every loop
# iteration, so unbounded output silently inflates cost and risks blowing
# the context window. 8 KB is enough for a couple hundred lines of code or
# a meaningful grep result; the model can always re-call with a narrower
# range if it wants more.
MAX_TOOL_OUTPUT_CHARS = 8000

# File reads are line-bounded as well; this is the upper bound enforced
# even if the model asks for more.
MAX_READ_LINES = 400

# Directories we never expose to the model. ``.git`` would let it read
# refs / object data; the rest are caches that bloat output without
# adding signal.
DENY_DIR_NAMES = frozenset(
    {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache",
     ".pytest_cache", ".ruff_cache", "dist", "build"}
)


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a slice of a file from the checked-out PR head. Use this "
                "to gather context that the diff alone doesn't show (callers, "
                "callees, surrounding class, related constants). Lines are 1-indexed."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the repository root.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to return (1-indexed). Defaults to 1.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": (
                            f"Last line to return, inclusive. Capped at "
                            f"{MAX_READ_LINES} lines per call."
                        ),
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": (
                "List the entries (files and subdirectories) of a directory in "
                "the checked-out PR head. Use this to discover where related "
                "code lives before reading specific files."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory relative to repo root. Defaults to the root.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search for a regex pattern across the checked-out PR head "
                "using git grep. Returns matches as `path:line:text`. Use this "
                "to find call sites, definitions, or references not visible in the diff."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern (extended POSIX, as accepted by git grep -E).",
                    },
                    "path": {
                        "type": "string",
                        "description": "Path-spec to scope the search. Defaults to the whole repo.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Cap on returned matches. Defaults to 50, hard max 200.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
]


@dataclass
class ToolEnv:
    """The repo root the tools operate against. Constructed once per
    review and threaded into ``run_tool``."""

    repo_root: str

    def __post_init__(self) -> None:
        self.repo_root = os.path.realpath(self.repo_root)
        if not os.path.isdir(self.repo_root):
            raise ValueError(f"repo_root is not a directory: {self.repo_root}")


def run_tool(env: ToolEnv, name: str, arguments: dict[str, Any]) -> str:
    """Dispatch a tool call. Always returns a string; on failure, returns
    a short error message rather than raising — the model sees it as the
    ``tool`` message content and can decide what to do next."""
    try:
        if name == "read_file":
            return _read_file(env, arguments)
        if name == "list_dir":
            return _list_dir(env, arguments)
        if name == "grep":
            return _grep(env, arguments)
        return f"error: unknown tool {name!r}"
    except _ToolError as exc:
        return f"error: {exc}"
    except Exception:  # pragma: no cover — defensive
        log.exception("tool %s crashed", name)
        return f"error: tool {name} crashed; see action log"


class _ToolError(Exception):
    pass


def _resolve_path(env: ToolEnv, raw: Any, *, default: str = ".") -> str:
    if raw is None or raw == "":
        raw = default
    if not isinstance(raw, str):
        raise _ToolError(f"path must be a string, got {type(raw).__name__}")
    # Reject obviously-hostile inputs early so the error message is clearer
    # than a realpath mismatch.
    if raw.startswith("/") or raw.startswith("~"):
        raise _ToolError("absolute or home-relative paths are not allowed")
    candidate = os.path.realpath(os.path.join(env.repo_root, raw))
    root_with_sep = env.repo_root + os.sep
    if candidate != env.repo_root and not candidate.startswith(root_with_sep):
        raise _ToolError(f"path {raw!r} escapes the repository root")
    parts = os.path.relpath(candidate, env.repo_root).split(os.sep)
    if any(p in DENY_DIR_NAMES for p in parts):
        raise _ToolError(f"path {raw!r} is inside a denylisted directory")
    return candidate


def _truncate(text: str, *, suffix_note: str = "") -> str:
    if len(text) <= MAX_TOOL_OUTPUT_CHARS:
        return text
    head = text[:MAX_TOOL_OUTPUT_CHARS]
    note = (
        f"\n[... truncated; output exceeded {MAX_TOOL_OUTPUT_CHARS} chars"
        + (f"; {suffix_note}" if suffix_note else "")
        + " ...]"
    )
    return head + note


def _read_file(env: ToolEnv, args: dict[str, Any]) -> str:
    path = _resolve_path(env, args.get("path"), default="")
    if not path or path == env.repo_root:
        raise _ToolError("read_file requires a file path, not the repo root")
    if not os.path.isfile(path):
        raise _ToolError(f"not a file: {os.path.relpath(path, env.repo_root)}")
    start = max(1, int(args.get("start_line") or 1))
    end_arg = args.get("end_line")
    end = int(end_arg) if end_arg is not None else start + MAX_READ_LINES - 1
    if end < start:
        raise _ToolError(f"end_line ({end}) must be >= start_line ({start})")
    end = min(end, start + MAX_READ_LINES - 1)
    rel = os.path.relpath(path, env.repo_root)
    lines: list[str] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f, start=1):
                if idx < start:
                    continue
                if idx > end:
                    break
                lines.append(f"{idx:>6}\t{line.rstrip(chr(10))}")
    except OSError as exc:
        raise _ToolError(f"could not read {rel}: {exc}") from exc
    if not lines:
        return f"{rel}: no lines in range {start}-{end}"
    body = "\n".join(lines)
    return _truncate(
        f"{rel} (lines {start}-{start + len(lines) - 1}):\n{body}",
        suffix_note="re-call with a narrower line range",
    )


def _list_dir(env: ToolEnv, args: dict[str, Any]) -> str:
    path = _resolve_path(env, args.get("path"), default=".")
    if not os.path.isdir(path):
        raise _ToolError(f"not a directory: {os.path.relpath(path, env.repo_root) or '.'}")
    rel = os.path.relpath(path, env.repo_root) or "."
    entries: list[str] = []
    try:
        for name in sorted(os.listdir(path)):
            if name in DENY_DIR_NAMES:
                continue
            full = os.path.join(path, name)
            if os.path.isdir(full):
                entries.append(f"{name}/")
            else:
                entries.append(name)
    except OSError as exc:
        raise _ToolError(f"could not list {rel}: {exc}") from exc
    if not entries:
        return f"{rel}: (empty)"
    return _truncate(f"{rel}:\n" + "\n".join(entries))


_SAFE_PATTERN_RE = re.compile(r"^[\x20-\x7E]+$")


def _grep(env: ToolEnv, args: dict[str, Any]) -> str:
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        raise _ToolError("pattern must be a non-empty string")
    if len(pattern) > 200:
        raise _ToolError("pattern is too long (max 200 chars)")
    if not _SAFE_PATTERN_RE.match(pattern):
        raise _ToolError("pattern contains non-printable or non-ASCII characters")
    path_arg = args.get("path")
    pathspec = _resolve_path(env, path_arg, default=".") if path_arg else env.repo_root
    rel_pathspec = os.path.relpath(pathspec, env.repo_root) or "."
    max_results = int(args.get("max_results") or 50)
    max_results = max(1, min(max_results, 200))
    cmd = [
        "git", "-C", env.repo_root, "grep", "-n", "-I", "-E",
        "--max-count=10", "--", pattern,
    ]
    if path_arg:
        cmd.append(rel_pathspec)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise _ToolError("grep timed out (20s)") from exc
    except FileNotFoundError as exc:
        raise _ToolError("git is not installed in the runner") from exc
    if proc.returncode == 1 and not proc.stdout:
        return f"no matches for {pattern!r} in {rel_pathspec}"
    if proc.returncode > 1:
        return f"error: git grep exited {proc.returncode}: {proc.stderr.strip()[:300]}"
    lines = proc.stdout.splitlines()
    truncated = len(lines) > max_results
    lines = lines[:max_results]
    body = "\n".join(lines) if lines else "(no output)"
    suffix = f"showing {len(lines)} of {len(lines)}+ matches" if truncated else ""
    header = f"git grep -E {pattern!r} -- {rel_pathspec}:"
    return _truncate(f"{header}\n{body}", suffix_note=suffix)
