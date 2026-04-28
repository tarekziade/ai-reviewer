"""Runs an optional repo-local script that can inject extra context into
the reviewer prompt.

The script lives in the target repo (typically at `.ai/context-script`),
must be executable, and is invoked with no arguments. We send a JSON
document on stdin describing the PR; whatever the script writes to stdout
is stitched into the user prompt as repo-supplied context.

Trust model: the script is whatever sits at the configured path in the
checked-out tree — in Action mode, that's the default branch (Actions
checks out the event ref, which for `issue_comment` is the default
branch). Treat it at the same trust level as `.ai/review-rules.md`.

stdin schema:
    {
      "title": str,
      "body":  str,
      "files": [
        {"path": str,
         "status": "added"|"modified"|"removed"|"renamed"|"copied"|"changed",
         "additions": int,
         "deletions": int,
         "previous_path": str | null}
      ]
    }
"""

import json
import logging
import os
import subprocess
from typing import Optional

log = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 8000


def _shape_files(files: list[dict]) -> list[dict]:
    out: list[dict] = []
    for f in files:
        path = f.get("filename")
        if not path:
            continue
        out.append(
            {
                "path": path,
                "status": f.get("status") or "modified",
                "additions": int(f.get("additions") or 0),
                "deletions": int(f.get("deletions") or 0),
                "previous_path": f.get("previous_filename"),
            }
        )
    return out


def run_context_script(
    script_path: str,
    *,
    title: str,
    body: str,
    files: list[dict],
    timeout_seconds: int,
    cwd: Optional[str] = None,
) -> Optional[str]:
    """Run `script_path` and return its stdout, or None if it should be skipped.

    A return of None means "don't add anything to the prompt" — for any of:
    not configured, file missing, not executable, timed out, crashed,
    non-zero exit, or empty output. We never raise to the caller; failures
    are logged and swallowed because a broken context hook should not break
    the review.
    """
    if not script_path:
        return None
    base = cwd or os.getcwd()
    full = script_path if os.path.isabs(script_path) else os.path.join(base, script_path)
    if not os.path.isfile(full):
        log.debug("context script %s not found, skipping", full)
        return None
    if not os.access(full, os.X_OK):
        log.warning("context script %s exists but is not executable; skipping", full)
        return None

    payload = json.dumps(
        {
            "title": title or "",
            "body": body or "",
            "files": _shape_files(files),
        }
    )

    try:
        proc = subprocess.run(
            [full],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=base,
        )
    except subprocess.TimeoutExpired:
        log.warning("context script %s timed out after %ds", full, timeout_seconds)
        return None
    except Exception:
        log.exception("context script %s failed to launch", full)
        return None

    if proc.stderr:
        log.info("context script stderr: %s", proc.stderr.strip()[:1000])
    if proc.returncode != 0:
        log.warning(
            "context script %s exited %d; ignoring output", full, proc.returncode
        )
        return None

    out = (proc.stdout or "").strip()
    if not out:
        return None
    if len(out) > MAX_OUTPUT_CHARS:
        out = (
            out[:MAX_OUTPUT_CHARS]
            + f"\n[... truncated, {len(out) - MAX_OUTPUT_CHARS} chars omitted ...]"
        )
    return out
