import re
from dataclasses import dataclass, field
from typing import Optional

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


@dataclass
class DiffPosition:
    """Where a line from a unified diff lives, for the GitHub review API."""

    side: str  # "RIGHT" for added/context, "LEFT" for deletion
    line: int  # file line number on that side


@dataclass
class ParsedFile:
    path: str
    # (side, line) -> True for quick membership checks
    valid_positions: set[tuple[str, int]] = field(default_factory=set)
    # Human-readable, line-numbered diff to feed the LLM
    annotated: str = ""
    # Raw patch text, retained so we can extract per-comment snippets
    # later (for the web UI's GitHub-style inline-comment view).
    raw_patch: str = ""


def parse_patch(path: str, patch: str) -> ParsedFile:
    """Annotate a file-level unified diff patch and collect the lines that can
    legally receive an inline comment via the GitHub Review API.

    Added/context lines are addressable on the RIGHT side (new file line
    numbers). Deleted lines are addressable on the LEFT side (old file line
    numbers).
    """
    parsed = ParsedFile(path=path, raw_patch=patch or "")
    if not patch:
        return parsed

    annotated_lines: list[str] = []
    new_line: int | None = None
    old_line: int | None = None

    for raw in patch.split("\n"):
        m = _HUNK_RE.match(raw)
        if m:
            old_line = int(m.group(1))
            new_line = int(m.group(2))
            annotated_lines.append(raw)
            continue

        if new_line is None or old_line is None:
            annotated_lines.append(raw)
            continue

        if raw.startswith("+") and not raw.startswith("+++"):
            parsed.valid_positions.add(("RIGHT", new_line))
            annotated_lines.append(f"[R{new_line:>5}] {raw}")
            new_line += 1
        elif raw.startswith("-") and not raw.startswith("---"):
            parsed.valid_positions.add(("LEFT", old_line))
            annotated_lines.append(f"[L{old_line:>5}] {raw}")
            old_line += 1
        elif raw.startswith(" "):
            parsed.valid_positions.add(("RIGHT", new_line))
            annotated_lines.append(f"[R{new_line:>5}] {raw}")
            new_line += 1
            old_line += 1
        else:
            annotated_lines.append(raw)

    parsed.annotated = "\n".join(annotated_lines)
    return parsed


@dataclass
class DiffSnippetLine:
    """One line in a per-comment diff snippet for the web UI. ``op`` is
    one of ``"+"``, ``"-"``, ``" "``. ``old`` / ``new`` are the old / new
    file line numbers (one of them is None for adds and deletes)."""

    op: str
    old: Optional[int]
    new: Optional[int]
    text: str
    is_target: bool = False


def extract_hunk_snippet(
    patch: str, side: str, target_line: int, *, before: int = 4, after: int = 1
) -> list[DiffSnippetLine]:
    """Return a small slice of the diff around the line a review comment
    is anchored to. Matches the way GitHub renders the "diff hunk"
    above an inline comment: a handful of context/added/deleted lines
    ending at (and shortly after) the commented line.

    Returns an empty list if (side, target_line) can't be located in
    ``patch`` — callers should fall back to no snippet."""
    if not patch:
        return []

    structured: list[DiffSnippetLine] = []
    new_line: Optional[int] = None
    old_line: Optional[int] = None

    for raw in patch.split("\n"):
        m = _HUNK_RE.match(raw)
        if m:
            old_line = int(m.group(1))
            new_line = int(m.group(2))
            continue
        if new_line is None or old_line is None:
            continue

        if raw.startswith("+") and not raw.startswith("+++"):
            is_target = side == "RIGHT" and new_line == target_line
            structured.append(
                DiffSnippetLine("+", None, new_line, raw[1:], is_target)
            )
            new_line += 1
        elif raw.startswith("-") and not raw.startswith("---"):
            is_target = side == "LEFT" and old_line == target_line
            structured.append(
                DiffSnippetLine("-", old_line, None, raw[1:], is_target)
            )
            old_line += 1
        elif raw.startswith(" "):
            is_target = side == "RIGHT" and new_line == target_line
            structured.append(
                DiffSnippetLine(" ", old_line, new_line, raw[1:], is_target)
            )
            new_line += 1
            old_line += 1
        # Ignore "\ No newline at end of file" and anything else.

    target_idx = next(
        (i for i, ln in enumerate(structured) if ln.is_target), None
    )
    if target_idx is None:
        return []

    start = max(0, target_idx - before)
    end = min(len(structured), target_idx + 1 + after)
    return structured[start:end]
