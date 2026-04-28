import re
from dataclasses import dataclass, field

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


def parse_patch(path: str, patch: str) -> ParsedFile:
    """Annotate a file-level unified diff patch and collect the lines that can
    legally receive an inline comment via the GitHub Review API.

    Added/context lines are addressable on the RIGHT side (new file line
    numbers). Deleted lines are addressable on the LEFT side (old file line
    numbers).
    """
    parsed = ParsedFile(path=path)
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
