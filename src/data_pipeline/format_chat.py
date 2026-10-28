"""Shared prompt formatter: teacher labeling, eval, training (M2) and inference (M5) all call this,
so there is no train/serve skew (PROJECT_PLAN.md §3.2)."""
import re

SYSTEM_PROMPT = (
    "You are an automated code review engine. Analyze the git diff for critical defects ONLY "
    "(SECURITY, BUG, PERFORMANCE, RESOURCE_LEAK). Ignore formatting, naming, and style. "
    "Respond strictly in valid JSON matching the schema."
)

_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def number_diff(diff: str) -> str:
    """Prefix context/added lines with their new-file line number (`42| +code`).

    Removed lines get a blank gutter (`   | -code`) because `-` lines do not advance the
    new-file counter. File headers, `@@` lines and `\\ No newline` markers pass through as-is.
    """
    out, new = [], None  # new is None until the first hunk header (so ---/+++ stay headers)
    for line in diff.splitlines():
        m = _HUNK.match(line)
        if m:
            new = int(m.group(1))
            out.append(line)
        elif new is None or line.startswith("\\"):
            out.append(line)
        elif line.startswith("-"):
            out.append(f"   | {line}")
        else:  # '+' or context (' ')
            out.append(f"{new}| {line}")
            new += 1
    return "\n".join(out)


def build_messages(file_path: str, diff: str) -> list[dict]:
    """System + user turns only; the assistant turn (label JSON) is added by the caller."""
    user = f"File: {file_path}\n<git_diff>\n{number_diff(diff)}\n</git_diff>"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

