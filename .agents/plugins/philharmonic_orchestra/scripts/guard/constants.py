"""Constants, path anchors, regex patterns, and shell/command helpers for Philharmonic guard."""
from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent.parent

ALLOWED_SUBAGENT_MODELS = {"flash", "flash_lite"}
DEFAULT_MODEL_BY_TYPE = {
    "scout": "flash_lite",
    "research": "flash",
    "researcher": "flash",
    "implementer": "flash",
}

MUTATION_VERBS_RE = re.compile(
    r"\b(edit|modify|write|create|delete|remove|replace|patch|implement|refactor)\b",
    re.I,
)
NUMBERED_RULE_RE = re.compile(r"(?m)^\s*[0-9]+[.)]\s+")
ALLOWED_FILES_HEADER_RE = re.compile(r"(?m)^\s*ALLOWED_FILES:\s*(.+)$", re.I)
UUID_RE = re.compile(
    r"\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b",
    re.I,
)
SUBAGENT_SPAWN_RE = re.compile(
    r'(?:\\*["\']?conversationId\\*["\']?\s*[:=]\s*\\*["\']?)([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})',
    re.I,
)

PENDING_SPAWN_TTL_SECONDS = 120
MAX_STATE_INVOCATION_HISTORY = 25
MAX_STATE_LIST_ENTRIES = 100

MAX_DIRECT_MUTATION_FILES = 2
MAX_DIRECT_COMMANDS = 2
MAX_DIRECT_READ_FILES = 2
MAX_SUBAGENT_REPORT_CHARS = 1600
MAX_REPORT_BULLET_LINES = 10
MAX_CONSECUTIVE_FAILURES = 2

REPORT_BULLET_RE = re.compile(r"^[-*]\s+\S+")
REPORT_LOG_URI_RE = re.compile(r"file://\S+\.log\b", re.I)
REPORT_LINE_PATTERNS = (
    re.compile(r"^STATUS:\s*(?:COMPLETED|BLOCKED|FAILED)\b"),
    re.compile(r"^FILES_CHANGED:\s*\S+"),
    re.compile(r"^VERIFICATION:\s*\S+"),
    re.compile(r"^NOTES:\s*\S+"),
)

READ_TOOLS = {"read_file", "list_dir", "view_file"}
SEARCH_TOOLS = {"grep_search", "find_by_name"}
MUTATION_TOOLS = {
    "write_file",
    "edit_file",
    "write_to_file",
    "replace_file_content",
    "multi_replace_file_content",
}
PATH_ARG_KEYS = (
    "Path",
    "TargetFile",
    "AbsolutePath",
    "FilePath",
    "file_path",
    "SearchPath",
    "SearchDirectory",
    "DirectoryPath",
)

FORBIDDEN_DIRECT_READ_RE = re.compile(
    r"(?:^|/)transcript(?:_full|[0-9a-zA-Z_\-]*)?\.jsonl$|\.log$", re.I
)

GIT_COMMAND_RE = re.compile(r"^\s*git\b")
TEST_CMD_START_RE = re.compile(
    r"^\s*(?:(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*"
    r"(?:(?:python3?|py|uv\s+run|poetry\s+run|pipenv\s+run|npx|bunx)\s+(?:-m\s+)?)?)?"
    r"(?:pytest|unittest|cargo\s+test|npm\s+test|pnpm\s+test|yarn\s+test|go\s+test|jest|vitest|tox|nox)\b"
)

DOC_EXTENSIONS = {".md", ".txt", ".rst", ".adoc"}
TEST_FILE_RE = re.compile(
    r"(^|/)(tests?|__tests__)/|(^|/)(test_[^/]+\.py|[^/]+_test\.(py|go|rs)|[^/]+\.(test|spec)\.[jt]sx?)$",
    re.I,
)
COMMENT_LINE_RE = re.compile(r"^\s*(?:#|//)")
TEST_DEF_RE = re.compile(
    r"(?m)^\s*(?:async\s+def|def)\s+test_[A-Za-z0-9_]+\s*\(|^\s*(?:it|test)\s*\("
)
ASSERTION_RE = re.compile(
    r"\b(?:assert\b|self\.assert[A-Za-z0-9_]*\s*\(|expect\s*\(|should\.)"
)
TAMPER_SKIP_RE = re.compile(
    r"(?m)^\s*(?:@unittest\.(?:skip|expectedFailure)\b|@pytest\.mark\.(?:skip|xfail)\b|"
    r"pytest\.(?:skip|xfail)\s*\(|self\.skipTest\s*\(|unittest\.SkipTest\b|"
    r"(?:it|test|describe)\.(?:skip|only)\s*\(|(?:xit|xtest|xdescribe)\s*\(|#\s*\[ignore\b)"
)
TRIVIAL_ASSERT_RE = re.compile(
    r"(?m)^\s*(?:assert\s+True\b|self\.assertTrue\s*\(\s*True\s*\))"
)
ALLOWED_SYSTEM_PATH_PREFIXES = (
    "/dev/",
    "/usr/",
    "/bin/",
    "/lib/",
    "/lib64/",
    "/proc/",
    "/tmp/",
)

SLASH_COMMAND_RE = re.compile(r"(?:^|[\s`'\"])/philharmonic(?:\b|$)", re.I)


def _find_project_root() -> Path:
    """Resolve the project workspace root from the script location."""
    current = SCRIPT_DIR
    for parent in [current] + list(current.parents):
        if parent.name == ".agents":
            return parent.parent.resolve()
    return Path.cwd().resolve()


PROJECT_ROOT = _find_project_root()


def _split_shell_commands(cmd: str) -> List[str]:
    """Split a shell command string on unquoted &&, ||, ;, |, &, and newline delimiters."""
    if not cmd:
        return []
    parts: List[str] = []
    current: List[str] = []
    in_single = False
    in_double = False
    escaped = False
    i = 0
    n = len(cmd)
    while i < n:
        ch = cmd[i]
        if escaped:
            current.append(ch)
            escaped = False
            i += 1
            continue
        if ch == "\\" and not in_single:
            escaped = True
            current.append(ch)
            i += 1
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            current.append(ch)
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            current.append(ch)
            i += 1
            continue
        if not in_single and not in_double:
            if cmd[i : i + 2] in ("&&", "||"):
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                i += 2
                continue
            if ch == "&" and (
                (i > 0 and cmd[i - 1] in (">", "<"))
                or (i + 1 < n and cmd[i + 1] == ">")
            ):
                current.append(ch)
                i += 1
                continue
            if ch in (";", "|", "&", "\n"):
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                i += 1
                continue
        current.append(ch)
        i += 1
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _is_git_only_command(cmd: str) -> bool:
    """Return True only if every sub-command is a direct git command with no command substitution."""
    if not cmd or "$(" in cmd or "`" in cmd:
        return False
    sub_cmds = _split_shell_commands(cmd)
    if not sub_cmds:
        return False
    return all(bool(GIT_COMMAND_RE.match(c)) for c in sub_cmds)


def _is_test_command(cmd: str) -> bool:
    """Return True if any sub-command invokes a recognized test runner in command position."""
    if not cmd:
        return False
    sub_cmds = _split_shell_commands(cmd)
    return any(bool(TEST_CMD_START_RE.match(c)) for c in sub_cmds)


def _strip_comments(text: str) -> str:
    """Remove comments (C/JS block, Python triple-quoted standalone docstrings, single-line # and //) before counting test definitions or assertions."""
    if not text:
        return ""
    t = re.sub(r"/\*[\s\S]*?\*/", "", text)
    t = re.sub(r'(?m)^[ \t]*(?:"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')[ \t]*$', "", t)
    return "\n".join(
        line for line in t.splitlines() if not COMMENT_LINE_RE.match(line)
    )


def _extract_call_id(payload: Dict[str, Any]) -> str:
    """Extract a deterministic tool call identifier from toolCall.id or stepIdx."""
    tool_call = payload.get("toolCall") or {}
    tc_id = tool_call.get("id")
    if tc_id is not None and str(tc_id) != "":
        return str(tc_id)
    step_idx = payload.get("stepIdx")
    if step_idx is not None and str(step_idx) != "":
        return f"step:{step_idx}"
    return ""
