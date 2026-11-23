"""Constants, path anchors, regex patterns, and test-audit helpers for Philharmonic guard."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict

from guard.model_tiers import (
    ALLOWED_BEST_OF_N_TIERS,
    ALLOWED_COUNCIL_PROPOSER_TIERS,
    ALLOWED_HIGH_REASONING_TIERS,
    ALLOWED_STANDARD_SUBAGENT_TIERS,
    BLIND_JUDGE_LEAK_PATTERNS,
    CLAUDE_MODEL_CODENAMES,
    COUNCIL_NAME_LEAK_RE,
    DEFAULT_MODEL_ALIASES,
    DEFAULT_TIER_BY_TYPE,
    DEFAULT_TIER_TO_MODEL,
    GOOGLE_MODEL_CODENAMES,
    MODELS_CONFIG_PATH,
    MODEL_TIERS,
    MODEL_TIERS_SET,
    PROVIDER_BRAND_TOKENS,
    TIER_ENV_VARS,
    get_blind_judge_leak_patterns,
    get_council_name_leak_re,
    get_model_alias_map,
    get_model_for_tier,
    get_model_tier_map,
    infer_subagent_tier,
    resolve_model_tiers,
    select_target_tier,
    validate_model_config,
)

SCRIPT_DIR = Path(__file__).resolve().parent.parent

ALLOWED_SUBAGENT_MODELS = {DEFAULT_TIER_TO_MODEL["standard"], DEFAULT_TIER_TO_MODEL["lite"]}
STRICT_READONLY_SUBAGENT_TYPES = {"scout", "verifier", "reviewer", "judge"}
HIGH_REASONING_READONLY_TYPES = {"verifier", "reviewer", "judge"}
SCRATCH_ONLY_SUBAGENT_TYPES = {"research", "researcher", "council"}
READONLY_NON_VERIFICATION_TYPES = {
    "scout", "research", "researcher", "council", "verifier", "reviewer", "judge",
}
ALLOWED_VERIFIER_REVIEWER_MODELS = {DEFAULT_TIER_TO_MODEL[t] for t in ALLOWED_HIGH_REASONING_TIERS}
ALLOWED_COUNCIL_PROPOSER_MODELS = {DEFAULT_TIER_TO_MODEL[t] for t in ALLOWED_COUNCIL_PROPOSER_TIERS}
ISOLATED_WORKSPACES = {"branch", "share"}
DEFAULT_MODEL_BY_TYPE = {
    sub_type: DEFAULT_TIER_TO_MODEL[tier] for sub_type, tier in DEFAULT_TIER_BY_TYPE.items()
}

MUTATION_VERBS_RE = re.compile(
    r"\b(edit|modify|write|create|delete|remove|replace|patch|implement|refactor)\b",
    re.I,
)
NUMBERED_RULE_RE = re.compile(r"(?m)^\s*[0-9]+[.)]\s+")
ALLOWED_FILES_HEADER_RE = re.compile(r"(?m)^\s*ALLOWED_FILES:\s*(.+)$", re.I)
REPLAN_IF_HEADER_RE = re.compile(r"(?m)^\s*REPLAN_IF:\s*(.+)$", re.I)
REPLAN_HEADER_RE = re.compile(r"(?m)^\s*REPLAN:\s*(.+)$", re.I)
SPEC_FILE_HEADER_RE = re.compile(r"(?m)^\s*SPEC_FILE:\s*(.+)$", re.I)
STAGE_HEADER_RE = re.compile(r"(?m)^\s*STAGE:\s*(.+)$", re.I)
REVIEW_LENS_HEADER_RE = re.compile(r"(?m)^[ \t]*REVIEW_LENS:[ \t]*([^\r\n]+)$", re.I)
MODE_HEADER_RE = re.compile(r"(?m)^[ \t]*MODE:[ \t]*([^\r\n]+)$", re.I)
COHORT_HEADER_RE = re.compile(r"(?m)^[ \t]*COHORT:[ \t]*([^\r\n]+)$", re.I)
COUNCIL_ID_HEADER_RE = re.compile(r"(?m)^[ \t]*COUNCIL_ID:[ \t]*([^\r\n]+)$", re.I)
COUNCIL_ROLE_HEADER_RE = re.compile(r"(?m)^[ \t]*COUNCIL_ROLE:[ \t]*([^\r\n]+)$", re.I)
EVAL_CMD_HEADER_RE = re.compile(r"(?m)^[ \t]*EVAL_CMD:[ \t]*([^\r\n]+)$", re.I)
TRIVIAL_EVAL_CMD_RE = re.compile(
    r"^\s*(?:[A-Za-z_][A-Za-z0-9_]*\+?=\S+\s+)*(?:(?:(?:/(?:usr/)?bin/)?(?:true|echo|printf|pwd|ls|cat|sleep)|exit)\b|:(?=$|\s))",
    re.I,
)
TRIVIAL_TEST_FLAG_RE = re.compile(
    r"(?:^|\s)(?:--help|--version|(?-i:-[a-zA-Z]*[Vh])|--dry-run|--collect-only|--co|--fixtures|--fixtures-per-test|--cache-show|--markers|--setup-only|--setup-plan|--trace-config|--no-run|--listTests|--list-tests|--list|--showConfig|--show-config|--passWithNoTests|--pass-with-no-tests|--clearCache|--clear-cache)(?=$|\s)",
    re.I,
)
GROUNDED_SPEC_SECTION_PATTERNS = (
    re.compile(r"(?im)(?:^|[^A-Za-z0-9])signatures?(?:$|[^A-Za-z0-9])"),
    re.compile(r"(?im)(?:^|[^A-Za-z0-9])(?:pitfalls?|invariants?)(?:$|[^A-Za-z0-9])"),
    re.compile(r"(?im)(?:^|[^A-Za-z0-9])verification(?:$|[^A-Za-z0-9])"),
)
UUID_RE = re.compile(r"\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b", re.I)
SUBAGENT_SPAWN_RE = re.compile(
    r'(?:\\*["\']?conversationId\\*["\']?\s*[:=]\s*\\*["\']?)([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})',
    re.I,
)

PENDING_SPAWN_TTL_SECONDS = 120
MAX_STATE_INVOCATION_HISTORY = 25
MAX_STATE_LIST_ENTRIES = 100
MAX_DIRECT_MUTATION_FILES = 2
MAX_DIRECT_COMMANDS = 2
MAX_DIRECT_READ_FILES = 8
MAX_DIRECT_READ_LINES = 400
MAX_DIRECT_READ_BYTES = 16384
DEFAULT_UNBOUNDED_READ_LINES = 150
DEFAULT_UNBOUNDED_READ_BYTES = 6000
MAX_REPLAN_ATTEMPTS = 1
MAX_SUBAGENT_REPORT_CHARS = 1600
MAX_REPORT_BULLET_LINES = 10
MAX_CONSECUTIVE_FAILURES = 2

REPORT_BULLET_RE = re.compile(r"^[-*]\s+\S+")
REPORT_LOG_URI_RE = re.compile(r"file://\S+\.log\b", re.I)
REPORT_LINE_PATTERNS = (
    re.compile(r"^STATUS:\s*(?:COMPLETED|CHECKPOINT|NEEDS_REPLAN|BLOCKED|FAILED)\b"),
    re.compile(r"^FILES_CHANGED:\s*\S+"),
    re.compile(r"^VERIFICATION:\s*\S+"),
    re.compile(r"^NOTES:\s*\S+"),
)
SCOUT_COMPACT_LINE_PATTERNS = (
    re.compile(r"^STATUS:\s*(?:COMPLETED|BLOCKED|FAILED)\b"),
    re.compile(r"^NOTES:\s*\S+"),
)

READ_TOOLS = {"read_file", "view_file"}
DIR_LIST_TOOLS = {"list_dir"}
SEARCH_TOOLS = {"grep_search", "find_by_name"}
MUTATION_TOOLS = {"write_file", "edit_file", "write_to_file", "replace_file_content", "multi_replace_file_content"}
PATH_ARG_KEYS = ("Path", "TargetFile", "AbsolutePath", "FilePath", "file_path", "SearchPath", "SearchDirectory", "DirectoryPath")
FORBIDDEN_DIRECT_READ_RE = re.compile(r"(?:^|/)transcript(?:_full|[0-9a-zA-Z_\-]*)?\.jsonl$|\.log$", re.I)

SANDBOX_BOUNDARY_ERROR_RE = re.compile(
    r"(?:There were sandbox errors that may or may not be related to the failure"
    r"|\.git/[A-Za-z0-9._/\-]*['\"]?:\s*Read-only file system"
    r"|Unable to create ['\"]?[^'\"]*\.git/[^'\"]*\.lock['\"]?:\s*Read-only file system"
    r"|cannot open ['\"]?[^'\"]*\.git/[A-Za-z0-9._/\-]*['\"]?:\s*Read-only file system)",
    re.I,
)
TEST_CMD_START_RE = re.compile(
    r"^\s*(?:(?:[A-Za-z_][A-Za-z0-9_]*\+?=\S+\s+)*"
    r"(?:(?:python3?|py|uv\s+run|poetry\s+run|pipenv\s+run|npx|bunx)\s+(?:-m\s+)?)?)?"
    r"(?:pytest|unittest|cargo\s+test|npm\s+test|pnpm\s+test|yarn\s+test|go\s+test|jest|vitest|tox|nox)\b"
)
DOC_EXTENSIONS = {".md", ".txt", ".rst", ".adoc"}
TEST_FILE_RE = re.compile(
    r"(^|/)(tests?|__tests__)/|(^|/)(test_[^/]+\.py|[^/]+_test\.(py|go|rs)|[^/]+\.(test|spec)\.[jt]sx?)$",
    re.I,
)
COMMENT_LINE_RE = re.compile(r"^\s*(?:#|//)")
TEST_DEF_RE = re.compile(r"(?m)^\s*(?:async\s+def|def)\s+test_[A-Za-z0-9_]+\s*\(|^\s*(?:it|test)\s*\(")
ASSERTION_RE = re.compile(r"\b(?:assert\b|self\.assert[A-Za-z0-9_]*\s*\(|expect\s*\(|should\.)")
TAMPER_SKIP_RE = re.compile(
    r"(?m)(?:^\s*(?:@unittest\.(?:skip|expectedFailure)\b|@unittest\.skipIf\s*\(\s*(?:True\b|([0-9]+)\s*==\s*\1(?=$|[\s,)]))|"
    r"@unittest\.skipUnless\s*\(\s*False\b|"
    r"@pytest\.mark\.(?:skip|xfail)\b|@pytest\.mark\.skipif\s*\(\s*(?:True\b|([0-9]+)\s*==\s*\2(?=$|[\s,)]))|#\s*\[ignore\b)"
    r"|\bpytest\.(?:skip|xfail)\s*\(|\bself\.skipTest\s*\(|(?:\braise\s+)?(?:\bunittest\.)?\bSkipTest\b|"
    r"\b(?:it|test|describe)\.(?:skip|only)\s*\(|\b(?:xit|xtest|xdescribe)\s*\()"
)
TRIVIAL_ASSERT_RE = re.compile(
    r"(?m)(?:^\s*|[:;]\s*)(?:assert\s+(?:True\b|\(\s*True\s*\))(?=$|[\s,#;])|"
    r"assert\s+(?:([0-9]+)\s*==\s*\1|\(\s*([0-9]+)\s*==\s*\2\s*\))(?=$|[\s,#;])|"
    r"self\.assertTrue\s*\(\s*True\s*\))"
)
_STRIP_NON_CODE_RE = re.compile(
    r'/\*[\s\S]*?\*/|"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|#(?!\s*\[ignore\b)[^\n]*|^\s*//[^\n]*',
    re.M,
)
_STRIP_NON_CODE_C_STYLE_RE = re.compile(
    r'/\*[\s\S]*?\*/|"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|#(?!\s*\[ignore\b)[^\n]*|//[^\n]*',
    re.M,
)
ALLOWED_SYSTEM_PATH_PREFIXES = ("/dev/", "/usr/", "/bin/", "/lib/", "/lib64/", "/proc/", "/tmp/")
SLASH_COMMAND_RE = re.compile(r"(?:^|[\s`'\"])/philharmonic(?:\b|$)", re.I)


def _find_project_root() -> Path:
    """Resolve the project workspace root from the script location."""
    for parent in [SCRIPT_DIR] + list(SCRIPT_DIR.parents):
        if parent.name == ".agents":
            return parent.parent.resolve()
    return Path.cwd().resolve()


PROJECT_ROOT = _find_project_root()


def _is_sandbox_boundary_error(payload: Dict[str, Any], error_text: str, output_text: str) -> bool:
    """Return True if the error is an environment/sandbox restriction rather than a genuine code/command failure."""
    return (
        payload.get("sandboxError") is True
        or bool(error_text and SANDBOX_BOUNDARY_ERROR_RE.search(error_text))
        or bool(output_text and SANDBOX_BOUNDARY_ERROR_RE.search(output_text))
    )


def _strip_comments(text: str, file_name: str = "", *, preserve_ignore: bool = False) -> str:
    """Remove comments and string literals before counting test definitions, returns, or assertions."""
    if not text:
        return ""

    def _replace_non_code(match: re.Match[str]) -> str:
        tok = match.group(0)
        return '""' if tok.startswith(('"""', "'''", '"')) else ("''" if tok.startswith("'") else "")

    is_non_py = bool(file_name) and Path(file_name).suffix.lower() not in {".py", ".pyi"}
    t = (_STRIP_NON_CODE_C_STYLE_RE if is_non_py else _STRIP_NON_CODE_RE).sub(_replace_non_code, text)
    if preserve_ignore:
        return "\n".join(ln for ln in t.splitlines() if not COMMENT_LINE_RE.match(ln) or re.match(r"^\s*#\s*\[ignore\b", ln))
    return "\n".join(ln for ln in t.splitlines() if not COMMENT_LINE_RE.match(ln))


def _count_early_test_returns(text: str, file_name: str = "") -> int:
    """Count return statements directly inside `def test_*` bodies (ignoring nested defs)."""
    if not text:
        return 0
    count, in_test_def, test_indent, body_indent, nested_indent = 0, False, -1, None, None
    for line in _strip_comments(text, file_name=file_name).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip(" \t"))
        if re.match(r"^(?:async\s+)?def\s+test_[A-Za-z0-9_]+\s*\(", stripped):
            if m_inline := re.search(r"\)\s*(?:->[^:]+)?:\s*(\S.*)$", stripped):
                if re.search(r"(?:^|[;:]\s*)return(?:\b|$)", m_inline.group(1)):
                    count += 1
                in_test_def, test_indent, body_indent, nested_indent = False, -1, None, None
            else:
                in_test_def, test_indent, body_indent, nested_indent = True, indent, None, None
            continue
        if in_test_def:
            if indent <= test_indent and not stripped.startswith(("@", ")")):
                in_test_def, test_indent, body_indent, nested_indent = False, -1, None, None
                continue
            if nested_indent is not None:
                if indent <= nested_indent:
                    nested_indent = None
                else:
                    continue
            if re.match(r"^(?:async\s+)?def\s+", stripped):
                nested_indent = indent
                continue
            if body_indent is None:
                body_indent = indent
            if indent >= body_indent and re.search(r"(?:^|[;:]\s*)return(?:\b|$)", stripped):
                count += 1
    return count


def _extract_call_id(payload: Dict[str, Any]) -> str:
    """Extract a deterministic tool call identifier from toolCall.id or stepIdx."""
    raw_tc = payload.get("toolCall")
    tool_call = raw_tc if isinstance(raw_tc, dict) else {}
    if (tc_id := tool_call.get("id")) is not None and str(tc_id) != "":
        return str(tc_id)
    if (step_idx := payload.get("stepIdx")) is not None and str(step_idx) != "":
        return f"step:{step_idx}"
    return ""
