#!/usr/bin/env python3
"""Deterministic enforcement and workspace permission hook for Philharmonic (pro-orchestrator).

Responsibilities:
1. PreInvocation:
   - Records the root orchestrator conversation ID and current invocation number.
   - Injects a subagent context isolation reminder into worker subagents, deduplicated per
     (conversationId, invocationNum).
2. PreToolUse:
   - Enforces native Antigravity tool schemas (`read_file`, `write_file`, `edit_file`,
     `grep_search`, `find_by_name`, `run_command`, `define_subagent`, `invoke_subagent`,
     `send_message`, `manage_subagents`).
   - Forces `EnableSubagentTools=False` on custom subagent definitions (`define_subagent`).
   - Blocks `Model='pro'`, rewrites `Model='inherit'` to `'flash'` or `'flash_lite'`, and
     auto-downgrades simple read-only tasks (<= 3 rules) to `'flash_lite'` on `invoke_subagent`.
   - Enforces the Direct Execution Fast-Path budget on the root orchestrator:
     <= 2 workspace file reads, <= 2 workspace file mutations, <= 2 non-git commands per turn,
     and Option A single verification test pass after direct edits (blocking iterative
     edit-and-test loops).
   - Blocks root orchestrator from reading or grepping raw `.log` or `transcript*.jsonl` files.
   - Confines `research` subagents to writing markdown/text notes under `scratch/`.
   - Blocks test tampering (commenting out or deleting tests/assertions, adding skip/xfail
     markers, or inserting trivial `assert True`).
3. PostToolUse:
   - Synchronizes spawned subagent metadata and branch lifecycle state in
     `<artifactDirectoryPath>/orchestrator_state.json`.
   - Tracks consecutive failures for the 2-turn circuit breaker and records empirical test receipts.
4. Stop:
   - Allows turn completion when `fullyIdle` is False (supporting reactive wakeup).
   - Blocks premature final completion when uncleaned branched subagents remain, when a worker
     subagent violates the 4-header REPORT FORMAT, when git diff auditing detects test tampering,
     or when an empirical test receipt is missing.
5. CLI Diagnostics (`status`, `reset`, `doctor`):
   - Provides terminal inspection and state recovery commands.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent

ALLOWED_SUBAGENT_MODELS = {"flash", "flash_lite"}
DEFAULT_MODEL_BY_TYPE = {
    "scout": "flash_lite",
    "research": "flash",
    "researcher": "flash",
    "implementer": "flash",
    "self": "flash",
    "image-generator": "flash",
}

MUTATION_VERBS_RE = re.compile(
    r"\b(edit|modify|write|create|delete|remove|replace|patch|implement|refactor)\b",
    re.I,
)
NUMBERED_RULE_RE = re.compile(r"(?m)^\s*[0-9]+[.)]\s+")
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

READ_TOOLS = {"read_file", "list_dir"}
SEARCH_TOOLS = {"grep_search", "find_by_name"}
MUTATION_TOOLS = {"write_file", "edit_file"}

FORBIDDEN_DIRECT_READ_RE = re.compile(
    r"(?:^|/)transcript(?:_full|[0-9a-zA-Z_\-]*)?\.jsonl$|\.log$", re.I
)
RAW_LOG_OR_TRANSCRIPT_RE = FORBIDDEN_DIRECT_READ_RE

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
    """Remove full-line comments before counting test definitions or assertions."""
    if not text:
        return ""
    return "\n".join(
        line for line in text.splitlines() if not COMMENT_LINE_RE.match(line)
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


def _state_file_path(payload: Dict[str, Any]) -> Path:
    artifact_dir = payload.get("artifactDirectoryPath")
    if artifact_dir:
        base = Path(artifact_dir)
    else:
        base = PROJECT_ROOT / ".agents" / ".state"
    base.mkdir(parents=True, exist_ok=True)
    return base / "orchestrator_state.json"


def _current_invocation_key(payload: Dict[str, Any], state: Dict[str, Any]) -> str:
    """Return invocation key from payload (if non-zero) or persisted state from PreInvocation."""
    inv = payload.get("invocationNum")
    if inv is not None and inv != 0 and inv != "0":
        return str(inv)
    return str(state.get("current_invocation_num", 0))


def _default_state() -> Dict[str, Any]:
    return {
        "isolation_reminded_invocations": [],
        "recorded_spawn_calls": [],
        "pending_spawns": [],
        "subagents": {},
        "stop_reminded_executions": [],
        "direct_budget": {},
        "test_receipt": {},
        "circuit_breaker": {},
        "current_invocation_num": 0,
    }


def _load_state(payload: Dict[str, Any]) -> Dict[str, Any]:
    path = _state_file_path(payload)
    if path.exists():
        try:
            with open(path, encoding="utf-8") as fh:
                fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
                try:
                    data = json.load(fh)
                finally:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                if isinstance(data, dict):
                    data["isolation_reminded_invocations"] = (
                        data.get("isolation_reminded_invocations")
                        or data.get("warned_invocations")
                        or []
                    )
                    return data
        except (OSError, json.JSONDecodeError):
            pass
    return _default_state()


def _prune_state(state: Dict[str, Any]) -> None:
    """Bound per-invocation dictionaries so state files cannot grow without limit."""
    bucket = state.get("direct_budget")
    if isinstance(bucket, dict) and len(bucket) > MAX_STATE_INVOCATION_HISTORY:
        sorted_keys = sorted(
            bucket.keys(),
            key=lambda k: int(k) if str(k).isdigit() else -1,
        )
        for old_key in sorted_keys[:-MAX_STATE_INVOCATION_HISTORY]:
            bucket.pop(old_key, None)


def _save_state(payload: Dict[str, Any], state: Dict[str, Any]) -> None:
    path = _state_file_path(payload)
    _prune_state(state)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".orch_state_", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                json.dump(state, fh, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
    except OSError:
        pass


def _sync_subagents_from_transcript(
    payload: Dict[str, Any], state: Dict[str, Any]
) -> bool:
    """Scan transcript.jsonl incrementally to pair pending_spawns with conversationIds."""
    transcript_path = payload.get("transcriptPath")
    if not transcript_path or not os.path.exists(transcript_path):
        return False

    pending: List[Dict[str, Any]] = state.get("pending_spawns", [])
    subagents: Dict[str, Dict[str, Any]] = state.setdefault("subagents", {})
    parent_cid = (payload.get("conversationId") or "").lower()
    changed = False

    try:
        file_size = os.path.getsize(transcript_path)
        offset = int(state.get("transcript_offset", 0))
        if offset < 0 or offset > file_size:
            offset = 0

        with open(transcript_path, "rb") as fh:
            fh.seek(offset)
            chunk = fh.read()
            new_offset = fh.tell()

        if new_offset != offset:
            state["transcript_offset"] = new_offset
            changed = True

        if not chunk:
            return changed

        text = chunk.decode("utf-8", errors="replace")
        for raw_line in text.splitlines():
            if not raw_line.strip():
                continue
            if (
                "docs/hooks.md" in raw_line
                or "docs/" in raw_line
                or "view_file" in raw_line
                or "read_file" in raw_line
                or "File Path: `file://" in raw_line
            ):
                continue
            if not (
                "Created the following subagent" in raw_line
                or "invoke_subagent" in raw_line
            ):
                continue
            matches = list(SUBAGENT_SPAWN_RE.finditer(raw_line))
            if not matches and "Created the following subagent" in raw_line:
                matches = list(UUID_RE.finditer(raw_line))
            for match in matches:
                cid = match.group(1).lower()
                if cid == parent_cid or cid in subagents:
                    continue
                if pending:
                    spawn_meta = pending.pop(0)
                    subagents[cid] = spawn_meta
                    changed = True
    except OSError:
        return changed

    return changed


def _allowed_roots(payload: Dict[str, Any]) -> List[Path]:
    roots: List[Path] = [PROJECT_ROOT]
    for wp in payload.get("workspacePaths") or []:
        try:
            roots.append(Path(wp).resolve())
        except OSError:
            pass
    artifact_dir = payload.get("artifactDirectoryPath")
    if artifact_dir:
        try:
            resolved_artifact = Path(artifact_dir).resolve()
            roots.append(resolved_artifact)
            if resolved_artifact.parent.name == "brain":
                roots.append(resolved_artifact.parent)
                worktrees_dir = resolved_artifact.parent.parent / "worktrees"
                roots.append(worktrees_dir)
        except OSError:
            pass
    return roots


def _is_within_allowed_roots(target: Optional[str], roots: List[Path]) -> bool:
    if not target:
        return False
    try:
        resolved = Path(os.path.expanduser(target)).resolve()
    except OSError:
        return False
    for root in roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _command_stays_in_workspace(
    command_line: str, cwd: str, roots: List[Path]
) -> bool:
    """Return True unless the command targets paths outside allowed roots."""
    base_dir = Path(os.path.expanduser(cwd)).resolve() if cwd else PROJECT_ROOT
    for match in re.finditer(r"(?:^|[\s=\"'])(/[A-Za-z0-9._/\-]+)", command_line):
        abs_path = match.group(1)
        if abs_path == "/tmp" or abs_path.startswith(ALLOWED_SYSTEM_PATH_PREFIXES):
            continue
        if not _is_within_allowed_roots(abs_path, roots):
            return False
    for match in re.finditer(r"(?:^|[\s=\"'])(~[A-Za-z0-9._/\-]*)", command_line):
        tilde_path = os.path.expanduser(match.group(1))
        if not _is_within_allowed_roots(tilde_path, roots):
            return False
    for match in re.finditer(
        r"(?:^|[\s=\"'])([A-Za-z0-9._\-/]*\.\.[A-Za-z0-9._\-/]*)", command_line
    ):
        rel_token = match.group(1)
        candidate = (base_dir / rel_token).resolve()
        if not _is_within_allowed_roots(str(candidate), roots):
            return False
    return True


def _is_parent_orchestrator(cid: str, state: Dict[str, Any]) -> bool:
    """Return True if cid is recorded as the root/parent orchestrator."""
    if not cid or cid in state.get("subagents", {}):
        return False
    if state.get("root_conversation_id") == cid:
        return True
    prefix = f'["{cid}"'
    return any(
        call.startswith(prefix) for call in state.get("recorded_spawn_calls", [])
    )


def _normalize_subagent_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(meta)
    if "type" in normalized and "type_name" not in normalized:
        normalized["type_name"] = normalized["type"]
    elif "type_name" in normalized and "type" not in normalized:
        normalized["type"] = normalized["type_name"]
    return normalized


def _is_known_subagent(payload: Dict[str, Any], state: Dict[str, Any]) -> bool:
    """Return True if this invocation belongs to a spawned worker subagent."""
    if state.get("self_subagent_meta"):
        return True
    if payload.get("isSubagent") is True:
        return True
    cid = (payload.get("conversationId") or "").strip().lower()
    if cid and cid in state.get("subagents", {}):
        return True
    if _is_parent_orchestrator(cid, state):
        return False
    for wp in payload.get("workspacePaths") or []:
        if "/worktrees/" in str(wp).replace("\\", "/"):
            return True
    artifact_dir = payload.get("artifactDirectoryPath")
    if artifact_dir:
        try:
            resolved_art = Path(artifact_dir).resolve()
            if resolved_art.parent.name == "brain":
                for sibling_state_path in resolved_art.parent.glob(
                    "*/orchestrator_state.json"
                ):
                    if sibling_state_path.parent == resolved_art:
                        continue
                    try:
                        sib_data = json.loads(
                            sibling_state_path.read_text(encoding="utf-8")
                        )
                    except (OSError, json.JSONDecodeError):
                        continue
                    sibling_subagents = sib_data.get("subagents") or {}
                    if cid and cid in sibling_subagents:
                        state["self_subagent_meta"] = _normalize_subagent_meta(
                            sibling_subagents[cid]
                        )
                        _save_state(payload, state)
                        return True
                    pending_list = sib_data.get("pending_spawns") or []
                    if pending_list and cid != sib_data.get("root_conversation_id"):
                        try:
                            age = time.time() - sibling_state_path.stat().st_mtime
                            if age < PENDING_SPAWN_TTL_SECONDS:
                                state["self_subagent_meta"] = _normalize_subagent_meta(
                                    pending_list[0]
                                )
                                _save_state(payload, state)
                                return True
                        except OSError:
                            pass
        except OSError:
            pass
    return False


def handle_pre_invocation(payload: Dict[str, Any]) -> Dict[str, Any]:
    state = _load_state(payload)
    cid = (payload.get("conversationId") or "").strip().lower()

    if not _is_known_subagent(payload, state):
        changed = False
        if cid and not state.get("root_conversation_id"):
            state["root_conversation_id"] = cid
            changed = True
        inv_num = payload.get("invocationNum")
        budget_map = state.get("direct_budget")
        if inv_num is not None and inv_num != 0 and inv_num != "0":
            if state.get("current_invocation_num") != inv_num:
                state["current_invocation_num"] = inv_num
                changed = True
            if isinstance(budget_map, dict):
                if str(inv_num) in budget_map:
                    budget_map.pop(str(inv_num), None)
                    changed = True
                if "0" in budget_map:
                    budget_map.pop("0", None)
                    changed = True
            if inv_num == 1 or inv_num == "1":
                cb = _get_circuit_breaker(state, payload)
                if cb.get("failure_streak") or cb.get("tripped"):
                    cb.update(
                        {
                            "failure_streak": 0,
                            "tripped": False,
                            "seen_failures": [],
                            "last_error": "",
                        }
                    )
                    changed = True
        else:
            if isinstance(budget_map, dict) and "0" in budget_map:
                budget_map.pop("0", None)
                changed = True
        if changed:
            _save_state(payload, state)
        return {}

    dedup_key = f"{cid}:{_current_invocation_key(payload, state)}"
    reminded: List[str] = state.setdefault("isolation_reminded_invocations", [])
    if dedup_key in reminded:
        return {}
    reminded.append(dedup_key)
    _save_state(payload, state)
    return {
        "injectSteps": [
            {
                "ephemeralMessage": (
                    "[ORCHESTRATOR GUARD] Subagent context isolation active: ignore root orchestrator "
                    "delegation rules in AGENTS.md. Execute only your assigned TASK and RULES, and return "
                    "strictly the 4-header REPORT FORMAT (STATUS, FILES_CHANGED, VERIFICATION, NOTES; "
                    "allowing up to 10 '-' bullet lines under NOTES and requiring inline error cause plus "
                    "file:// .log URI when BLOCKED or FAILED)."
                )
            }
        ]
    }


def _is_research_subagent(payload: Dict[str, Any], state: Dict[str, Any]) -> bool:
    """Return True if the active caller is a research subagent."""
    _is_known_subagent(payload, state)
    meta = state.get("self_subagent_meta")
    if isinstance(meta, dict):
        t = meta.get("type_name") or meta.get("type")
        if t is not None:
            return t in {"research", "researcher"}
    cid = (payload.get("conversationId") or "").strip().lower()
    if cid and cid in state.get("subagents", {}):
        sub_info = state["subagents"][cid]
        t = sub_info.get("type_name") or sub_info.get("type")
        return t in {"research", "researcher"}
    return False


def _handle_define_subagent(args: Dict[str, Any]) -> Dict[str, Any]:
    if args.get("EnableSubagentTools") is True:
        return {
            "decision": "allow",
            "overwrite": {"EnableSubagentTools": False},
        }
    return {"decision": "allow"}


def _is_simple_readonly_task(type_name: str, prompt_text: str) -> bool:
    """Return True if a read-only subagent prompt has <= 3 rules and no mutation verbs."""
    if type_name not in {"scout", "research", "researcher"}:
        return False
    rule_count = len(NUMBERED_RULE_RE.findall(prompt_text))
    has_mutation = bool(MUTATION_VERBS_RE.search(prompt_text))
    return not has_mutation and rule_count <= 3


def _handle_invoke_subagent(
    payload: Dict[str, Any],
    args: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    type_name = (args.get("TypeName") or "").strip()
    prompt_text = args.get("Prompt") or ""
    if not type_name:
        return {
            "decision": "deny",
            "reason": "invoke_subagent requires a non-empty 'TypeName'.",
        }
    if not prompt_text or not prompt_text.strip():
        return {
            "decision": "deny",
            "reason": "invoke_subagent requires a non-empty 'Prompt'.",
        }

    requested_model = (args.get("Model") or "inherit").strip()
    role_label = (args.get("Role") or type_name or "subagent").strip()

    if requested_model == "pro":
        return {
            "decision": "deny",
            "reason": (
                f"Subagent '{role_label}' requested Model='pro'. "
                "Subagents must use 'flash' or 'flash_lite'."
            ),
        }

    target_model = requested_model
    if target_model not in ALLOWED_SUBAGENT_MODELS:
        target_model = DEFAULT_MODEL_BY_TYPE.get(type_name, "flash")

    if target_model == "flash" and _is_simple_readonly_task(type_name, prompt_text):
        target_model = "flash_lite"

    workspace_mode = (args.get("Workspace") or "inherit").strip()
    spawn_record = {
        "role": role_label,
        "type": type_name,
        "type_name": type_name,
        "model": target_model,
        "workspace": workspace_mode,
        "active_branch": workspace_mode == "branch",
    }

    if state is None:
        state = _load_state(payload)
    call_id = _extract_call_id(payload)
    call_sig = json.dumps(
        [
            payload.get("conversationId", ""),
            _current_invocation_key(payload, state),
            payload.get("executionNum", 0),
            call_id,
            type_name,
            role_label,
            workspace_mode,
        ],
        sort_keys=True,
    )
    recorded_calls: List[str] = state.setdefault("recorded_spawn_calls", [])
    if call_sig not in recorded_calls:
        recorded_calls.append(call_sig)
        state.setdefault("pending_spawns", []).append(spawn_record)
        _save_state(payload, state)

    if args.get("Model") != target_model:
        return {
            "decision": "allow",
            "overwrite": {"Model": target_model},
        }
    return {"decision": "allow"}


def _handle_send_message(
    payload: Dict[str, Any],
    args: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    recipient = (args.get("Recipient") or "").strip().lower()
    message = args.get("Message") or ""

    if not message or not message.strip():
        return {"decision": "deny", "reason": "send_message.Message cannot be empty."}

    if state is None:
        state = _load_state(payload)
    _sync_subagents_from_transcript(payload, state)
    _save_state(payload, state)

    subagent_info = state.get("subagents", {}).get(recipient)
    if subagent_info and subagent_info.get("model") == "flash_lite":
        rule_count = len(NUMBERED_RULE_RE.findall(message))
        has_mutation = bool(MUTATION_VERBS_RE.search(message))
        if has_mutation or rule_count > 3:
            return {
                "decision": "deny",
                "reason": (
                    f"Recipient '{recipient}' is a 'flash_lite' ({subagent_info.get('type', 'scout')}) "
                    "subagent. Do not send code mutation or multi-step (>3 rules) tasks to 'flash_lite'. "
                    "Invoke a 'flash' subagent ('research' or 'implementer') instead."
                ),
            }

    return {"decision": "allow"}


def _handle_manage_subagents(
    payload: Dict[str, Any],
    args: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    action = args.get("Action", "")
    if state is None:
        state = _load_state(payload)
    _sync_subagents_from_transcript(payload, state)
    subagents = state.get("subagents", {})

    if action == "kill_all":
        for meta in subagents.values():
            meta["active_branch"] = False
        for pending in state.get("pending_spawns", []):
            pending["active_branch"] = False
        state["pending_spawns"] = []
        _save_state(payload, state)
    elif action == "kill":
        target_ids = {
            cid.strip().lower() for cid in (args.get("ConversationIds") or [])
        }
        for cid in target_ids:
            if cid in subagents:
                subagents[cid]["active_branch"] = False
        remaining_pending = []
        killed_left = len(target_ids)
        for pending in state.get("pending_spawns", []):
            if pending.get("active_branch") and killed_left > 0:
                pending["active_branch"] = False
                killed_left -= 1
            remaining_pending.append(pending)
        state["pending_spawns"] = remaining_pending
        _save_state(payload, state)

    return {"decision": "allow"}


def _is_root_orchestrator(
    payload: Dict[str, Any], state: Optional[Dict[str, Any]] = None
) -> bool:
    """Return True when the caller is the root orchestrator."""
    if state is None:
        state = _load_state(payload)
    cid = (payload.get("conversationId") or "").strip().lower()
    if cid and cid in state.get("subagents", {}):
        return False
    if _is_known_subagent(payload, state):
        return False
    return True


_is_orchestrator_caller = _is_root_orchestrator


def _is_artifact_or_worktree_path(target: str, payload: Dict[str, Any]) -> bool:
    """Return True if target is inside the conversation artifact/brain directory or a subagent worktree."""
    artifact_dir = payload.get("artifactDirectoryPath")
    if not artifact_dir or not target:
        return False
    try:
        resolved_target = Path(os.path.expanduser(target)).resolve()
        resolved_artifact = Path(artifact_dir).resolve()
    except OSError:
        return False

    exempt_roots = [resolved_artifact]
    if resolved_artifact.parent.name == "brain":
        exempt_roots.append(resolved_artifact.parent)
        exempt_roots.append(resolved_artifact.parent.parent / "worktrees")

    for root in exempt_roots:
        try:
            resolved_target.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _is_allowed_research_scratch_write(target: str, payload: Dict[str, Any]) -> bool:
    """Return True if target is a .md or .txt file inside a scratch directory under brain/."""
    if not target or not _is_artifact_or_worktree_path(target, payload):
        return False
    try:
        resolved = Path(os.path.expanduser(target)).resolve()
    except OSError:
        return False
    return resolved.suffix.lower() in {".md", ".txt"} and "scratch" in resolved.parts


def _extract_target_path(tool_name: str, args: Dict[str, Any]) -> str:
    """Extract the target file or directory path for native Antigravity tools."""
    if tool_name in SEARCH_TOOLS:
        return str(
            args.get("SearchPath")
            or args.get("SearchDirectory")
            or args.get("DirectoryPath")
            or args.get("Path")
            or ""
        )
    if tool_name in READ_TOOLS:
        return str(args.get("Path") or args.get("DirectoryPath") or "")
    if tool_name in MUTATION_TOOLS:
        return str(args.get("Path") or "")
    return str(
        args.get("Path")
        or args.get("SearchPath")
        or args.get("SearchDirectory")
        or args.get("DirectoryPath")
        or ""
    )


def _check_and_record_direct_budget(
    payload: Dict[str, Any],
    tool_name: str,
    args: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Enforce direct execution fast-path limits on the root orchestrator."""
    if state is None:
        state = _load_state(payload)
    if not _is_root_orchestrator(payload, state):
        return None

    inv_key = _current_invocation_key(payload, state)
    budget = state.setdefault("direct_budget", {}).setdefault(
        inv_key,
        {
            "mutated_files": [],
            "read_files": [],
            "commands": [],
            "seen_calls": [],
            "test_runs_after_edit": 0,
        },
    )
    call_id = _extract_call_id(payload)

    if tool_name in SEARCH_TOOLS:
        search_target = _extract_target_path(tool_name, args)
        if search_target:
            norm_slashes = search_target.replace("\\", "/")
            if FORBIDDEN_DIRECT_READ_RE.search(norm_slashes):
                return {
                    "decision": "deny",
                    "reason": (
                        "[ORCHESTRATOR GUARD] Context separation violation: root orchestrator cannot "
                        f"directly read raw log or transcript file '{Path(search_target).name}'. "
                        "Delegate log slicing to a 'scout' subagent (Model='flash_lite')."
                    ),
                }
        return None

    if tool_name in READ_TOOLS:
        target = _extract_target_path(tool_name, args)
        if not target:
            return None
        norm_slashes = str(target).replace("\\", "/")
        if FORBIDDEN_DIRECT_READ_RE.search(norm_slashes):
            return {
                "decision": "deny",
                "reason": (
                    "[ORCHESTRATOR GUARD] Context separation violation: root orchestrator cannot "
                    f"directly read raw log or transcript file '{Path(target).name}'. "
                    "Delegate log slicing to a 'scout' subagent (Model='flash_lite')."
                ),
            }
        if _is_artifact_or_worktree_path(target, payload) or "/skills/" in norm_slashes:
            return None
        try:
            norm_target = str(Path(os.path.expanduser(target)).resolve())
        except OSError:
            return None

        read_files: List[str] = budget.setdefault("read_files", [])
        if norm_target not in read_files:
            if len(read_files) >= MAX_DIRECT_READ_FILES:
                return {
                    "decision": "deny",
                    "reason": (
                        f"[ORCHESTRATOR GUARD] Direct execution fast-path read limit "
                        f"({MAX_DIRECT_READ_FILES} workspace files) exceeded when reading "
                        f"'{Path(norm_target).name}'. Delegate multi-file reads or discovery to a "
                        "'scout' (Model='flash_lite') or 'research' (Model='flash') subagent."
                    ),
                }
            read_files.append(norm_target)
            _save_state(payload, state)
        return None

    if tool_name in MUTATION_TOOLS:
        target = _extract_target_path(tool_name, args)
        if not target or _is_artifact_or_worktree_path(target, payload):
            return None
        try:
            norm_target = str(Path(os.path.expanduser(target)).resolve())
        except OSError:
            return None

        if budget.get("test_runs_after_edit", 0) >= 1:
            return {
                "decision": "deny",
                "reason": (
                    "[ORCHESTRATOR GUARD] Iterative edit-and-test loop detected in the orchestrator. "
                    "Delegate multi-step code and test execution loops to an 'implementer' subagent (Model='flash')."
                ),
            }

        mutated_files: List[str] = budget.setdefault("mutated_files", [])
        if norm_target not in mutated_files:
            if len(mutated_files) >= MAX_DIRECT_MUTATION_FILES:
                return {
                    "decision": "deny",
                    "reason": (
                        f"[ORCHESTRATOR GUARD] Direct execution fast-path limit "
                        f"({MAX_DIRECT_MUTATION_FILES} workspace files) exceeded when editing "
                        f"'{Path(norm_target).name}'. Delegate multi-file changes to an "
                        "'implementer' subagent (Model='flash')."
                    ),
                }
            mutated_files.append(norm_target)
            _save_state(payload, state)
        return None

    if tool_name == "run_command":
        cmd = (args.get("CommandLine") or "").strip()
        if _is_git_only_command(cmd):
            return None

        seen_calls: List[str] = budget.setdefault("seen_calls", [])
        if call_id and call_id in seen_calls:
            return None

        mutated_files = budget.get("mutated_files", [])
        is_test = _is_test_command(cmd)
        if mutated_files and is_test:
            if budget.get("test_runs_after_edit", 0) >= 1:
                return {
                    "decision": "deny",
                    "reason": (
                        "[ORCHESTRATOR GUARD] Direct workspace edit followed by iterative test execution "
                        "detected in the orchestrator. Delegate code and test execution loops "
                        "to an 'implementer' subagent (Model='flash')."
                    ),
                }
            budget["test_runs_after_edit"] = budget.get("test_runs_after_edit", 0) + 1

        commands: List[str] = budget.setdefault("commands", [])
        if len(commands) >= MAX_DIRECT_COMMANDS:
            return {
                "decision": "deny",
                "reason": (
                    f"[ORCHESTRATOR GUARD] Direct execution fast-path command limit "
                    f"({MAX_DIRECT_COMMANDS} commands per turn) exceeded. Delegate iterative "
                    "command or test execution to a 'scout', 'research', or 'implementer' subagent."
                ),
            }

        if call_id:
            seen_calls.append(call_id)
        commands.append(cmd)
        _save_state(payload, state)
        return None

    return None


def _is_test_file(path_str: str) -> bool:
    if not path_str:
        return False
    norm = path_str.replace("\\", "/")
    return bool(TEST_FILE_RE.search(norm))


def _is_code_file(path_str: str) -> bool:
    if not path_str:
        return False
    ext = Path(path_str).suffix.lower()
    return ext not in DOC_EXTENSIONS


def _check_test_tampering_in_edit(
    tool_name: str, args: Dict[str, Any], payload: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Block edits to test files that delete/comment-out tests/assertions or add skip markers."""
    target = _extract_target_path(tool_name, args)
    if not target or _is_artifact_or_worktree_path(target, payload):
        return None
    if not _is_test_file(target):
        return None

    chunks: List[Tuple[str, str]] = []
    if tool_name == "edit_file":
        chunks.append((args.get("Target") or "", args.get("Replacement") or ""))
    elif tool_name == "write_file":
        new_code = args.get("Content") or ""
        old_code = ""
        try:
            resolved = Path(os.path.expanduser(target)).resolve()
            if resolved.exists():
                old_code = resolved.read_text(encoding="utf-8")
        except OSError:
            old_code = ""
        if args.get("Append") is True and old_code:
            chunks.append((old_code, old_code + "\n" + new_code))
        else:
            chunks.append((old_code, new_code))

    file_name = Path(target).name
    for old_text, new_text in chunks:
        old_skips = len(TAMPER_SKIP_RE.findall(old_text))
        new_skips = len(TAMPER_SKIP_RE.findall(new_text))
        if new_skips > old_skips:
            return {
                "decision": "deny",
                "reason": (
                    f"[ORCHESTRATOR GUARD] Test tampering blocked in '{file_name}': "
                    "adding skip/xfail decorators or disabling tests is prohibited."
                ),
            }

        old_trivial = len(TRIVIAL_ASSERT_RE.findall(old_text))
        new_trivial = len(TRIVIAL_ASSERT_RE.findall(new_text))
        if new_trivial > old_trivial:
            return {
                "decision": "deny",
                "reason": (
                    f"[ORCHESTRATOR GUARD] Test tampering blocked in '{file_name}': "
                    "replacing checks with trivial 'assert True' is prohibited."
                ),
            }

        if old_text:
            old_clean = _strip_comments(old_text)
            new_clean = _strip_comments(new_text)
            old_tests = len(TEST_DEF_RE.findall(old_clean))
            new_tests = len(TEST_DEF_RE.findall(new_clean))
            if new_tests < old_tests:
                return {
                    "decision": "deny",
                    "reason": (
                        f"[ORCHESTRATOR GUARD] Test tampering blocked in '{file_name}': "
                        f"edit reduces test count from {old_tests} to {new_tests}."
                    ),
                }

            old_asserts = len(ASSERTION_RE.findall(old_clean))
            new_asserts = len(ASSERTION_RE.findall(new_clean))
            if new_asserts < old_asserts:
                return {
                    "decision": "deny",
                    "reason": (
                        f"[ORCHESTRATOR GUARD] Test tampering blocked in '{file_name}': "
                        f"edit reduces assertion count from {old_asserts} to {new_asserts}."
                    ),
                }

    return None


def audit_diff_text(diff_text: str) -> Optional[str]:
    """Audit unified git diff text for test tampering across test files."""
    if not diff_text:
        return None

    current_file: Optional[str] = None
    removed_lines: List[str] = []
    added_lines: List[str] = []

    def _check_accumulated(
        fname: Optional[str], rem: List[str], add: List[str]
    ) -> Optional[str]:
        if not fname or not _is_test_file(fname):
            return None
        rem_blob = "\n".join(rem)
        add_blob = "\n".join(add)
        short_name = Path(fname).name

        if len(TAMPER_SKIP_RE.findall(add_blob)) > len(
            TAMPER_SKIP_RE.findall(rem_blob)
        ):
            return f"[ORCHESTRATOR GUARD] Diff audit detected test skip/xfail marker added in '{short_name}'."
        if len(TRIVIAL_ASSERT_RE.findall(add_blob)) > len(
            TRIVIAL_ASSERT_RE.findall(rem_blob)
        ):
            return f"[ORCHESTRATOR GUARD] Diff audit detected trivial 'assert True' added in '{short_name}'."
        rem_clean = _strip_comments(rem_blob)
        add_clean = _strip_comments(add_blob)
        rem_tests = len(TEST_DEF_RE.findall(rem_clean))
        add_tests = len(TEST_DEF_RE.findall(add_clean))
        if rem_tests > add_tests:
            return (
                f"[ORCHESTRATOR GUARD] Diff audit detected net removal of {rem_tests - add_tests} "
                f"test definition(s) in '{short_name}'."
            )
        rem_asserts = len(ASSERTION_RE.findall(rem_clean))
        add_asserts = len(ASSERTION_RE.findall(add_clean))
        if rem_asserts > add_asserts:
            return (
                f"[ORCHESTRATOR GUARD] Diff audit detected net removal of {rem_asserts - add_asserts} "
                f"assertion(s) in '{short_name}'."
            )
        return None

    for raw_line in diff_text.splitlines():
        if raw_line.startswith("+++ "):
            verdict = _check_accumulated(current_file, removed_lines, added_lines)
            if verdict:
                return verdict
            if raw_line.startswith("+++ b/"):
                current_file = raw_line[6:].strip()
            else:
                current_file = None
            removed_lines = []
            added_lines = []
        elif (
            raw_line.startswith("---")
            or raw_line.startswith("@@")
            or raw_line.startswith("diff --git")
        ):
            continue
        elif raw_line.startswith("-"):
            removed_lines.append(raw_line[1:])
        elif raw_line.startswith("+"):
            added_lines.append(raw_line[1:])

    return _check_accumulated(current_file, removed_lines, added_lines)


def _get_circuit_breaker(
    state: Dict[str, Any], payload: Dict[str, Any]
) -> Dict[str, Any]:
    cb_root = state.setdefault("circuit_breaker", {})
    if "failure_streak" not in cb_root:
        cb_root.clear()
        cb_root.update(
            {
                "failure_streak": 0,
                "tripped": False,
                "seen_failures": [],
                "last_error": "",
            }
        )
    return cb_root


def _get_test_receipt(
    state: Dict[str, Any], payload: Dict[str, Any]
) -> Dict[str, Any]:
    receipt_root = state.setdefault("test_receipt", {})
    if "dirty" not in receipt_root:
        receipt_root.clear()
        receipt_root.update(
            {
                "dirty": False,
                "verified": False,
                "last_mutated_file": "",
                "last_test_cmd": "",
            }
        )
    return receipt_root


def handle_pre_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    tool_call = payload.get("toolCall", {})
    tool_name = tool_call.get("name", "")
    args = tool_call.get("args", {}) or {}

    state = _load_state(payload)
    cb = _get_circuit_breaker(state, payload)
    if cb.get("tripped"):
        if tool_name in MUTATION_TOOLS:
            target = _extract_target_path(tool_name, args)
            if not _is_artifact_or_worktree_path(target, payload):
                return {
                    "decision": "deny",
                    "reason": (
                        f"[ORCHESTRATOR GUARD] 2-turn circuit breaker tripped after "
                        f"{cb.get('failure_streak', 2)} consecutive failures "
                        f"(last error: {cb.get('last_error', 'command/subagent failed')}). "
                        "Stop retrying, write failure details to the scratch log if needed, "
                        "and report STATUS: BLOCKED."
                    ),
                }
        elif tool_name == "run_command":
            cmd = (args.get("CommandLine") or "").strip()
            if not re.match(r"^\s*git\s+(status|diff|log)\b", cmd):
                return {
                    "decision": "deny",
                    "reason": (
                        f"[ORCHESTRATOR GUARD] 2-turn circuit breaker tripped after "
                        f"{cb.get('failure_streak', 2)} consecutive failures "
                        f"(last error: {cb.get('last_error', 'command/subagent failed')}). "
                        "Stop retrying, write failure details to the scratch log if needed, "
                        "and report STATUS: BLOCKED."
                    ),
                }
        elif tool_name in {"invoke_subagent", "send_message"}:
            return {
                "decision": "deny",
                "reason": (
                    f"[ORCHESTRATOR GUARD] 2-turn circuit breaker tripped after "
                    f"{cb.get('failure_streak', 2)} consecutive failures "
                    f"(last error: {cb.get('last_error', 'command/subagent failed')}). "
                    "Stop retrying, write failure details to the scratch log if needed, "
                    "and report STATUS: BLOCKED."
                ),
            }

    if tool_name in SEARCH_TOOLS:
        budget_denial = _check_and_record_direct_budget(payload, tool_name, args, state)
        if budget_denial:
            return budget_denial

    if tool_name == "define_subagent":
        return _handle_define_subagent(args)

    if tool_name == "invoke_subagent":
        return _handle_invoke_subagent(payload, args, state)

    if tool_name == "send_message":
        return _handle_send_message(payload, args, state)

    if tool_name == "manage_subagents":
        return _handle_manage_subagents(payload, args, state)

    if tool_name in MUTATION_TOOLS:
        if _is_research_subagent(payload, state):
            target = _extract_target_path(tool_name, args)
            if not _is_allowed_research_scratch_write(target, payload):
                return {
                    "decision": "deny",
                    "reason": (
                        "[ORCHESTRATOR GUARD] Research subagents may only write markdown reports "
                        "to <artifactDirectoryPath>/scratch/*.md. Workspace mutations are prohibited."
                    ),
                }

        tamper_denial = _check_test_tampering_in_edit(tool_name, args, payload)
        if tamper_denial is not None:
            return tamper_denial

    roots = _allowed_roots(payload)

    if tool_name == "run_command":
        cwd = args.get("Cwd", "") or str(PROJECT_ROOT)
        cmd = args.get("CommandLine", "")
        if _is_within_allowed_roots(cwd, roots) and _command_stays_in_workspace(
            cmd, cwd, roots
        ):
            budget_denial = _check_and_record_direct_budget(
                payload, tool_name, args, state
            )
            if budget_denial:
                return budget_denial
            overrides = [f"command({cmd})", "command(*)"]
            for r in roots:
                overrides.append(f"file({r}/**)")
            return {
                "decision": "allow",
                "permissionOverrides": overrides,
            }
        return {"decision": "ask"}

    for path_key in (
        "Path",
        "SearchPath",
        "SearchDirectory",
        "DirectoryPath",
    ):
        if path_key in args:
            target_path = args.get(path_key)
            if _is_within_allowed_roots(target_path, roots):
                budget_denial = _check_and_record_direct_budget(
                    payload, tool_name, args, state
                )
                if budget_denial:
                    return budget_denial
                if tool_name in MUTATION_TOOLS:
                    target = target_path or ""
                    if not _is_artifact_or_worktree_path(
                        target, payload
                    ) and _is_code_file(target):
                        receipt = _get_test_receipt(state, payload)
                        receipt["dirty"] = True
                        receipt["verified"] = False
                        try:
                            receipt["last_mutated_file"] = str(
                                Path(os.path.expanduser(target)).resolve()
                            )
                        except OSError:
                            receipt["last_mutated_file"] = target
                        _save_state(payload, state)
                return {"decision": "allow"}
            return {"decision": "ask"}

    return {"decision": "allow"}


def handle_post_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    state = _load_state(payload)
    tool_call = payload.get("toolCall", {})
    tool_name = tool_call.get("name", "")
    if tool_name == "invoke_subagent" and payload.get("error"):
        pending = state.get("pending_spawns", [])
        if pending:
            pending.pop()
    else:
        _sync_subagents_from_transcript(payload, state)

    cb = _get_circuit_breaker(state, payload)
    receipt = _get_test_receipt(state, payload)
    error_text = str(payload.get("error") or "").strip()
    call_id = _extract_call_id(payload)

    if tool_name in {"run_command", "invoke_subagent"}:
        cmd = ((tool_call.get("args") or {}).get("CommandLine") or "").strip()
        is_test_cmd = tool_name == "run_command" and _is_test_command(cmd)
        if error_text:
            seen_failures: List[str] = cb.setdefault("seen_failures", [])
            dedup_id = f"{tool_name}:{call_id}:{error_text}"
            if not call_id or dedup_id not in seen_failures:
                if call_id:
                    seen_failures.append(dedup_id)
                cb["failure_streak"] = cb.get("failure_streak", 0) + 1
                cb["last_error"] = error_text
                if cb["failure_streak"] >= MAX_CONSECUTIVE_FAILURES:
                    cb["tripped"] = True
            if is_test_cmd:
                receipt["verified"] = False
                receipt["last_test_cmd"] = cmd
        else:
            if (
                tool_name == "invoke_subagent"
                or is_test_cmd
                or (tool_name == "run_command" and not _is_git_only_command(cmd))
            ):
                cb["failure_streak"] = 0
                cb["tripped"] = False
            invoked_type = (
                ((tool_call.get("args") or {}).get("TypeName") or "").strip().lower()
            )
            is_verification_subagent = (
                tool_name == "invoke_subagent"
                and invoked_type not in {"scout", "research", "researcher"}
            )
            if is_test_cmd or (is_verification_subagent and receipt.get("dirty")):
                receipt["dirty"] = False
                receipt["verified"] = True
                receipt["last_test_cmd"] = cmd or "invoke_subagent"

    _save_state(payload, state)
    return {}


def validate_subagent_report(text: str) -> Optional[str]:
    """Validate that a subagent's final output matches the 4-header REPORT FORMAT (Option A + B)."""
    stripped = (text or "").strip()
    if not stripped:
        return "Subagent response is empty."
    if len(stripped) > MAX_SUBAGENT_REPORT_CHARS:
        return (
            f"Subagent report length ({len(stripped)} chars) exceeds maximum "
            f"{MAX_SUBAGENT_REPORT_CHARS} chars."
        )
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    max_lines = 4 + MAX_REPORT_BULLET_LINES
    if len(lines) < 4 or len(lines) > max_lines:
        return (
            f"Expected 4 header lines plus at most {MAX_REPORT_BULLET_LINES} bullet lines "
            f"(4-{max_lines} lines total), got {len(lines)}."
        )
    expected_prefixes = ("STATUS:", "FILES_CHANGED:", "VERIFICATION:", "NOTES:")
    for idx, (pattern, prefix) in enumerate(
        zip(REPORT_LINE_PATTERNS, expected_prefixes), start=1
    ):
        if not pattern.match(lines[idx - 1]):
            return f"Line {idx} must match '{prefix} <value>'."

    bullet_lines = lines[4:]
    for b_idx, b_line in enumerate(bullet_lines, start=5):
        if not REPORT_BULLET_RE.match(b_line):
            return f"Line {b_idx} after NOTES must be a compact bullet starting with '- ' or '* '."

    status_val = lines[0].split(":", 1)[1].strip().upper().split()[0]
    if status_val in {"BLOCKED", "FAILED"}:
        notes_body = lines[3].split(":", 1)[1].strip()
        combined_notes = " ".join(
            [notes_body] + [b[1:].strip() for b in bullet_lines]
        ).strip()
        if not REPORT_LOG_URI_RE.search(combined_notes):
            return f"When STATUS is {status_val}, NOTES must include a 'file://.../*.log' scratch log URI."
        non_uri_text = REPORT_LOG_URI_RE.sub("", combined_notes).strip(" -—:,.;()")
        if len(non_uri_text.split()) < 3:
            return (
                f"When STATUS is {status_val}, NOTES must include an inline error summary "
                "(at least 3 words) in addition to the file:// scratch log URI."
            )

    return None


def _extract_last_model_response(payload_or_path: Any) -> Optional[str]:
    """Extract the final model response text from payload or transcriptPath if available."""
    if isinstance(payload_or_path, dict):
        if (
            "lastAssistantMessage" in payload_or_path
            and payload_or_path["lastAssistantMessage"] is not None
        ):
            return str(payload_or_path["lastAssistantMessage"])
        transcript_path = payload_or_path.get("transcriptPath")
    else:
        transcript_path = str(payload_or_path)

    if not transcript_path or not os.path.exists(transcript_path):
        return None

    try:
        file_size = os.path.getsize(transcript_path)
        tail_size = 65536
        offset = max(0, file_size - tail_size)
        with open(transcript_path, "rb") as fh:
            fh.seek(offset)
            tail_bytes = fh.read()
        tail_text = tail_bytes.decode("utf-8", errors="replace")
        lines = tail_text.splitlines()
        if offset > 0 and len(lines) > 1:
            lines = lines[1:]
        for raw_line in reversed(lines):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                step = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if step.get("type") == "PLANNER_RESPONSE" and not step.get("tool_calls"):
                content = step.get("content")
                if isinstance(content, str) and content.strip():
                    return content
    except OSError:
        return None
    return None


def _get_workspace_git_diff(payload: Dict[str, Any]) -> str:
    """Return unified git diff from payload or by running git diff HEAD in workspace."""
    if payload.get("gitDiff"):
        return str(payload["gitDiff"])
    for diff_cmd in (["git", "diff", "HEAD", "-U0"], ["git", "diff", "-U0"]):
        try:
            proc = subprocess.run(
                diff_cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if proc.returncode == 0:
                return proc.stdout or ""
        except (OSError, subprocess.SubprocessError):
            continue
    return ""


def handle_stop(payload: Dict[str, Any]) -> Dict[str, Any]:
    termination_reason = payload.get("terminationReason", "")
    if termination_reason in {
        "USER_CANCELED",
        "ERROR",
        "MAX_INVOCATIONS",
        "MAX_FORCED_INVOCATIONS",
        "MAX_TOKEN_BUDGET_EXCEEDED",
    }:
        return {"decision": "allow"}

    if not payload.get("fullyIdle", True):
        return {"decision": "allow"}

    state = _load_state(payload)
    _sync_subagents_from_transcript(payload, state)

    active_branches = [
        cid
        for cid, meta in state.get("subagents", {}).items()
        if meta.get("active_branch")
    ]
    active_pending = [
        p.get("role", "implementer")
        for p in state.get("pending_spawns", [])
        if p.get("active_branch")
    ]

    if active_branches or active_pending:
        exec_key = f"{payload.get('conversationId', '')}:{payload.get('executionNum', 0)}"
        reminded = state.setdefault("stop_reminded_executions", [])
        if exec_key not in reminded:
            reminded.append(exec_key)
            _save_state(payload, state)
            targets = ", ".join(active_branches or active_pending)
            return {
                "decision": "continue",
                "reason": (
                    f"[ORCHESTRATOR GUARD] Uncleaned branched subagent worktrees remain ({targets}). "
                    "Verify/merge branch changes and call manage_subagents(Action='kill') before finishing."
                ),
            }

    if not _is_orchestrator_caller(payload, state):
        last_msg = _extract_last_model_response(payload)
        if last_msg is not None:
            report_err = validate_subagent_report(last_msg)
            if report_err:
                reminded = state.setdefault("stop_reminded_executions", [])
                report_key = f"report:{payload.get('conversationId', '')}:{payload.get('executionNum', 0)}"
                if report_key not in reminded:
                    reminded.append(report_key)
                    _save_state(payload, state)
                    return {
                        "decision": "continue",
                        "reason": (
                            f"[ORCHESTRATOR GUARD] Subagent context-separation violation: {report_err} "
                            "Return ONLY the exact 4-header REPORT FORMAT (STATUS, FILES_CHANGED, VERIFICATION, NOTES; "
                            "allowing up to 10 '-' bullet lines under NOTES and requiring inline error cause plus "
                            "file:// .log URI when BLOCKED or FAILED)."
                        ),
                    }

    cb = _get_circuit_breaker(state, payload)
    if cb.get("tripped"):
        return {"decision": "allow"}

    inv_key = _current_invocation_key(payload, state)
    mutated_in_turn = bool(
        state.get("direct_budget", {}).get(inv_key, {}).get("mutated_files")
    )
    receipt = _get_test_receipt(state, payload)
    diff_text = (
        _get_workspace_git_diff(payload)
        if (payload.get("gitDiff") or receipt.get("dirty") or mutated_in_turn)
        else ""
    )
    if diff_text:
        tamper_finding = audit_diff_text(diff_text)
        if tamper_finding:
            exec_key = f"tamper:{payload.get('conversationId', '')}:{payload.get('executionNum', 0)}"
            reminded = state.setdefault("stop_reminded_executions", [])
            if exec_key not in reminded:
                reminded.append(exec_key)
                _save_state(payload, state)
                return {"decision": "continue", "reason": tamper_finding}

    if receipt.get("dirty") and not receipt.get("verified"):
        exec_key = f"receipt:{payload.get('conversationId', '')}:{payload.get('executionNum', 0)}"
        reminded = state.setdefault("stop_reminded_executions", [])
        if exec_key not in reminded:
            reminded.append(exec_key)
            _save_state(payload, state)
            last_file = Path(receipt.get("last_mutated_file") or "workspace").name
            return {
                "decision": "continue",
                "reason": (
                    f"[ORCHESTRATOR GUARD] Empirical test receipt missing: code file "
                    f"'{last_file}' was modified without a subsequent passing test run "
                    "recorded in PostToolUse. Run the test suite to verify changes before finishing."
                ),
            }

    return {"decision": "allow"}


def register_subagent(
    payload: Dict[str, Any],
    conversation_id: str,
    type_name: str,
    model: str,
    workspace: str = "inherit",
    role: str = "",
) -> None:
    """Helper for explicitly registering a subagent conversationId -> tier mapping."""
    state = _load_state(payload)
    cid = conversation_id.strip().lower()
    state.setdefault("subagents", {})[cid] = {
        "role": role or type_name,
        "type": type_name,
        "type_name": type_name,
        "model": model,
        "workspace": workspace,
        "active_branch": workspace == "branch",
    }
    _save_state(payload, state)


def _cli_status(payload: Dict[str, Any]) -> Dict[str, Any]:
    state = _load_state(payload)
    inv_key = _current_invocation_key(payload, state)
    return {
        "projectRoot": str(PROJECT_ROOT),
        "stateFile": str(_state_file_path(payload)),
        "rootConversationId": state.get("root_conversation_id"),
        "currentInvocation": inv_key,
        "activeSubagents": state.get("subagents", {}),
        "pendingSpawns": state.get("pending_spawns", []),
        "directBudget": state.get("direct_budget", {}).get(inv_key, {}),
        "testReceipt": _get_test_receipt(state, payload),
        "circuitBreaker": _get_circuit_breaker(state, payload),
    }


def _cli_reset(payload: Dict[str, Any]) -> Dict[str, Any]:
    fresh = _default_state()
    _save_state(payload, fresh)
    return {"status": "reset", "stateFile": str(_state_file_path(payload))}


def _cli_doctor() -> Dict[str, Any]:
    plugin_hooks = (
        PROJECT_ROOT / ".agents" / "plugins" / "pro-orchestrator" / "hooks.json"
    )
    workspace_hooks = PROJECT_ROOT / ".agents" / "hooks.json"
    root_agents = PROJECT_ROOT / "AGENTS.md"
    plugin_agents = (
        PROJECT_ROOT
        / ".agents"
        / "plugins"
        / "pro-orchestrator"
        / "rules"
        / "AGENTS.md"
    )
    issues: List[str] = []
    if workspace_hooks.exists():
        try:
            wh_data = json.loads(workspace_hooks.read_text(encoding="utf-8"))
            if wh_data.get("pro-orchestrator-guard", {}).get("enabled", True) is not False:
                issues.append(
                    "Duplicate .agents/hooks.json is enabled alongside plugin hooks.json."
                )
        except (OSError, json.JSONDecodeError):
            issues.append("Unreadable .agents/hooks.json.")
    if root_agents.exists() and plugin_agents.exists():
        try:
            if (
                root_agents.read_text(encoding="utf-8")
                != plugin_agents.read_text(encoding="utf-8")
            ):
                issues.append(
                    "Root AGENTS.md content diverges from .agents/plugins/pro-orchestrator/rules/AGENTS.md."
                )
        except OSError:
            issues.append("Unable to compare AGENTS.md files.")
    return {
        "healthy": len(issues) == 0 and plugin_hooks.exists(),
        "pluginHooksPresent": plugin_hooks.exists(),
        "issues": issues,
    }


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "pre-tool"
    if mode == "doctor":
        json.dump(_cli_doctor(), sys.stdout, indent=2)
        return

    try:
        payload = json.load(sys.stdin) if not sys.stdin.isatty() else {}
    except json.JSONDecodeError:
        payload = {}

    if mode == "status":
        json.dump(_cli_status(payload), sys.stdout, indent=2)
        return
    if mode == "reset":
        json.dump(_cli_reset(payload), sys.stdout, indent=2)
        return

    if mode == "pre-invocation":
        result = handle_pre_invocation(payload)
    elif mode == "post-tool":
        result = handle_post_tool(payload)
    elif mode == "stop":
        result = handle_stop(payload)
    else:
        result = handle_pre_tool(payload)

    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
