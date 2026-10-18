#!/usr/bin/env python3
"""Deterministic enforcement and workspace permission hook for the Pro-Pinned Orchestrator.

Responsibilities:
1. PreInvocation:
   - Verifies parent orchestrator is pinned to a Pro-tier model (Pro / Argon / Ultra).
   - Deduplicates warnings per (conversationId, invocationNum).
2. PreToolUse:
   - Auto-allows tool executions (run_command, write_to_file, replace_file_content,
     view_file, etc.) that operate strictly within the project workspace or
     conversation artifact directory.
   - Forces enable_subagent_tools=False on custom subagent definitions (define_subagent).
   - Blocks Model='pro' and rewrites Model='inherit' to 'flash' or 'flash_lite'
     on invoke_subagent.
   - Runs ASD-STE100 structural linting (ste_lint.py) on invoke_subagent Prompts
     and send_message Messages.
   - Guards send_message against tier-bypass (preventing multi-step or mutation
     tasks from being sent to a flash_lite 'scout' subagent).
3. PostToolUse:
   - Synchronizes spawned subagent metadata and branch lifecycle state in
     <artifactDirectoryPath>/orchestrator_state.json.
4. Stop:
   - Allows turn completion when fullyIdle is False (so reactive wakeup works).
   - Blocks premature final completion when fullyIdle is True and uncleaned
     branched subagent worktrees remain active.
"""

import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import ste_lint  # noqa: E402

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
CONVERSATION_ID_RE = re.compile(
    r'\\*"conversationId\\*"\s*:\s*\\*"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\\*"',
    re.I,
)

MAX_DIRECT_MUTATION_FILES = 2
MAX_DIRECT_COMMANDS = 2
MUTATION_TOOLS = {
    "write_to_file",
    "replace_file_content",
    "multi_replace_file_content",
}
GIT_COMMAND_RE = re.compile(r"^\s*git\b")
TEST_RUNNER_RE = re.compile(
    r"\b(pytest|unittest|cargo\s+test|npm\s+test|pnpm\s+test|yarn\s+test|go\s+test|jest|vitest|tox|nox)\b"
)
MAX_CONSECUTIVE_FAILURES = 2
DOC_EXTENSIONS = {".md", ".txt", ".rst", ".adoc"}
TEST_FILE_RE = re.compile(
    r"(^|/)(tests?|__tests__)/|(^|/)(test_[^/]+\.py|[^/]+_test\.(py|go|rs)|[^/]+\.(test|spec)\.[jt]sx?)$",
    re.I,
)
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




def _find_project_root() -> Path:
    """Resolve the project workspace root from the script location."""
    current = SCRIPT_DIR
    for parent in [current] + list(current.parents):
        if parent.name == ".agents":
            return parent.parent.resolve()
    return Path.cwd().resolve()


PROJECT_ROOT = _find_project_root()


SMALL_TIER_RE = re.compile(
    r"(?<![a-z])(flash|lite|mini|nano|haiku|small|tiny)(?![a-z])", re.I
)


def is_pro_tier(model_name: str) -> bool:
    """Accept any frontier orchestrator model and reject small or auto-routed tiers.

    Uses a denylist of small-tier markers so that flagship models from any vendor
    (Gemini Pro/Argon/Ultra, Claude Opus/Sonnet, GPT, o-series, Grok, DeepSeek)
    pass without code changes.
    """
    normalized = (model_name or "").strip().lower()
    if not normalized or normalized in {"auto", "none"}:
        return False
    return not SMALL_TIER_RE.search(normalized)


def _state_file_path(payload: Dict[str, Any]) -> Path:
    artifact_dir = payload.get("artifactDirectoryPath")
    if artifact_dir:
        base = Path(artifact_dir)
    else:
        base = PROJECT_ROOT / ".agents" / ".state"
    base.mkdir(parents=True, exist_ok=True)
    return base / "orchestrator_state.json"


def _load_state(payload: Dict[str, Any]) -> Dict[str, Any]:
    path = _state_file_path(payload)
    if path.exists():
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    return data
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "warned_invocations": [],
        "recorded_spawn_calls": [],
        "pending_spawns": [],
        "subagents": {},
        "stop_reminded_executions": [],
        "direct_budget": {},
        "test_receipt": {},
        "circuit_breaker": {},
    }


def _save_state(payload: Dict[str, Any], state: Dict[str, Any]) -> None:
    path = _state_file_path(payload)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, sort_keys=True)
    except OSError:
        pass


def _sync_subagents_from_transcript(
    payload: Dict[str, Any], state: Dict[str, Any]
) -> None:
    """Scan transcript.jsonl (if readable) to pair pending_spawns with conversationIds."""
    transcript_path = payload.get("transcriptPath")
    if not transcript_path or not os.path.exists(transcript_path):
        return

    pending: List[Dict[str, Any]] = state.get("pending_spawns", [])
    subagents: Dict[str, Dict[str, Any]] = state.setdefault("subagents", {})
    parent_cid = (payload.get("conversationId") or "").lower()

    try:
        with open(transcript_path, encoding="utf-8") as fh:
            for raw_line in fh:
                if "invoke_subagent" not in raw_line and not pending:
                    continue
                try:
                    step = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                content = json.dumps(step)
                for match in CONVERSATION_ID_RE.finditer(content):
                    cid = match.group(1).lower()
                    if cid == parent_cid or cid in subagents:
                        continue
                    if pending:
                        spawn_meta = pending.pop(0)
                        subagents[cid] = spawn_meta
    except OSError:
        return


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
            # Also allow sibling brain/<conversation-id>/scratch and worktrees/<conversation-id>
            # so branched/shared subagents can write to their worktree and parent scratch logs.
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
    """Return True unless the command targets absolute or relative paths outside allowed roots."""
    base_dir = Path(os.path.expanduser(cwd)).resolve() if cwd else PROJECT_ROOT
    # Check absolute path tokens
    for match in re.finditer(r"(?:^|[\s=\"'])(/[A-Za-z0-9._/\-]+)", command_line):
        abs_path = match.group(1)
        if abs_path.startswith(("/dev/", "/usr/", "/bin/", "/lib/", "/proc/")):
            continue
        if not _is_within_allowed_roots(abs_path, roots):
            return False
    # Check relative upward traversal tokens containing '..'
    for match in re.finditer(
        r"(?:^|[\s=\"'])([A-Za-z0-9._\-/]*\.\.[A-Za-z0-9._\-/]*)", command_line
    ):
        rel_token = match.group(1)
        candidate = (base_dir / rel_token).resolve()
        if not _is_within_allowed_roots(str(candidate), roots):
            return False
    return True


def _lint_inter_agent_text(text: str, label: str) -> Optional[str]:
    """Run STE-100 structural checks on inter-agent prompt/message text."""
    if not text or not text.strip():
        return f"{label} cannot be empty."
    findings, _ = ste_lint.lint(text, filename=label, strict_instructions=True)
    hard_findings = [f for f in findings if f.get("level") == "advisory-free"]
    if not hard_findings:
        return None
    formatted = "; ".join(
        f"line {f['line']} [{f['rule']}]: {f['message']} (matched: {f['match']!r})"
        for f in hard_findings[:5]
    )
    return (
        f"[STE-100 LINT GATE] {label} has {len(hard_findings)} structural violation(s): "
        f"{formatted}. Rewrite using short active-voice sentences without semicolons or phrasal verbs."
    )


def handle_pre_invocation(payload: Dict[str, Any]) -> Dict[str, Any]:
    model_name = payload.get("modelName", "")
    if is_pro_tier(model_name):
        return {}

    state = _load_state(payload)
    dedup_key = f"{payload.get('conversationId', '')}:{payload.get('invocationNum', 0)}"
    warned: List[str] = state.setdefault("warned_invocations", [])
    if dedup_key in warned:
        return {}

    warned.append(dedup_key)
    _save_state(payload, state)
    return {
        "injectSteps": [
            {
                "ephemeralMessage": (
                    f"[ORCHESTRATOR GUARD] Active session model is '{model_name}', "
                    "which is a small or auto-routed tier. Warn the user to switch Model Selection "
                    "to a frontier model (for example Gemini Pro, Claude Opus, or GPT) before running orchestration tasks."
                )
            }
        ]
    }


def _handle_define_subagent(args: Dict[str, Any]) -> Dict[str, Any]:
    if args.get("enable_subagent_tools") is True:
        return {
            "decision": "allow",
            "overwrite": {"enable_subagent_tools": False},
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
    payload: Dict[str, Any], args: Dict[str, Any]
) -> Dict[str, Any]:
    subagents = args.get("Subagents", [])
    if not subagents:
        return {"decision": "deny", "reason": "Subagents list cannot be empty."}

    rewritten = False
    updated_subagents: List[Dict[str, Any]] = []
    spawn_records: List[Dict[str, Any]] = []

    for idx, entry in enumerate(subagents):
        entry_copy = dict(entry)
        requested_model = entry_copy.get("Model", "inherit")
        type_name = entry_copy.get("TypeName", "")
        role_label = entry_copy.get("Role", type_name or f"subagent[{idx}]")

        if requested_model == "pro":
            return {
                "decision": "deny",
                "reason": (
                    f"Subagent '{role_label}' requested Model='pro'. "
                    "Subagents must use 'flash' or 'flash_lite'."
                ),
            }

        prompt_text = entry_copy.get("Prompt", "")
        lint_error = _lint_inter_agent_text(prompt_text, f"Subagent[{role_label}].Prompt")
        if lint_error:
            return {"decision": "deny", "reason": lint_error}

        target_model = requested_model
        if target_model not in ALLOWED_SUBAGENT_MODELS:
            target_model = DEFAULT_MODEL_BY_TYPE.get(type_name, "flash")

        # Token optimization: automatically downgrade read-only subagents with <= 3 rules to flash_lite
        if target_model == "flash" and _is_simple_readonly_task(type_name, prompt_text):
            target_model = "flash_lite"

        if entry_copy.get("Model") != target_model:
            entry_copy["Model"] = target_model
            rewritten = True

        workspace_mode = entry_copy.get("Workspace", "inherit")
        spawn_records.append(
            {
                "role": role_label,
                "type": type_name,
                "model": entry_copy["Model"],
                "workspace": workspace_mode,
                "active_branch": workspace_mode == "branch",
            }
        )
        updated_subagents.append(entry_copy)

    state = _load_state(payload)
    tool_call = payload.get("toolCall", {})
    call_sig = json.dumps(
        [
            payload.get("conversationId", ""),
            payload.get("invocationNum", 0),
            payload.get("executionNum", 0),
            tool_call.get("id", ""),
            [(r["type"], r["role"], r["workspace"]) for r in spawn_records],
        ],
        sort_keys=True,
    )
    recorded_calls: List[str] = state.setdefault("recorded_spawn_calls", [])
    if call_sig not in recorded_calls:
        recorded_calls.append(call_sig)
        state.setdefault("pending_spawns", []).extend(spawn_records)
        _save_state(payload, state)

    if rewritten:
        return {
            "decision": "allow",
            "overwrite": {"Subagents": updated_subagents},
        }
    return {"decision": "allow"}


def _handle_send_message(
    payload: Dict[str, Any], args: Dict[str, Any]
) -> Dict[str, Any]:
    recipient = (args.get("Recipient") or "").strip().lower()
    message = args.get("Message") or ""

    lint_error = _lint_inter_agent_text(message, "send_message.Message")
    if lint_error:
        return {"decision": "deny", "reason": lint_error}

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
    payload: Dict[str, Any], args: Dict[str, Any]
) -> Dict[str, Any]:
    action = args.get("Action", "")
    state = _load_state(payload)
    _sync_subagents_from_transcript(payload, state)
    subagents = state.get("subagents", {})

    if action == "kill_all":
        for meta in subagents.values():
            meta["active_branch"] = False
        for pending in state.get("pending_spawns", []):
            pending["active_branch"] = False
        _save_state(payload, state)
    elif action == "kill":
        target_ids = {
            cid.strip().lower() for cid in (args.get("ConversationIds") or [])
        }
        for cid in target_ids:
            if cid in subagents:
                subagents[cid]["active_branch"] = False
        # Also clear matching count from pending_spawns if transcript wasn't synced
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


def _is_orchestrator_caller(payload: Dict[str, Any], state: Dict[str, Any]) -> bool:
    """Return True when the caller is the Pro-tier root orchestrator rather than a worker subagent."""
    if not is_pro_tier(payload.get("modelName", "")):
        return False
    cid = (payload.get("conversationId") or "").strip().lower()
    if cid and cid in state.get("subagents", {}):
        return False
    return True


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


def _check_and_record_direct_budget(
    payload: Dict[str, Any], tool_name: str, args: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Enforce direct execution fast-path limits on the Pro-tier orchestrator."""
    state = _load_state(payload)
    if not _is_orchestrator_caller(payload, state):
        return None

    inv_key = str(payload.get("invocationNum", 0))
    budget = state.setdefault("direct_budget", {}).setdefault(
        inv_key, {"mutated_files": [], "commands": [], "seen_calls": []}
    )
    call_id = (payload.get("toolCall") or {}).get("id") or ""

    if tool_name in MUTATION_TOOLS:
        target = args.get("TargetFile") or args.get("AbsolutePath") or ""
        if not target or _is_artifact_or_worktree_path(target, payload):
            return None
        try:
            norm_target = str(Path(os.path.expanduser(target)).resolve())
        except OSError:
            return None

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
        if GIT_COMMAND_RE.match(cmd):
            return None

        mutated_files = budget.get("mutated_files", [])
        if mutated_files and TEST_RUNNER_RE.search(cmd):
            return {
                "decision": "deny",
                "reason": (
                    "[ORCHESTRATOR GUARD] Direct workspace edit followed by test execution "
                    "detected in the orchestrator. Delegate code and test execution loops "
                    "to an 'implementer' subagent (Model='flash')."
                ),
            }

        seen_calls: List[str] = budget.setdefault("seen_calls", [])
        if call_id and call_id in seen_calls:
            return None

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
    """Block edits to test files that delete tests/assertions or add skip markers."""
    target = args.get("TargetFile") or args.get("AbsolutePath") or ""
    if not target or _is_artifact_or_worktree_path(target, payload):
        return None
    if not _is_test_file(target):
        return None

    chunks: List[Tuple[str, str]] = []
    if tool_name == "replace_file_content":
        chunks.append(
            (args.get("TargetContent") or "", args.get("ReplacementContent") or "")
        )
    elif tool_name == "multi_replace_file_content":
        for ch in args.get("ReplacementChunks") or []:
            chunks.append(
                (ch.get("TargetContent") or "", ch.get("ReplacementContent") or "")
            )
    elif tool_name == "write_to_file":
        new_code = args.get("CodeContent") or ""
        old_code = ""
        try:
            resolved = Path(os.path.expanduser(target)).resolve()
            if resolved.exists():
                old_code = resolved.read_text(encoding="utf-8")
        except OSError:
            old_code = ""
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
            old_tests = len(TEST_DEF_RE.findall(old_text))
            new_tests = len(TEST_DEF_RE.findall(new_text))
            if new_tests < old_tests:
                return {
                    "decision": "deny",
                    "reason": (
                        f"[ORCHESTRATOR GUARD] Test tampering blocked in '{file_name}': "
                        f"edit reduces test count from {old_tests} to {new_tests}."
                    ),
                }

            old_asserts = len(ASSERTION_RE.findall(old_text))
            new_asserts = len(ASSERTION_RE.findall(new_text))
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

    def _check_accumulated(fname: Optional[str], rem: List[str], add: List[str]) -> Optional[str]:
        if not fname or not _is_test_file(fname):
            return None
        rem_blob = "\n".join(rem)
        add_blob = "\n".join(add)
        short_name = Path(fname).name

        if len(TAMPER_SKIP_RE.findall(add_blob)) > len(TAMPER_SKIP_RE.findall(rem_blob)):
            return (
                f"[ORCHESTRATOR GUARD] Diff audit detected test skip/xfail marker added in '{short_name}'."
            )
        if len(TRIVIAL_ASSERT_RE.findall(add_blob)) > len(TRIVIAL_ASSERT_RE.findall(rem_blob)):
            return (
                f"[ORCHESTRATOR GUARD] Diff audit detected trivial 'assert True' added in '{short_name}'."
            )
        rem_tests = len(TEST_DEF_RE.findall(rem_blob))
        add_tests = len(TEST_DEF_RE.findall(add_blob))
        if rem_tests > add_tests:
            return (
                f"[ORCHESTRATOR GUARD] Diff audit detected net removal of {rem_tests - add_tests} "
                f"test definition(s) in '{short_name}'."
            )
        rem_asserts = len(ASSERTION_RE.findall(rem_blob))
        add_asserts = len(ASSERTION_RE.findall(add_blob))
        if rem_asserts > add_asserts:
            return (
                f"[ORCHESTRATOR GUARD] Diff audit detected net removal of {rem_asserts - add_asserts} "
                f"assertion(s) in '{short_name}'."
            )
        return None

    for raw_line in diff_text.splitlines():
        if raw_line.startswith("+++ b/"):
            verdict = _check_accumulated(current_file, removed_lines, added_lines)
            if verdict:
                return verdict
            current_file = raw_line[6:].strip()
            removed_lines = []
            added_lines = []
        elif raw_line.startswith("---") or raw_line.startswith("@@") or raw_line.startswith("diff --git"):
            continue
        elif raw_line.startswith("-"):
            removed_lines.append(raw_line[1:])
        elif raw_line.startswith("+"):
            added_lines.append(raw_line[1:])

    return _check_accumulated(current_file, removed_lines, added_lines)


def _get_circuit_breaker(state: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    inv_key = str(payload.get("invocationNum", 0))
    return state.setdefault("circuit_breaker", {}).setdefault(
        inv_key,
        {"failure_streak": 0, "tripped": False, "seen_failures": [], "last_error": ""},
    )


def _get_test_receipt(state: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    inv_key = str(payload.get("invocationNum", 0))
    return state.setdefault("test_receipt", {}).setdefault(
        inv_key,
        {
            "dirty": False,
            "verified": False,
            "last_mutated_file": "",
            "last_test_cmd": "",
        },
    )


def handle_pre_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    tool_call = payload.get("toolCall", {})
    tool_name = tool_call.get("name", "")
    args = tool_call.get("args", {}) or {}

    state = _load_state(payload)
    cb = _get_circuit_breaker(state, payload)
    if cb.get("tripped"):
        if tool_name in MUTATION_TOOLS:
            target = args.get("TargetFile") or args.get("AbsolutePath") or ""
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

    if tool_name == "define_subagent":
        return _handle_define_subagent(args)

    if tool_name == "invoke_subagent":
        return _handle_invoke_subagent(payload, args)

    if tool_name == "send_message":
        return _handle_send_message(payload, args)

    if tool_name == "manage_subagents":
        return _handle_manage_subagents(payload, args)

    if tool_name in MUTATION_TOOLS:
        tamper_denial = _check_test_tampering_in_edit(tool_name, args, payload)
        if tamper_denial is not None:
            return tamper_denial

    # Workspace auto-allow gate for project-scoped operations
    roots = _allowed_roots(payload)

    if tool_name == "run_command":
        cwd = args.get("Cwd", "")
        cmd = args.get("CommandLine", "")
        if _is_within_allowed_roots(cwd, roots) and _command_stays_in_workspace(
            cmd, cwd, roots
        ):
            budget_denial = _check_and_record_direct_budget(payload, tool_name, args)
            if budget_denial:
                return budget_denial
            return {
                "decision": "allow",
                "permissionOverrides": [f"command({cmd})", "command(*)"],
            }
        return {"decision": "ask"}

    for path_key in ("TargetFile", "AbsolutePath", "SearchPath", "DirectoryPath"):
        if path_key in args:
            target_path = args.get(path_key)
            if _is_within_allowed_roots(target_path, roots):
                budget_denial = _check_and_record_direct_budget(payload, tool_name, args)
                if budget_denial:
                    return budget_denial
                if tool_name in MUTATION_TOOLS:
                    target = target_path or ""
                    if not _is_artifact_or_worktree_path(target, payload) and _is_code_file(target):
                        state = _load_state(payload)
                        receipt = _get_test_receipt(state, payload)
                        receipt["dirty"] = True
                        receipt["verified"] = False
                        try:
                            receipt["last_mutated_file"] = str(Path(os.path.expanduser(target)).resolve())
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
        subagents = (tool_call.get("args") or {}).get("Subagents") or []
        pending = state.get("pending_spawns", [])
        if subagents and len(pending) >= len(subagents):
            del pending[-len(subagents) :]
        else:
            state["pending_spawns"] = []
    else:
        _sync_subagents_from_transcript(payload, state)

    cb = _get_circuit_breaker(state, payload)
    receipt = _get_test_receipt(state, payload)
    error_text = str(payload.get("error") or "").strip()
    call_id = str(tool_call.get("id") or payload.get("stepIdx") or "")

    if tool_name in {"run_command", "invoke_subagent"}:
        cmd = ((tool_call.get("args") or {}).get("CommandLine") or "").strip()
        is_test_cmd = (tool_name == "run_command" and bool(TEST_RUNNER_RE.search(cmd)))
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
            if tool_name == "invoke_subagent" or is_test_cmd or (tool_name == "run_command" and not GIT_COMMAND_RE.match(cmd)):
                cb["failure_streak"] = 0
                cb["tripped"] = False
            if is_test_cmd:
                receipt["dirty"] = False
                receipt["verified"] = True
                receipt["last_test_cmd"] = cmd

    _save_state(payload, state)
    return {}


def handle_stop(payload: Dict[str, Any]) -> Dict[str, Any]:
    termination_reason = payload.get("terminationReason", "")
    # Never block user cancellations, errors, or hard budget caps
    if termination_reason in {
        "USER_CANCELED",
        "ERROR",
        "MAX_INVOCATIONS",
        "MAX_FORCED_INVOCATIONS",
        "MAX_TOKEN_BUDGET_EXCEEDED",
    }:
        return {"decision": "allow"}

    # When background tasks or subagents are still running (fullyIdle == False),
    # allow the turn to end so Antigravity's reactive wakeup can resume execution.
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

    cb = _get_circuit_breaker(state, payload)
    if cb.get("tripped"):
        return {"decision": "allow"}

    if payload.get("gitDiff"):
        tamper_finding = audit_diff_text(payload["gitDiff"])
        if tamper_finding:
            exec_key = f"tamper:{payload.get('conversationId', '')}:{payload.get('executionNum', 0)}"
            reminded = state.setdefault("stop_reminded_executions", [])
            if exec_key not in reminded:
                reminded.append(exec_key)
                _save_state(payload, state)
                return {"decision": "continue", "reason": tamper_finding}

    receipt = _get_test_receipt(state, payload)
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
        "model": model,
        "workspace": workspace,
        "active_branch": workspace == "branch",
    }
    _save_state(payload, state)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "pre-tool"
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        payload = {}

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
