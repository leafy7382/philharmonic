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


def _find_project_root() -> Path:
    """Resolve the project workspace root from the script location."""
    current = SCRIPT_DIR
    for parent in [current] + list(current.parents):
        if parent.name == ".agents":
            return parent.parent.resolve()
    return Path.cwd().resolve()


PROJECT_ROOT = _find_project_root()


def is_pro_tier(model_name: str) -> bool:
    """Matches current and future Pro releases while rejecting Flash, Lite, or Auto."""
    normalized = (model_name or "").strip().lower()
    if not normalized or normalized in {"auto", "none"}:
        return False
    if "flash" in normalized or "lite" in normalized:
        return False
    return any(token in normalized for token in ("pro", "argon", "ultra"))


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
                    "which is not a Pro-tier model. Warn the user to switch Model Selection "
                    "to a Pro model before running orchestration tasks."
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


def handle_pre_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    tool_call = payload.get("toolCall", {})
    tool_name = tool_call.get("name", "")
    args = tool_call.get("args", {}) or {}

    if tool_name == "define_subagent":
        return _handle_define_subagent(args)

    if tool_name == "invoke_subagent":
        return _handle_invoke_subagent(payload, args)

    if tool_name == "send_message":
        return _handle_send_message(payload, args)

    if tool_name == "manage_subagents":
        return _handle_manage_subagents(payload, args)

    # Workspace auto-allow gate for project-scoped operations
    roots = _allowed_roots(payload)

    if tool_name == "run_command":
        cwd = args.get("Cwd", "")
        cmd = args.get("CommandLine", "")
        if _is_within_allowed_roots(cwd, roots) and _command_stays_in_workspace(
            cmd, cwd, roots
        ):
            return {
                "decision": "allow",
                "permissionOverrides": [f"command({cmd})", "command(*)"],
            }
        return {"decision": "ask"}

    for path_key in ("TargetFile", "AbsolutePath", "SearchPath", "DirectoryPath"):
        if path_key in args:
            if _is_within_allowed_roots(args.get(path_key), roots):
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
