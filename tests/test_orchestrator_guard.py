#!/usr/bin/env python3
"""Unit tests for orchestrator_guard.py and ste_lint.py."""

import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, Optional
import unittest

SCRIPT_DIR = (
    Path(__file__).resolve().parent.parent
    / ".agents"
    / "plugins"
    / "pro-orchestrator"
    / "scripts"
)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import orchestrator_guard  # noqa: E402
import ste_lint  # noqa: E402


VALID_PROMPT = """TASK: Add input length check to `parse_token` in `src/auth/parser.py`.

RULES:
1. Open `src/auth/parser.py`.
2. Reject strings longer than 4096 bytes with `ValueError`.
3. Do not change any file outside `src/auth/parser.py`.
4. Run `pytest tests/test_parser.py` to check the result.

REPORT FORMAT:
Return only these four lines:
STATUS: COMPLETED | BLOCKED | FAILED
FILES_CHANGED: <comma-separated paths or NONE>
VERIFICATION: <one sentence stating the command run and exit result>
NOTES: <one sentence stating findings or blocker details>
"""


class TestOrchestratorGuard(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.artifact_dir = self.temp_dir.name
        self.base_payload = {
            "conversationId": "6c0f93da-2912-400b-9b41-a46eacc1bfa3",
            "workspacePaths": [str(orchestrator_guard.PROJECT_ROOT)],
            "artifactDirectoryPath": self.artifact_dir,
            "modelName": "Gemini 4 Argon (Medium)",
            "invocationNum": 1,
            "executionNum": 1,
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_is_pro_tier_classification(self) -> None:
        self.assertTrue(orchestrator_guard.is_pro_tier("Gemini 4 Argon (Medium)"))
        self.assertTrue(orchestrator_guard.is_pro_tier("gemini-3.1-pro-preview"))
        self.assertFalse(orchestrator_guard.is_pro_tier("gemini-3-flash"))
        self.assertFalse(orchestrator_guard.is_pro_tier("gemini-flash-lite"))
        self.assertFalse(orchestrator_guard.is_pro_tier("auto"))
        self.assertFalse(orchestrator_guard.is_pro_tier(""))

    def test_is_pro_tier_accepts_other_frontier_vendors(self) -> None:
        for name in (
            "Claude Opus 5.5 (Medium)",
            "claude-sonnet-5",
            "gpt-6",
            "gpt-6-pro",
            "o5",
            "grok-5",
            "deepseek-v4",
        ):
            self.assertTrue(orchestrator_guard.is_pro_tier(name), name)
        for name in ("claude-haiku-5", "gpt-6-mini", "gpt-6-nano", "gemini-3-flash-lite"):
            self.assertFalse(orchestrator_guard.is_pro_tier(name), name)

    def test_pre_invocation_pro_vs_non_pro_and_dedup(self) -> None:
        # Root on a Pro model returns {}
        res_pro = orchestrator_guard.handle_pre_invocation(self.base_payload)
        self.assertEqual(res_pro, {})

        # Root on a small model returns {} and stores root_conversation_id in state
        small_payload = dict(self.base_payload, modelName="gemini-flash", invocationNum=2)
        res_small = orchestrator_guard.handle_pre_invocation(small_payload)
        self.assertEqual(res_small, {})
        state = orchestrator_guard._load_state(small_payload)
        self.assertEqual(
            state.get("root_conversation_id"),
            self.base_payload["conversationId"].lower(),
        )

        # A registered subagent gets injectSteps with isolation reminder once, deduped on second call
        sub_cid = "sub-dedup-uuid-9999"
        orchestrator_guard.register_subagent(
            self.base_payload,
            conversation_id=sub_cid,
            type_name="scout",
            model="flash_lite",
        )
        sub_payload = dict(
            self.base_payload,
            conversationId=sub_cid,
            modelName="gemini-flash-lite",
            invocationNum=1,
        )
        res_sub1 = orchestrator_guard.handle_pre_invocation(sub_payload)
        self.assertIn("injectSteps", res_sub1)
        self.assertIn(
            "Subagent context isolation active",
            res_sub1["injectSteps"][0]["ephemeralMessage"],
        )

        res_sub2 = orchestrator_guard.handle_pre_invocation(sub_payload)
        self.assertEqual(res_sub2, {})

    def test_define_subagent_forces_subagent_tools_false(self) -> None:
        payload = dict(
            self.base_payload,
            toolCall={
                "name": "define_subagent",
                "args": {
                    "name": "scout",
                    "enable_subagent_tools": True,
                },
            },
        )
        res = orchestrator_guard.handle_pre_tool(payload)
        self.assertEqual(res["decision"], "allow")
        self.assertEqual(res["overwrite"], {"enable_subagent_tools": False})

    def test_invoke_subagent_blocks_pro_model(self) -> None:
        payload = dict(
            self.base_payload,
            toolCall={
                "name": "invoke_subagent",
                "args": {
                    "Subagents": [
                        {
                            "TypeName": "implementer",
                            "Role": "Auth Implementer",
                            "Model": "pro",
                            "Prompt": VALID_PROMPT,
                        }
                    ]
                },
            },
        )
        res = orchestrator_guard.handle_pre_tool(payload)
        self.assertEqual(res["decision"], "deny")
        self.assertIn("Model='pro'", res["reason"])

    def test_invoke_subagent_rewrites_inherit_to_tier_defaults(self) -> None:
        payload = dict(
            self.base_payload,
            toolCall={
                "name": "invoke_subagent",
                "args": {
                    "Subagents": [
                        {
                            "TypeName": "scout",
                            "Role": "Symbol Scout",
                            "Model": "inherit",
                            "Prompt": VALID_PROMPT,
                        },
                        {
                            "TypeName": "implementer",
                            "Role": "Code Worker",
                            "Prompt": VALID_PROMPT,
                        },
                    ]
                },
            },
        )
        res = orchestrator_guard.handle_pre_tool(payload)
        self.assertEqual(res["decision"], "allow")
        rewritten = res["overwrite"]["Subagents"]
        self.assertEqual(rewritten[0]["Model"], "flash_lite")
        self.assertEqual(rewritten[1]["Model"], "flash")

    def test_ste100_lint_gate_blocks_violations(self) -> None:
        bad_prompt = (
            "TASK: Spin up the parser; perform an analysis of the seamless token.\n"
        )
        payload = dict(
            self.base_payload,
            toolCall={
                "name": "invoke_subagent",
                "args": {
                    "Subagents": [
                        {
                            "TypeName": "scout",
                            "Role": "Bad Scout",
                            "Model": "flash_lite",
                            "Prompt": bad_prompt,
                        }
                    ]
                },
            },
        )
        res = orchestrator_guard.handle_pre_tool(payload)
        self.assertEqual(res["decision"], "deny")
        self.assertIn("[STE-100 LINT GATE]", res["reason"])
        self.assertIn("semicolon", res["reason"])

    def test_send_message_tier_bypass_blocked_for_flash_lite(self) -> None:
        scout_cid = "11111111-2222-3333-4444-555555555555"
        orchestrator_guard.register_subagent(
            self.base_payload,
            conversation_id=scout_cid,
            type_name="scout",
            model="flash_lite",
            workspace="inherit",
            role="File Scout",
        )

        # Sending code edit instructions to flash_lite is denied
        mutate_payload = dict(
            self.base_payload,
            toolCall={
                "name": "send_message",
                "args": {
                    "Recipient": scout_cid,
                    "Message": "Edit `src/auth/parser.py` to add token validation.",
                },
            },
        )
        res_deny = orchestrator_guard.handle_pre_tool(mutate_payload)
        self.assertEqual(res_deny["decision"], "deny")
        self.assertIn("flash_lite", res_deny["reason"])

        # Sending a simple read-only follow-up to flash_lite is allowed
        read_payload = dict(
            self.base_payload,
            toolCall={
                "name": "send_message",
                "args": {
                    "Recipient": scout_cid,
                    "Message": "Read line 40 of `src/auth/parser.py`.",
                },
            },
        )
        res_allow = orchestrator_guard.handle_pre_tool(read_payload)
        self.assertEqual(res_allow["decision"], "allow")

    def test_stop_hook_blocks_uncleaned_branch_until_killed(self) -> None:
        branch_cid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        orchestrator_guard.register_subagent(
            self.base_payload,
            conversation_id=branch_cid,
            type_name="implementer",
            model="flash",
            workspace="branch",
            role="Branch Worker",
        )

        # When fullyIdle is False, Stop is allowed so reactive wakeup works
        not_idle_payload = dict(
            self.base_payload,
            terminationReason="NO_TOOL_CALL",
            fullyIdle=False,
        )
        self.assertEqual(
            orchestrator_guard.handle_stop(not_idle_payload)["decision"], "allow"
        )

        # When fullyIdle is True and branch is still active, Stop returns continue
        idle_payload = dict(
            self.base_payload,
            terminationReason="NO_TOOL_CALL",
            fullyIdle=True,
        )
        res_stop = orchestrator_guard.handle_stop(idle_payload)
        self.assertEqual(res_stop["decision"], "continue")
        self.assertIn(branch_cid, res_stop["reason"])

        # Killing the branch clears the gate
        kill_payload = dict(
            self.base_payload,
            toolCall={
                "name": "manage_subagents",
                "args": {"Action": "kill", "ConversationIds": [branch_cid]},
            },
        )
        orchestrator_guard.handle_pre_tool(kill_payload)

        idle_payload_2 = dict(idle_payload, executionNum=2)
        res_after_kill = orchestrator_guard.handle_stop(idle_payload_2)
        self.assertEqual(res_after_kill["decision"], "allow")

    def test_workspace_auto_allow_vs_external_paths(self) -> None:
        # Command inside project root is auto-allowed with permissionOverrides
        inside_cmd = dict(
            self.base_payload,
            toolCall={
                "name": "run_command",
                "args": {
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                    "CommandLine": "python3 -m unittest discover -s tests",
                },
            },
        )
        res_in = orchestrator_guard.handle_pre_tool(inside_cmd)
        self.assertEqual(res_in["decision"], "allow")
        self.assertIn("permissionOverrides", res_in)

        # Command referencing an outside path falls back to ask
        outside_cmd = dict(
            self.base_payload,
            toolCall={
                "name": "run_command",
                "args": {
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                    "CommandLine": "cat /etc/shadow",
                },
            },
        )
        res_out = orchestrator_guard.handle_pre_tool(outside_cmd)
        self.assertEqual(res_out["decision"], "ask")

        # Relative path traversal outside workspace falls back to ask
        rel_outside_cmd = dict(
            self.base_payload,
            toolCall={
                "name": "run_command",
                "args": {
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                    "CommandLine": "cat ../outside.txt",
                },
            },
        )
        res_rel = orchestrator_guard.handle_pre_tool(rel_outside_cmd)
        self.assertEqual(res_rel["decision"], "ask")

    def test_post_tool_rolls_back_failed_branch_spawn(self) -> None:
        spawn_payload = dict(
            self.base_payload,
            toolCall={
                "name": "invoke_subagent",
                "args": {
                    "Subagents": [
                        {
                            "TypeName": "implementer",
                            "Role": "Failed Branch Worker",
                            "Model": "flash",
                            "Workspace": "branch",
                            "Prompt": VALID_PROMPT,
                        }
                    ]
                },
            },
        )
        res_pre = orchestrator_guard.handle_pre_tool(spawn_payload)
        self.assertEqual(res_pre["decision"], "allow")

        # Simulate runtime failure in PostToolUse
        post_error_payload = dict(
            spawn_payload,
            error="failed to create git worktree",
        )
        orchestrator_guard.handle_post_tool(post_error_payload)

        # Stop hook should now allow termination because the failed branch was rolled back
        idle_payload = dict(
            self.base_payload,
            terminationReason="NO_TOOL_CALL",
            fullyIdle=True,
        )
        self.assertEqual(
            orchestrator_guard.handle_stop(idle_payload)["decision"], "allow"
        )

    def test_transcript_sync_ignores_non_subagent_uuids(self) -> None:
        transcript_file = Path(self.artifact_dir) / "transcript.jsonl"
        payload = dict(
            self.base_payload,
            transcriptPath=str(transcript_file),
            toolCall={
                "name": "invoke_subagent",
                "args": {
                    "Subagents": [
                        {
                            "TypeName": "scout",
                            "Role": "Symbol Scout",
                            "Model": "flash_lite",
                            "Prompt": VALID_PROMPT,
                        }
                    ]
                },
            },
        )
        orchestrator_guard.handle_pre_tool(payload)

        # Write transcript lines with invoke_subagent and non-subagent UUIDs
        non_subagent_step_uuid = "11111111-1111-1111-1111-111111111111"
        non_subagent_task_uuid = "22222222-2222-2222-2222-222222222222"
        actual_subagent_cid = "33333333-3333-3333-3333-333333333333"

        # Line 1 has invoke_subagent and a step UUID but no conversationId key.
        # Line 2 has stepId and callId UUIDs alongside the real conversationId key.
        transcript_file.write_text(
            json.dumps({
                "stepId": non_subagent_step_uuid,
                "message": "Invoked tool invoke_subagent",
            })
            + "\n"
            + json.dumps({
                "stepId": "44444444-4444-4444-4444-444444444444",
                "toolCallId": non_subagent_task_uuid,
                "name": "invoke_subagent",
                "result": {
                    "conversationId": actual_subagent_cid,
                },
            })
            + "\n",
            encoding="utf-8",
        )

        orchestrator_guard.handle_post_tool(payload)

        state = orchestrator_guard._load_state(payload)
        subagents = state.get("subagents", {})
        self.assertIn(actual_subagent_cid, subagents)
        self.assertNotIn(non_subagent_step_uuid, subagents)
        self.assertNotIn(non_subagent_task_uuid, subagents)
        self.assertNotIn("44444444-4444-4444-4444-444444444444", subagents)
        self.assertEqual(len(state.get("pending_spawns", [])), 0)
        self.assertEqual(subagents[actual_subagent_cid]["model"], "flash_lite")

    def test_auto_downgrade_simple_readonly_research_to_flash_lite(self) -> None:
        simple_read_prompt = """TASK: Read `src/auth/parser.py` to check token constants.

RULES:
1. Open `src/auth/parser.py`.
2. Read the constant definitions at the top of the file.
3. Do not change any file.

REPORT FORMAT:
Return only these four lines:
STATUS: COMPLETED | BLOCKED | FAILED
FILES_CHANGED: NONE
VERIFICATION: Read `src/auth/parser.py`.
NOTES: <one sentence stating the constant values>
"""
        multi_step_read_prompt = """TASK: Audit authentication flow across four modules.

RULES:
1. Open `src/auth/parser.py`.
2. Open `src/auth/session.py`.
3. Open `src/auth/middleware.py`.
4. Compare token expiration checks across all three modules.

REPORT FORMAT:
Return only these four lines:
STATUS: COMPLETED | BLOCKED | FAILED
FILES_CHANGED: NONE
VERIFICATION: Inspected all three authentication modules.
NOTES: <one sentence summarizing findings>
"""
        payload = dict(
            self.base_payload,
            toolCall={
                "id": "call_downgrade_test",
                "name": "invoke_subagent",
                "args": {
                    "Subagents": [
                        {
                            "TypeName": "research",
                            "Role": "Quick Reader",
                            "Model": "flash",
                            "Prompt": simple_read_prompt,
                        },
                        {
                            "TypeName": "research",
                            "Role": "Deep Researcher",
                            "Model": "flash",
                            "Prompt": multi_step_read_prompt,
                        },
                    ]
                },
            },
        )
        res = orchestrator_guard.handle_pre_tool(payload)
        self.assertEqual(res["decision"], "allow")
        rewritten = res["overwrite"]["Subagents"]
        self.assertEqual(rewritten[0]["Model"], "flash_lite")
        self.assertEqual(rewritten[1]["Model"], "flash")

    def test_pending_spawns_deduplicated_across_duplicate_pre_tool_hooks(self) -> None:
        payload = dict(
            self.base_payload,
            toolCall={
                "id": "call_dedup_1",
                "name": "invoke_subagent",
                "args": {
                    "Subagents": [
                        {
                            "TypeName": "scout",
                            "Role": "Dedup Scout",
                            "Model": "flash_lite",
                            "Prompt": VALID_PROMPT,
                        }
                    ]
                },
            },
        )
        orchestrator_guard.handle_pre_tool(payload)
        orchestrator_guard.handle_pre_tool(payload)
        orchestrator_guard.handle_pre_tool(payload)

        state = orchestrator_guard._load_state(payload)
        self.assertEqual(len(state.get("pending_spawns", [])), 1)

    def test_worktree_and_sibling_brain_paths_auto_allowed(self) -> None:
        app_data = Path(self.artifact_dir) / "appdata"
        subagent_brain = app_data / "brain" / "subagent-uuid"
        parent_scratch = app_data / "brain" / "parent-uuid" / "scratch" / "task.log"
        worktree_file = app_data / "worktrees" / "parent-uuid" / "subagent-worktree" / "README.md"
        subagent_brain.mkdir(parents=True, exist_ok=True)
        parent_scratch.parent.mkdir(parents=True, exist_ok=True)
        worktree_file.parent.mkdir(parents=True, exist_ok=True)

        payload_scratch = dict(
            self.base_payload,
            artifactDirectoryPath=str(subagent_brain),
            toolCall={
                "name": "write_to_file",
                "args": {"TargetFile": str(parent_scratch)},
            },
        )
        self.assertEqual(
            orchestrator_guard.handle_pre_tool(payload_scratch)["decision"], "allow"
        )

        payload_worktree = dict(
            self.base_payload,
            artifactDirectoryPath=str(subagent_brain),
            toolCall={
                "name": "write_to_file",
                "args": {"TargetFile": str(worktree_file)},
            },
        )
        self.assertEqual(
            orchestrator_guard.handle_pre_tool(payload_worktree)["decision"], "allow"
        )

    def test_direct_mutation_budget_blocks_third_workspace_file(self) -> None:
        file1 = str(orchestrator_guard.PROJECT_ROOT / "file1.py")
        file2 = str(orchestrator_guard.PROJECT_ROOT / "file2.py")
        file3 = str(orchestrator_guard.PROJECT_ROOT / "file3.py")

        payload1 = dict(
            self.base_payload,
            toolCall={
                "name": "write_to_file",
                "args": {"TargetFile": file1},
            },
        )
        self.assertEqual(orchestrator_guard.handle_pre_tool(payload1)["decision"], "allow")

        payload1_repeat = dict(
            self.base_payload,
            toolCall={
                "name": "replace_file_content",
                "args": {"TargetFile": file1},
            },
        )
        self.assertEqual(
            orchestrator_guard.handle_pre_tool(payload1_repeat)["decision"], "allow"
        )

        payload2 = dict(
            self.base_payload,
            toolCall={
                "name": "write_to_file",
                "args": {"TargetFile": file2},
            },
        )
        self.assertEqual(orchestrator_guard.handle_pre_tool(payload2)["decision"], "allow")

        payload2_repeat = dict(
            self.base_payload,
            toolCall={
                "name": "replace_file_content",
                "args": {"TargetFile": file2},
            },
        )
        self.assertEqual(
            orchestrator_guard.handle_pre_tool(payload2_repeat)["decision"], "allow"
        )

        payload3 = dict(
            self.base_payload,
            toolCall={
                "name": "write_to_file",
                "args": {"TargetFile": file3},
            },
        )
        res3 = orchestrator_guard.handle_pre_tool(payload3)
        self.assertEqual(res3["decision"], "deny")
        self.assertIn("[ORCHESTRATOR GUARD]", res3["reason"])

        subagent_cid = "sub-mutation-cid-1111"
        orchestrator_guard.register_subagent(
            self.base_payload,
            conversation_id=subagent_cid,
            type_name="implementer",
            model="flash",
        )
        subagent_payload = dict(
            self.base_payload,
            conversationId=subagent_cid,
            modelName="gemini-3-flash",
            toolCall={
                "name": "write_to_file",
                "args": {"TargetFile": file3},
            },
        )
        res_sub = orchestrator_guard.handle_pre_tool(subagent_payload)
        self.assertEqual(res_sub["decision"], "allow")

    def test_direct_edit_then_test_command_blocked_for_orchestrator(self) -> None:
        file1 = str(orchestrator_guard.PROJECT_ROOT / "file1.py")
        payload_write = dict(
            self.base_payload,
            toolCall={
                "name": "write_to_file",
                "args": {"TargetFile": file1},
            },
        )
        self.assertEqual(
            orchestrator_guard.handle_pre_tool(payload_write)["decision"], "allow"
        )

        git_payload = dict(
            self.base_payload,
            toolCall={
                "name": "run_command",
                "args": {
                    "CommandLine": "git status",
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                },
            },
        )
        self.assertEqual(
            orchestrator_guard.handle_pre_tool(git_payload)["decision"], "allow"
        )

        test_payload = dict(
            self.base_payload,
            toolCall={
                "name": "run_command",
                "args": {
                    "CommandLine": "python3 -m unittest discover -s tests",
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                },
            },
        )
        res_test = orchestrator_guard.handle_pre_tool(test_payload)
        self.assertEqual(res_test["decision"], "deny")
        self.assertIn("[ORCHESTRATOR GUARD]", res_test["reason"])

    def test_direct_command_budget_blocks_third_command_and_deduplicates(self) -> None:
        payload_cmd1 = dict(
            self.base_payload,
            toolCall={
                "id": "cmd_call_1",
                "name": "run_command",
                "args": {
                    "CommandLine": "ls -la",
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                },
            },
        )
        self.assertEqual(
            orchestrator_guard.handle_pre_tool(payload_cmd1)["decision"], "allow"
        )
        self.assertEqual(
            orchestrator_guard.handle_pre_tool(payload_cmd1)["decision"], "allow"
        )

        state = orchestrator_guard._load_state(payload_cmd1)
        self.assertEqual(len(state["direct_budget"]["1"]["commands"]), 1)

        payload_cmd2 = dict(
            self.base_payload,
            toolCall={
                "id": "cmd_call_2",
                "name": "run_command",
                "args": {
                    "CommandLine": "python3 --version",
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                },
            },
        )
        self.assertEqual(
            orchestrator_guard.handle_pre_tool(payload_cmd2)["decision"], "allow"
        )

        payload_cmd3 = dict(
            self.base_payload,
            toolCall={
                "id": "cmd_call_3",
                "name": "run_command",
                "args": {
                    "CommandLine": "echo hello",
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                },
            },
        )
        res_cmd3 = orchestrator_guard.handle_pre_tool(payload_cmd3)
        self.assertEqual(res_cmd3["decision"], "deny")
        self.assertIn("[ORCHESTRATOR GUARD]", res_cmd3["reason"])

        payload_next_inv = dict(
            self.base_payload,
            invocationNum=2,
            toolCall={
                "id": "cmd_call_4",
                "name": "run_command",
                "args": {
                    "CommandLine": "echo hello",
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                },
            },
        )
        self.assertEqual(
            orchestrator_guard.handle_pre_tool(payload_next_inv)["decision"], "allow"
        )

    def test_empirical_test_receipt_required_before_stop(self) -> None:
        code_file = str(orchestrator_guard.PROJECT_ROOT / "src" / "app.py")
        edit_payload = dict(
            self.base_payload,
            toolCall={
                "name": "write_to_file",
                "args": {"TargetFile": code_file, "CodeContent": "print('hello')"},
            },
        )
        self.assertEqual(
            orchestrator_guard.handle_pre_tool(edit_payload)["decision"], "allow"
        )

        stop_payload_1 = dict(
            self.base_payload,
            fullyIdle=True,
            terminationReason="NO_TOOL_CALL",
            executionNum=1,
        )
        res_stop_1 = orchestrator_guard.handle_stop(stop_payload_1)
        self.assertEqual(res_stop_1["decision"], "continue")
        self.assertIn("Empirical test receipt missing", res_stop_1["reason"])

        # Failing test run
        fail_test_payload = dict(
            self.base_payload,
            toolCall={
                "id": "test_cmd_1",
                "name": "run_command",
                "args": {
                    "CommandLine": "pytest tests/",
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                },
            },
            error="exit status 1",
        )
        orchestrator_guard.handle_post_tool(fail_test_payload)

        stop_payload_2 = dict(
            self.base_payload,
            fullyIdle=True,
            terminationReason="NO_TOOL_CALL",
            executionNum=2,
        )
        res_stop_2 = orchestrator_guard.handle_stop(stop_payload_2)
        self.assertEqual(res_stop_2["decision"], "continue")

        # Passing test run
        pass_test_payload = dict(
            self.base_payload,
            toolCall={
                "id": "test_cmd_2",
                "name": "run_command",
                "args": {
                    "CommandLine": "pytest tests/",
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                },
            },
            error="",
        )
        orchestrator_guard.handle_post_tool(pass_test_payload)

        stop_payload_3 = dict(
            self.base_payload,
            fullyIdle=True,
            terminationReason="NO_TOOL_CALL",
            executionNum=3,
        )
        res_stop_3 = orchestrator_guard.handle_stop(stop_payload_3)
        self.assertEqual(res_stop_3["decision"], "allow")

    def test_two_turn_circuit_breaker_halts_retries_and_allows_scratch_log(self) -> None:
        fail_1 = dict(
            self.base_payload,
            toolCall={
                "id": "cmd_fail_1",
                "name": "run_command",
                "args": {
                    "CommandLine": "pytest tests/",
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                },
            },
            error="exit status 1",
        )
        orchestrator_guard.handle_post_tool(fail_1)

        fail_2 = dict(
            self.base_payload,
            toolCall={
                "id": "cmd_fail_2",
                "name": "run_command",
                "args": {
                    "CommandLine": "pytest tests/",
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                },
            },
            error="exit status 1",
        )
        orchestrator_guard.handle_post_tool(fail_2)

        # 3rd run_command (pytest) denied
        cmd_3 = dict(
            self.base_payload,
            toolCall={
                "id": "cmd_fail_3",
                "name": "run_command",
                "args": {
                    "CommandLine": "pytest tests/",
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                },
            },
        )
        res_cmd_3 = orchestrator_guard.handle_pre_tool(cmd_3)
        self.assertEqual(res_cmd_3["decision"], "deny")
        self.assertIn("2-turn circuit breaker tripped", res_cmd_3["reason"])

        # Workspace code edit denied
        edit_code = dict(
            self.base_payload,
            toolCall={
                "name": "write_to_file",
                "args": {
                    "TargetFile": str(orchestrator_guard.PROJECT_ROOT / "src" / "app.py"),
                    "CodeContent": "print('retry')",
                },
            },
        )
        res_edit = orchestrator_guard.handle_pre_tool(edit_code)
        self.assertEqual(res_edit["decision"], "deny")
        self.assertIn("2-turn circuit breaker tripped", res_edit["reason"])

        # Writing failure output to scratch log is allowed
        scratch_log = str(Path(self.artifact_dir) / "scratch" / "fail.log")
        scratch_write = dict(
            self.base_payload,
            toolCall={
                "name": "write_to_file",
                "args": {
                    "TargetFile": scratch_log,
                    "CodeContent": "Traceback: test failed twice",
                },
            },
        )
        res_scratch = orchestrator_guard.handle_pre_tool(scratch_write)
        self.assertEqual(res_scratch["decision"], "allow")

        # handle_stop (fullyIdle=True) returns allow so agent can report STATUS: BLOCKED
        stop_payload = dict(
            self.base_payload,
            fullyIdle=True,
            terminationReason="NO_TOOL_CALL",
        )
        res_stop = orchestrator_guard.handle_stop(stop_payload)
        self.assertEqual(res_stop["decision"], "allow")

    def test_pre_tool_blocks_test_tampering_edits(self) -> None:
        test_file = str(orchestrator_guard.PROJECT_ROOT / "tests" / "test_sample.py")

        # 1. Deleting an assertion is denied
        payload_del_assert = dict(
            self.base_payload,
            toolCall={
                "name": "replace_file_content",
                "args": {
                    "TargetFile": test_file,
                    "TargetContent": "        self.assertEqual(a, b)\n",
                    "ReplacementContent": "",
                    "StartLine": 10,
                    "EndLine": 11,
                },
            },
        )
        res_del_assert = orchestrator_guard.handle_pre_tool(payload_del_assert)
        self.assertEqual(res_del_assert["decision"], "deny")
        self.assertIn("reduces assertion count", res_del_assert["reason"])

        # 2. Deleting a test function is denied
        payload_del_test = dict(
            self.base_payload,
            toolCall={
                "name": "replace_file_content",
                "args": {
                    "TargetFile": test_file,
                    "TargetContent": "    def test_login(self):\n        pass\n",
                    "ReplacementContent": "",
                    "StartLine": 5,
                    "EndLine": 7,
                },
            },
        )
        res_del_test = orchestrator_guard.handle_pre_tool(payload_del_test)
        self.assertEqual(res_del_test["decision"], "deny")
        self.assertIn("reduces test count", res_del_test["reason"])

        # 3. Adding a skip decorator is denied (use string concatenation so test file itself doesn't trigger skip regex)
        skip_marker = "@" + "unittest.skip('skip')"
        payload_add_skip = dict(
            self.base_payload,
            toolCall={
                "name": "replace_file_content",
                "args": {
                    "TargetFile": test_file,
                    "TargetContent": "    def test_login(self):",
                    "ReplacementContent": f"    {skip_marker}\n    def test_login(self):",
                    "StartLine": 5,
                    "EndLine": 6,
                },
            },
        )
        res_add_skip = orchestrator_guard.handle_pre_tool(payload_add_skip)
        self.assertEqual(res_add_skip["decision"], "deny")
        self.assertIn("adding skip/xfail decorators", res_add_skip["reason"])

        # 4. Adding trivial assert True is denied (use string concatenation)
        trivial_assert = "assert " + "True"
        payload_trivial = dict(
            self.base_payload,
            toolCall={
                "name": "replace_file_content",
                "args": {
                    "TargetFile": test_file,
                    "TargetContent": "        self.assertEqual(a, b)",
                    "ReplacementContent": f"        {trivial_assert}",
                    "StartLine": 10,
                    "EndLine": 11,
                },
            },
        )
        res_trivial = orchestrator_guard.handle_pre_tool(payload_trivial)
        self.assertEqual(res_trivial["decision"], "deny")
        self.assertIn("trivial 'assert True'", res_trivial["reason"])

        # 5. Adding new test function and new assertions is allowed
        payload_add_test = dict(
            self.base_payload,
            toolCall={
                "name": "replace_file_content",
                "args": {
                    "TargetFile": test_file,
                    "TargetContent": "    def test_existing(self):\n        self.assertEqual(1, 1)",
                    "ReplacementContent": (
                        "    def test_existing(self):\n"
                        "        self.assertEqual(1, 1)\n\n"
                        "    def test_new(self):\n"
                        "        self.assertEqual(2, 2)"
                    ),
                    "StartLine": 15,
                    "EndLine": 17,
                },
            },
        )
        res_add_test = orchestrator_guard.handle_pre_tool(payload_add_test)
        self.assertEqual(res_add_test["decision"], "allow")

    def test_audit_diff_text_detects_test_tampering(self) -> None:
        # 1. Removing an assertion in tests/test_auth.py returns error string
        diff_rem_assert = (
            "--- a/tests/test_auth.py\n"
            "+++ b/tests/test_auth.py\n"
            "@@ -10,3 +10,2 @@\n"
            "-    self.assertEqual(token, 'abc')\n"
            "     return True\n"
        )
        res_rem = orchestrator_guard.audit_diff_text(diff_rem_assert)
        self.assertIsNotNone(res_rem)
        self.assertIn("Diff audit detected net removal of 1 assertion(s)", res_rem)

        # 2. Adding skip marker in tests/test_auth.py returns error string
        skip_line = "+@" + "pytest.mark.skip"
        diff_add_skip = (
            "--- a/tests/test_auth.py\n"
            "+++ b/tests/test_auth.py\n"
            "@@ -5,2 +5,3 @@\n"
            f"{skip_line}\n"
            " def test_login():\n"
            "     pass\n"
        )
        res_skip = orchestrator_guard.audit_diff_text(diff_add_skip)
        self.assertIsNotNone(res_skip)
        self.assertIn("Diff audit detected test skip/xfail marker added", res_skip)

        # 3. Adding a new test + assertion in tests/test_auth.py returns None
        diff_add_test = (
            "--- a/tests/test_auth.py\n"
            "+++ b/tests/test_auth.py\n"
            "@@ -20,2 +20,6 @@\n"
            "+def test_new_feature():\n"
            "+    self.assertEqual(1, 1)\n"
            " def test_existing():\n"
        )
        res_good = orchestrator_guard.audit_diff_text(diff_add_test)
        self.assertIsNone(res_good)

    def test_subagent_pre_invocation_injects_isolation_reminder_instead_of_tier_warning(
        self,
    ) -> None:
        orchestrator_guard.register_subagent(
            self.base_payload,
            conversation_id="sub-1111",
            type_name="scout",
            model="flash_lite",
        )
        payload = dict(
            self.base_payload,
            conversationId="sub-1111",
            modelName="gemini-3-flash-lite",
            invocationNum=1,
        )
        res = orchestrator_guard.handle_pre_invocation(payload)
        self.assertIn("injectSteps", res)
        msg = res["injectSteps"][0]["ephemeralMessage"]
        self.assertIn("Subagent context isolation active", msg)
        self.assertIn("REPORT FORMAT", msg)

    def test_subagent_stop_enforces_four_line_report_format(self) -> None:
        sub_cid = "sub-stop-report-uuid-3333"
        orchestrator_guard.register_subagent(
            self.base_payload,
            conversation_id=sub_cid,
            type_name="implementer",
            model="flash",
        )
        payload_bad = dict(
            self.base_payload,
            conversationId=sub_cid,
            modelName="gemini-3-flash",
            fullyIdle=True,
            terminationReason="NO_TOOL_CALL",
            executionNum=1,
            lastAssistantMessage="Here is my long verbose summary:\nLine 2\nLine 3\nLine 4\nLine 5",
        )
        res_bad = orchestrator_guard.handle_stop(payload_bad)
        self.assertEqual(res_bad["decision"], "continue")
        self.assertIn("Subagent context-separation violation", res_bad["reason"])

        payload_good = dict(
            self.base_payload,
            conversationId=sub_cid,
            modelName="gemini-3-flash",
            fullyIdle=True,
            terminationReason="NO_TOOL_CALL",
            executionNum=2,
            lastAssistantMessage=(
                "STATUS: COMPLETED\n"
                "FILES_CHANGED: NONE\n"
                "VERIFICATION: Ran unittest with exit code 0.\n"
                "NOTES: All checks passed."
            ),
        )
        res_good = orchestrator_guard.handle_stop(payload_good)
        self.assertEqual(res_good["decision"], "allow")

        # Root orchestrator with omitted modelName in Stop hook should be allowed
        orchestrator_guard.handle_pre_invocation(self.base_payload)
        payload_root_stop = dict(
            self.base_payload,
            modelName="",
            fullyIdle=True,
            terminationReason="NO_TOOL_CALL",
            executionNum=3,
            lastAssistantMessage=(
                "Summary of work completed:\n\n"
                "First, we investigated the root cause of the bug.\n"
                "Second, we applied the fix across the affected components.\n\n"
                "Finally, all tests were verified successfully."
            ),
        )
        res_root = orchestrator_guard.handle_stop(payload_root_stop)
        self.assertEqual(res_root["decision"], "allow")

    def test_validate_subagent_report_allows_bounded_bullets_and_enforces_blocked_diagnosis(
        self,
    ) -> None:
        report_bullets = (
            "STATUS: COMPLETED\n"
            "FILES_CHANGED: src/app.py\n"
            "VERIFICATION: python3 -m unittest discover -s tests -v exited with 0.\n"
            "NOTES: All unit tests passed.\n"
            "- Added input validation check.\n"
            "- Updated docstrings.\n"
            "- Added unit tests in test_app.py."
        )
        self.assertIsNone(orchestrator_guard.validate_subagent_report(report_bullets))
        sub_cid = "sub-bullet-report-uuid-4444"
        orchestrator_guard.register_subagent(
            self.base_payload,
            conversation_id=sub_cid,
            type_name="implementer",
            model="flash",
        )
        payload_sub_stop = dict(
            self.base_payload,
            conversationId=sub_cid,
            modelName="gemini-3-flash",
            fullyIdle=True,
            terminationReason="NO_TOOL_CALL",
            executionNum=10,
            lastAssistantMessage=report_bullets,
        )
        res_stop = orchestrator_guard.handle_stop(payload_sub_stop)
        self.assertEqual(res_stop["decision"], "allow")

        report_non_bullet = (
            "STATUS: COMPLETED\n"
            "FILES_CHANGED: NONE\n"
            "VERIFICATION: Ran unittest with exit 0.\n"
            "NOTES: Everything finished.\n"
            "This is a non-bullet line."
        )
        err_non_bullet = orchestrator_guard.validate_subagent_report(report_non_bullet)
        self.assertIsNotNone(err_non_bullet)
        self.assertIn("must be a compact bullet", err_non_bullet)

        eleven_bullets = "\n".join(f"- bullet line {i}" for i in range(1, 12))
        report_too_many_bullets = (
            "STATUS: COMPLETED\n"
            "FILES_CHANGED: NONE\n"
            "VERIFICATION: Ran unittest with exit 0.\n"
            f"NOTES: Summary with 11 bullets.\n{eleven_bullets}"
        )
        err_too_many = orchestrator_guard.validate_subagent_report(report_too_many_bullets)
        self.assertIsNotNone(err_too_many)
        self.assertIn("at most 10 bullet lines", err_too_many)

        report_no_summary = (
            "STATUS: BLOCKED\n"
            "FILES_CHANGED: NONE\n"
            "VERIFICATION: pytest tests/ failed with exit 1.\n"
            "NOTES: file:///tmp/scratch/fail.log"
        )
        err_no_summary = orchestrator_guard.validate_subagent_report(report_no_summary)
        self.assertIsNotNone(err_no_summary)
        self.assertIn("inline error summary", err_no_summary)

        report_no_uri = (
            "STATUS: BLOCKED\n"
            "FILES_CHANGED: NONE\n"
            "VERIFICATION: pytest tests/ failed with exit 1.\n"
            "NOTES: AssertionError in test_login on line 42"
        )
        err_no_uri = orchestrator_guard.validate_subagent_report(report_no_uri)
        self.assertIsNotNone(err_no_uri)
        self.assertIn("scratch log URI", err_no_uri)

        report_valid_blocked = (
            "STATUS: BLOCKED\n"
            "FILES_CHANGED: NONE\n"
            "VERIFICATION: pytest tests/ failed with exit 1.\n"
            "NOTES: AssertionError in test_login on line 42 at file:///tmp/scratch/fail.log"
        )
        self.assertIsNone(orchestrator_guard.validate_subagent_report(report_valid_blocked))

    def test_orchestrator_direct_read_budget_and_log_peeking_blocked(self) -> None:
        file_a = str(orchestrator_guard.PROJECT_ROOT / "file_a.py")
        file_b = str(orchestrator_guard.PROJECT_ROOT / "file_b.py")
        file_c = str(orchestrator_guard.PROJECT_ROOT / "file_c.py")

        def make_read_payload(
            path: str,
            model: str = "Gemini 4 Argon (Medium)",
            cid: Optional[str] = None,
        ) -> Dict[str, Any]:
            p = dict(
                self.base_payload,
                modelName=model,
                toolCall={
                    "name": "view_file",
                    "args": {"AbsolutePath": path},
                },
            )
            if cid:
                p["conversationId"] = cid
            return p

        # 1. First 2 distinct workspace files are allowed, including repeated read of file_a
        res_a1 = orchestrator_guard.handle_pre_tool(make_read_payload(file_a))
        self.assertEqual(res_a1["decision"], "allow")

        res_b = orchestrator_guard.handle_pre_tool(make_read_payload(file_b))
        self.assertEqual(res_b["decision"], "allow")

        res_a2 = orchestrator_guard.handle_pre_tool(make_read_payload(file_a))
        self.assertEqual(res_a2["decision"], "allow")

        # 3rd distinct workspace file exceeds limit
        res_c = orchestrator_guard.handle_pre_tool(make_read_payload(file_c))
        self.assertEqual(res_c["decision"], "deny")
        self.assertIn("read limit (2 workspace files) exceeded", res_c["reason"])

        # 2. Reading raw log or transcript is denied for orchestrator
        fail_log = str(Path(self.artifact_dir) / "scratch" / "fail.log")
        transcript = str(Path(self.artifact_dir) / "transcript.jsonl")

        res_log = orchestrator_guard.handle_pre_tool(make_read_payload(fail_log))
        self.assertEqual(res_log["decision"], "deny")
        self.assertIn("cannot directly read raw log or transcript", res_log["reason"])

        res_tr = orchestrator_guard.handle_pre_tool(make_read_payload(transcript))
        self.assertEqual(res_tr["decision"], "deny")
        self.assertIn("cannot directly read raw log or transcript", res_tr["reason"])

        # 3. Scout subagent is allowed to read both fail.log and file_c.py
        scout_cid = "sub-scout-read-cid-4444"
        orchestrator_guard.register_subagent(
            self.base_payload,
            conversation_id=scout_cid,
            type_name="scout",
            model="flash_lite",
        )
        scout_log = orchestrator_guard.handle_pre_tool(
            make_read_payload(fail_log, model="gemini-3-flash-lite", cid=scout_cid)
        )
        self.assertEqual(scout_log["decision"], "allow")

        scout_c = orchestrator_guard.handle_pre_tool(
            make_read_payload(file_c, model="gemini-3-flash-lite", cid=scout_cid)
        )
        self.assertEqual(scout_c["decision"], "allow")

    def test_small_model_root_orchestrator_is_treated_as_root(self) -> None:
        payload = dict(self.base_payload, modelName="gemini-3-flash-lite")
        res_pre = orchestrator_guard.handle_pre_invocation(payload)
        self.assertEqual(res_pre, {})

        f1 = str(orchestrator_guard.PROJECT_ROOT / "root_f1.txt")
        f2 = str(orchestrator_guard.PROJECT_ROOT / "root_f2.txt")
        f3 = str(orchestrator_guard.PROJECT_ROOT / "root_f3.txt")

        p1 = dict(payload, toolCall={"name": "write_to_file", "args": {"TargetFile": f1}})
        self.assertEqual(orchestrator_guard.handle_pre_tool(p1)["decision"], "allow")

        p2 = dict(payload, toolCall={"name": "write_to_file", "args": {"TargetFile": f2}})
        self.assertEqual(orchestrator_guard.handle_pre_tool(p2)["decision"], "allow")

        p3 = dict(payload, toolCall={"name": "write_to_file", "args": {"TargetFile": f3}})
        res3 = orchestrator_guard.handle_pre_tool(p3)
        self.assertEqual(res3["decision"], "deny")
        self.assertIn("[ORCHESTRATOR GUARD]", res3["reason"])

        stop_payload = dict(
            payload,
            fullyIdle=True,
            terminationReason="NO_TOOL_CALL",
            lastAssistantMessage="This is a long verbose summary without four-line report format.\nMore details here.\nAnd here.",
        )
        res_stop = orchestrator_guard.handle_stop(stop_payload)
        self.assertEqual(res_stop["decision"], "allow")


if __name__ == "__main__":
    unittest.main()
