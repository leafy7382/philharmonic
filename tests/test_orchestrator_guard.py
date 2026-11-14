#!/usr/bin/env python3
"""Unit tests for orchestrator_guard.py."""

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
    / "philharmonic_orchestra"
    / "scripts"
)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import orchestrator_guard  # noqa: E402


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

    def test_autonomous_git_diff_audit_in_handle_stop(self) -> None:
        tampered_diff = (
            "diff --git a/tests/test_example.py b/tests/test_example.py\n"
            "--- a/tests/test_example.py\n"
            "+++ b/tests/test_example.py\n"
            "@@ -10,1 +10,2 @@\n"
            "+@unittest.skip('temporarily skipping')\n"
            " def test_feature():\n"
        )
        payload = dict(
            self.base_payload,
            fullyIdle=True,
            terminationReason="NO_TOOL_CALL",
            gitDiff=tampered_diff,
        )
        res = orchestrator_guard.handle_stop(payload)
        self.assertEqual(res["decision"], "continue")
        self.assertIn("Diff audit detected test skip", res.get("reason", ""))

        clean_diff = (
            "diff --git a/tests/test_example.py b/tests/test_example.py\n"
            "--- a/tests/test_example.py\n"
            "+++ b/tests/test_example.py\n"
            "@@ -10,0 +10,3 @@\n"
            "+def test_new_behavior():\n"
            "+    assert 1 + 1 == 2\n"
        )
        payload_clean = dict(
            self.base_payload,
            executionNum=2,
            fullyIdle=True,
            terminationReason="NO_TOOL_CALL",
            gitDiff=clean_diff,
        )
        res_clean = orchestrator_guard.handle_stop(payload_clean)
        self.assertEqual(res_clean["decision"], "allow")
        self.assertNotIn("reason", res_clean)

    def test_isolation_reminder_deduplication(self) -> None:
        sub_cid = "sub-dedup-worker-101"
        orchestrator_guard.register_subagent(
            self.base_payload,
            conversation_id=sub_cid,
            type_name="implementer",
            model="flash",
        )
        payload_1 = dict(
            self.base_payload,
            conversationId=sub_cid,
            invocationNum=1,
        )
        res_1 = orchestrator_guard.handle_pre_invocation(payload_1)
        self.assertIn("injectSteps", res_1)
        self.assertIn(
            "Subagent context isolation active",
            res_1["injectSteps"][0]["ephemeralMessage"],
        )

        res_2 = orchestrator_guard.handle_pre_invocation(payload_1)
        self.assertEqual(res_2, {})

        payload_2 = dict(
            self.base_payload,
            conversationId=sub_cid,
            invocationNum=2,
        )
        res_3 = orchestrator_guard.handle_pre_invocation(payload_2)
        self.assertIn("injectSteps", res_3)
        self.assertIn(
            "Subagent context isolation active",
            res_3["injectSteps"][0]["ephemeralMessage"],
        )

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
                    "Name": "scout",
                    "EnableSubagentTools": True,
                },
            },
        )
        res = orchestrator_guard.handle_pre_tool(payload)
        self.assertEqual(res["decision"], "allow")
        self.assertEqual(res["overwrite"], {"EnableSubagentTools": False})

        # Research subagent with EnableSubagentTools: True is allowed with EnableSubagentTools: False
        payload_research = dict(
            self.base_payload,
            toolCall={
                "name": "define_subagent",
                "args": {
                    "Name": "research",
                    "EnableWriteTools": True,
                    "EnableSubagentTools": True,
                },
            },
        )
        res_research = orchestrator_guard.handle_pre_tool(payload_research)
        self.assertEqual(res_research["decision"], "allow")
        self.assertEqual(res_research["overwrite"], {"EnableSubagentTools": False})

    def test_invoke_subagent_blocks_pro_model(self) -> None:
        payload = dict(
            self.base_payload,
            toolCall={
                "name": "invoke_subagent",
                "args": {
                    "TypeName": "implementer",
                    "Role": "Auth Implementer",
                    "Model": "pro",
                    "Prompt": VALID_PROMPT,
                },
            },
        )
        res = orchestrator_guard.handle_pre_tool(payload)
        self.assertEqual(res["decision"], "deny")
        self.assertIn("Model='pro'", res["reason"])

    def test_invoke_subagent_rewrites_inherit_to_tier_defaults(self) -> None:
        payload_scout = dict(
            self.base_payload,
            toolCall={
                "name": "invoke_subagent",
                "args": {
                    "TypeName": "scout",
                    "Role": "Symbol Scout",
                    "Model": "inherit",
                    "Prompt": VALID_PROMPT,
                },
            },
        )
        res_scout = orchestrator_guard.handle_pre_tool(payload_scout)
        self.assertEqual(res_scout["decision"], "allow")
        self.assertEqual(res_scout["overwrite"]["Model"], "flash_lite")

        payload_impl = dict(
            self.base_payload,
            toolCall={
                "name": "invoke_subagent",
                "args": {
                    "TypeName": "implementer",
                    "Role": "Code Worker",
                    "Prompt": VALID_PROMPT,
                },
            },
        )
        res_impl = orchestrator_guard.handle_pre_tool(payload_impl)
        self.assertEqual(res_impl["decision"], "allow")
        self.assertEqual(res_impl["overwrite"]["Model"], "flash")

    def test_invoke_subagent_and_send_message_reject_empty_prompts(self) -> None:
        # Empty Prompt in invoke_subagent is blocked
        empty_prompt_payload = dict(
            self.base_payload,
            toolCall={
                "name": "invoke_subagent",
                "args": {
                    "TypeName": "scout",
                    "Role": "Empty Scout",
                    "Model": "flash_lite",
                    "Prompt": "   ",
                },
            },
        )
        res_empty_prompt = orchestrator_guard.handle_pre_tool(empty_prompt_payload)
        self.assertEqual(res_empty_prompt["decision"], "deny")
        self.assertIn("non-empty 'Prompt'", res_empty_prompt["reason"])

        # Empty TypeName in invoke_subagent is blocked
        empty_type_payload = dict(
            self.base_payload,
            toolCall={
                "name": "invoke_subagent",
                "args": {
                    "TypeName": "",
                    "Role": "No Type",
                    "Model": "flash_lite",
                    "Prompt": VALID_PROMPT,
                },
            },
        )
        res_empty_type = orchestrator_guard.handle_pre_tool(empty_type_payload)
        self.assertEqual(res_empty_type["decision"], "deny")
        self.assertIn("non-empty 'TypeName'", res_empty_type["reason"])

        # Empty Message in send_message is blocked
        empty_msg_payload = dict(
            self.base_payload,
            toolCall={
                "name": "send_message",
                "args": {
                    "Recipient": "worker-1",
                    "Message": "   ",
                },
            },
        )
        res_empty_msg = orchestrator_guard.handle_pre_tool(empty_msg_payload)
        self.assertEqual(res_empty_msg["decision"], "deny")
        self.assertIn("Message cannot be empty", res_empty_msg["reason"])

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
        overrides = res_in["permissionOverrides"]
        self.assertTrue(any(o.startswith("file(") for o in overrides))

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
                    "TypeName": "implementer",
                    "Role": "Failed Branch Worker",
                    "Model": "flash",
                    "Workspace": "branch",
                    "Prompt": VALID_PROMPT,
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
                    "TypeName": "scout",
                    "Role": "Symbol Scout",
                    "Model": "flash_lite",
                    "Prompt": VALID_PROMPT,
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
        payload_simple = dict(
            self.base_payload,
            toolCall={
                "id": "call_downgrade_1",
                "name": "invoke_subagent",
                "args": {
                    "TypeName": "research",
                    "Role": "Quick Reader",
                    "Model": "flash",
                    "Prompt": simple_read_prompt,
                },
            },
        )
        res_simple = orchestrator_guard.handle_pre_tool(payload_simple)
        self.assertEqual(res_simple["decision"], "allow")
        self.assertEqual(res_simple["overwrite"]["Model"], "flash_lite")

        payload_deep = dict(
            self.base_payload,
            toolCall={
                "id": "call_downgrade_2",
                "name": "invoke_subagent",
                "args": {
                    "TypeName": "research",
                    "Role": "Deep Researcher",
                    "Model": "flash",
                    "Prompt": multi_step_read_prompt,
                },
            },
        )
        res_deep = orchestrator_guard.handle_pre_tool(payload_deep)
        self.assertEqual(res_deep["decision"], "allow")
        self.assertIsNone(res_deep.get("overwrite"))

    def test_pending_spawns_deduplicated_across_duplicate_pre_tool_hooks(self) -> None:
        payload = dict(
            self.base_payload,
            toolCall={
                "id": "call_dedup_1",
                "name": "invoke_subagent",
                "args": {
                    "TypeName": "scout",
                    "Role": "Dedup Scout",
                    "Model": "flash_lite",
                    "Prompt": VALID_PROMPT,
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
                "name": "write_file",
                "args": {"Path": str(parent_scratch)},
            },
        )
        self.assertEqual(
            orchestrator_guard.handle_pre_tool(payload_scratch)["decision"], "allow"
        )

        payload_worktree = dict(
            self.base_payload,
            artifactDirectoryPath=str(subagent_brain),
            toolCall={
                "name": "write_file",
                "args": {"Path": str(worktree_file)},
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
                "name": "write_file",
                "args": {"Path": file1},
            },
        )
        self.assertEqual(orchestrator_guard.handle_pre_tool(payload1)["decision"], "allow")

        payload1_repeat = dict(
            self.base_payload,
            toolCall={
                "name": "edit_file",
                "args": {"Path": file1},
            },
        )
        self.assertEqual(
            orchestrator_guard.handle_pre_tool(payload1_repeat)["decision"], "allow"
        )

        payload2 = dict(
            self.base_payload,
            toolCall={
                "name": "write_file",
                "args": {"Path": file2},
            },
        )
        self.assertEqual(orchestrator_guard.handle_pre_tool(payload2)["decision"], "allow")

        payload2_repeat = dict(
            self.base_payload,
            toolCall={
                "name": "edit_file",
                "args": {"Path": file2},
            },
        )
        self.assertEqual(
            orchestrator_guard.handle_pre_tool(payload2_repeat)["decision"], "allow"
        )

        payload3 = dict(
            self.base_payload,
            toolCall={
                "name": "write_file",
                "args": {"Path": file3},
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
                "name": "write_file",
                "args": {"Path": file3},
            },
        )
        res_sub = orchestrator_guard.handle_pre_tool(subagent_payload)
        self.assertEqual(res_sub["decision"], "allow")

    def test_direct_edit_then_test_command_blocked_for_orchestrator(self) -> None:
        file1 = str(orchestrator_guard.PROJECT_ROOT / "file1.py")
        payload_write = dict(
            self.base_payload,
            toolCall={
                "name": "write_file",
                "args": {"Path": file1},
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

        test_payload_1 = dict(
            self.base_payload,
            toolCall={
                "id": "test_1",
                "name": "run_command",
                "args": {
                    "CommandLine": "python3 -m unittest discover -s tests",
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                },
            },
        )
        res_test_1 = orchestrator_guard.handle_pre_tool(test_payload_1)
        self.assertEqual(res_test_1["decision"], "allow")

        test_payload_2 = dict(
            self.base_payload,
            toolCall={
                "id": "test_2",
                "name": "run_command",
                "args": {
                    "CommandLine": "python3 -m unittest discover -s tests",
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                },
            },
        )
        res_test_2 = orchestrator_guard.handle_pre_tool(test_payload_2)
        self.assertEqual(res_test_2["decision"], "deny")
        self.assertIn("[ORCHESTRATOR GUARD]", res_test_2["reason"])

        subsequent_write = dict(
            self.base_payload,
            toolCall={
                "name": "write_file",
                "args": {"Path": file1},
            },
        )
        res_subsequent_write = orchestrator_guard.handle_pre_tool(subsequent_write)
        self.assertEqual(res_subsequent_write["decision"], "deny")
        self.assertIn("[ORCHESTRATOR GUARD]", res_subsequent_write["reason"])

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
                "name": "write_file",
                "args": {"Path": code_file, "Content": "print('hello')"},
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
                "name": "write_file",
                "args": {
                    "Path": str(orchestrator_guard.PROJECT_ROOT / "src" / "app.py"),
                    "Content": "print('retry')",
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
                "name": "write_file",
                "args": {
                    "Path": scratch_log,
                    "Content": "Traceback: test failed twice",
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
                "name": "edit_file",
                "args": {
                    "Path": test_file,
                    "Target": "        self.assertEqual(a, b)\n",
                    "Replacement": "",
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
                "name": "edit_file",
                "args": {
                    "Path": test_file,
                    "Target": "    def test_login(self):\n        pass\n",
                    "Replacement": "",
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
                "name": "edit_file",
                "args": {
                    "Path": test_file,
                    "Target": "    def test_login(self):",
                    "Replacement": f"    {skip_marker}\n    def test_login(self):",
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
                "name": "edit_file",
                "args": {
                    "Path": test_file,
                    "Target": "        self.assertEqual(a, b)",
                    "Replacement": f"        {trivial_assert}",
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
                "name": "edit_file",
                "args": {
                    "Path": test_file,
                    "Target": "    def test_existing(self):\n        self.assertEqual(1, 1)",
                    "Replacement": (
                        "    def test_existing(self):\n"
                        "        self.assertEqual(1, 1)\n\n"
                        "    def test_new(self):\n"
                        "        self.assertEqual(2, 2)"
                    ),
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
                    "name": "read_file",
                    "args": {"Path": path},
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

        p1 = dict(payload, toolCall={"name": "write_file", "args": {"Path": f1}})
        self.assertEqual(orchestrator_guard.handle_pre_tool(p1)["decision"], "allow")

        p2 = dict(payload, toolCall={"name": "write_file", "args": {"Path": f2}})
        self.assertEqual(orchestrator_guard.handle_pre_tool(p2)["decision"], "allow")

        p3 = dict(payload, toolCall={"name": "write_file", "args": {"Path": f3}})
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

    def test_research_subagent_can_write_scratch_markdown_but_not_workspace_files(self) -> None:
        sub_cid = "research-worker-001"
        orchestrator_guard.register_subagent(
            self.base_payload,
            conversation_id=sub_cid,
            type_name="research",
            model="flash",
        )
        scratch_file = str(Path(self.artifact_dir) / "scratch" / "findings.md")
        payload_scratch = dict(
            self.base_payload,
            conversationId=sub_cid,
            toolCall={
                "name": "write_file",
                "args": {
                    "Path": scratch_file,
                    "Content": "# Research Findings\nAnalysis summary.",
                },
            },
        )
        res_scratch = orchestrator_guard.handle_pre_tool(payload_scratch)
        self.assertEqual(res_scratch["decision"], "allow")

        workspace_file = str(orchestrator_guard.PROJECT_ROOT / "src" / "app.py")
        payload_workspace = dict(
            self.base_payload,
            conversationId=sub_cid,
            toolCall={
                "name": "write_file",
                "args": {
                    "Path": workspace_file,
                    "Content": "# Workspace mutation",
                },
            },
        )
        res_workspace = orchestrator_guard.handle_pre_tool(payload_workspace)
        self.assertEqual(res_workspace["decision"], "deny")
        self.assertIn(
            "Research subagents may only write markdown reports",
            res_workspace.get("reason", ""),
        )

    def test_live_runtime_pitfalls_invocation_sync_and_cross_brain_research(self) -> None:
        # 1. Calling handle_pre_invocation with invocationNum=5 persists current_invocation_num=5,
        # so subsequent handle_pre_tool calls that omit invocationNum use bucket "5" instead of "0".
        payload_pre = dict(self.base_payload, invocationNum=5)
        orchestrator_guard.handle_pre_invocation(payload_pre)
        state_after_pre = orchestrator_guard._load_state(payload_pre)
        self.assertEqual(state_after_pre.get("current_invocation_num"), 5)

        tool_payload = dict(
            self.base_payload,
            toolCall={
                "id": "cmd_inv_5",
                "name": "run_command",
                "args": {
                    "CommandLine": "echo 'pitfall 1 test'",
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                },
            },
        )
        tool_payload.pop("invocationNum", None)
        res_tool = orchestrator_guard.handle_pre_tool(tool_payload)
        self.assertEqual(res_tool.get("decision"), "allow")
        state_after_tool = orchestrator_guard._load_state(tool_payload)
        self.assertIn("5", state_after_tool.get("direct_budget", {}))
        self.assertNotIn("0", state_after_tool.get("direct_budget", {}))

        # 2. A research subagent running in its own brain/sub-research directory
        # (while registered in brain/parent-root/orchestrator_state.json) is recognized
        # by _is_research_subagent, can write to brain/parent-root/scratch/brief.md,
        # and is denied when writing to PROJECT_ROOT / "src.py".
        app_data = Path(self.artifact_dir) / "appdata"
        parent_brain = app_data / "brain" / "parent-root"
        sub_brain = app_data / "brain" / "sub-research"
        parent_brain.mkdir(parents=True, exist_ok=True)
        sub_brain.mkdir(parents=True, exist_ok=True)
        (parent_brain / "scratch").mkdir(parents=True, exist_ok=True)

        research_cid = "sub-research-cross-brain-001"
        parent_payload = dict(
            self.base_payload,
            conversationId="parent-root",
            artifactDirectoryPath=str(parent_brain),
        )
        orchestrator_guard.register_subagent(
            parent_payload,
            conversation_id=research_cid,
            type_name="research",
            model="flash",
            role="Cross-Brain Researcher",
        )

        sub_payload = dict(
            self.base_payload,
            conversationId=research_cid,
            artifactDirectoryPath=str(sub_brain),
            modelName="gemini-3-flash",
        )
        sub_state = orchestrator_guard._load_state(sub_payload)
        self.assertTrue(orchestrator_guard._is_research_subagent(sub_payload, sub_state))

        brief_file = str(parent_brain / "scratch" / "brief.md")
        sub_write_scratch = dict(
            sub_payload,
            toolCall={
                "name": "write_file",
                "args": {
                    "Path": brief_file,
                    "Content": "# Sibling Research Brief\nContent here.",
                },
            },
        )
        res_scratch = orchestrator_guard.handle_pre_tool(sub_write_scratch)
        self.assertEqual(res_scratch.get("decision"), "allow")

        workspace_file = str(orchestrator_guard.PROJECT_ROOT / "src.py")
        sub_write_workspace = dict(
            sub_payload,
            toolCall={
                "name": "write_file",
                "args": {
                    "Path": workspace_file,
                    "Content": "x = 42\n",
                },
            },
        )
        res_workspace = orchestrator_guard.handle_pre_tool(sub_write_workspace)
        self.assertEqual(res_workspace.get("decision"), "deny")
        self.assertIn(
            "Research subagents may only write markdown reports",
            res_workspace.get("reason", ""),
        )

        # 3. When the root orchestrator has receipt["dirty"] = True and completes a successful
        # invoke_subagent in handle_post_tool, receipt["dirty"] is cleared to False and receipt["verified"] is True.
        root_state = orchestrator_guard._load_state(self.base_payload)
        receipt = orchestrator_guard._get_test_receipt(root_state, self.base_payload)
        receipt["dirty"] = True
        receipt["verified"] = False
        orchestrator_guard._save_state(self.base_payload, root_state)

        invoke_post_payload = dict(
            self.base_payload,
            toolCall={
                "id": "call_invoke_sub_success",
                "name": "invoke_subagent",
                "args": {},
            },
            error="",
        )
        orchestrator_guard.handle_post_tool(invoke_post_payload)

        updated_state = orchestrator_guard._load_state(self.base_payload)
        updated_receipt = orchestrator_guard._get_test_receipt(updated_state, self.base_payload)
        self.assertFalse(updated_receipt.get("dirty"))
        self.assertTrue(updated_receipt.get("verified"))
        self.assertEqual(updated_receipt.get("last_test_cmd"), "invoke_subagent")

    def test_direct_budget_uses_current_invocation_num_fallback(self) -> None:
        payload_inv3 = dict(self.base_payload, invocationNum=3)
        orchestrator_guard.handle_pre_invocation(payload_inv3)

        cmd_payload_1 = dict(
            self.base_payload,
            toolCall={
                "id": "cmd_1",
                "name": "run_command",
                "args": {
                    "CommandLine": "echo 'cmd 1'",
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                },
            },
        )
        cmd_payload_1.pop("invocationNum", None)
        res_cmd1 = orchestrator_guard.handle_pre_tool(cmd_payload_1)
        self.assertEqual(res_cmd1.get("decision"), "allow")

        cmd_payload_2 = dict(
            self.base_payload,
            toolCall={
                "id": "cmd_2",
                "name": "run_command",
                "args": {
                    "CommandLine": "echo 'cmd 2'",
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                },
            },
        )
        cmd_payload_2.pop("invocationNum", None)
        res_cmd2 = orchestrator_guard.handle_pre_tool(cmd_payload_2)
        self.assertEqual(res_cmd2.get("decision"), "allow")

        state = orchestrator_guard._load_state(self.base_payload)
        self.assertIn("3", state.get("direct_budget", {}))
        self.assertNotIn("0", state.get("direct_budget", {}))
        self.assertEqual(len(state["direct_budget"]["3"]["commands"]), 2)

        cmd_payload_3 = dict(
            self.base_payload,
            toolCall={
                "id": "cmd_3",
                "name": "run_command",
                "args": {
                    "CommandLine": "echo 'cmd 3'",
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                },
            },
        )
        cmd_payload_3.pop("invocationNum", None)
        res_cmd3 = orchestrator_guard.handle_pre_tool(cmd_payload_3)
        self.assertEqual(res_cmd3.get("decision"), "deny")
        self.assertIn("command limit", res_cmd3.get("reason", ""))

        payload_inv4 = dict(self.base_payload, invocationNum=4)
        orchestrator_guard.handle_pre_invocation(payload_inv4)

        cmd_payload_4 = dict(
            self.base_payload,
            toolCall={
                "id": "cmd_4",
                "name": "run_command",
                "args": {
                    "CommandLine": "echo 'fresh budget cmd'",
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                },
            },
        )
        cmd_payload_4.pop("invocationNum", None)
        res_cmd4 = orchestrator_guard.handle_pre_tool(cmd_payload_4)
        self.assertEqual(res_cmd4.get("decision"), "allow")

        state_after = orchestrator_guard._load_state(self.base_payload)
        self.assertIn("4", state_after.get("direct_budget", {}))
        self.assertEqual(len(state_after["direct_budget"]["4"]["commands"]), 1)

        # Verify protojson zero (0, "0") is treated as unpopulated in _current_invocation_key
        # and falls back to persisted current_invocation_num (which is 4)
        self.assertEqual(
            orchestrator_guard._current_invocation_key({"invocationNum": 0}, state_after),
            "4",
        )
        self.assertEqual(
            orchestrator_guard._current_invocation_key({"invocationNum": "0"}, state_after),
            "4",
        )
        self.assertEqual(
            orchestrator_guard._current_invocation_key({"invocationNum": 0}, {"current_invocation_num": 5}),
            "5",
        )
        self.assertEqual(
            orchestrator_guard._current_invocation_key({"invocationNum": None}, state_after),
            "4",
        )
        self.assertEqual(
            orchestrator_guard._current_invocation_key({}, state_after),
            "4",
        )
        self.assertEqual(
            orchestrator_guard._current_invocation_key({"invocationNum": 7}, state_after),
            "7",
        )
        self.assertEqual(
            orchestrator_guard._current_invocation_key({"invocationNum": "7"}, state_after),
            "7",
        )
        self.assertEqual(
            orchestrator_guard._current_invocation_key({"invocationNum": 0}, {"current_invocation_num": 0}),
            "0",
        )
        self.assertEqual(
            orchestrator_guard._current_invocation_key({"invocationNum": "0"}, {"current_invocation_num": 0}),
            "0",
        )

        # Tool call with protojson zero (0 or "0") uses current invocation bucket (4)
        cmd_payload_proto0 = dict(
            self.base_payload,
            invocationNum=0,
            toolCall={
                "id": "cmd_proto0",
                "name": "run_command",
                "args": {
                    "CommandLine": "echo 'proto0 cmd'",
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                },
            },
        )
        res_proto0 = orchestrator_guard.handle_pre_tool(cmd_payload_proto0)
        self.assertEqual(res_proto0.get("decision"), "allow")
        state_after_proto0 = orchestrator_guard._load_state(self.base_payload)
        self.assertEqual(len(state_after_proto0["direct_budget"]["4"]["commands"]), 2)
        self.assertNotIn("0", state_after_proto0.get("direct_budget", {}))

        # Calling handle_pre_invocation resets direct_budget for that invocation and "0"
        state_pre_reset = orchestrator_guard._load_state(self.base_payload)
        state_pre_reset["direct_budget"]["5"] = {"commands": ["old_cmd_5"]}
        state_pre_reset["direct_budget"]["0"] = {"commands": ["old_cmd_0"]}
        orchestrator_guard._save_state(self.base_payload, state_pre_reset)

        payload_inv5 = dict(self.base_payload, invocationNum=5)
        orchestrator_guard.handle_pre_invocation(payload_inv5)

        state_after_inv5 = orchestrator_guard._load_state(self.base_payload)
        self.assertEqual(state_after_inv5.get("current_invocation_num"), 5)
        self.assertNotIn("5", state_after_inv5.get("direct_budget", {}))
        self.assertNotIn("0", state_after_inv5.get("direct_budget", {}))

        # Calling handle_pre_invocation with protojson zero (0 or "0") preserves current_invocation_num
        # and resets direct_budget key "0"
        state_after_inv5["direct_budget"]["0"] = {"commands": ["old_cmd_0"]}
        state_after_inv5["direct_budget"]["6"] = {"commands": ["preserved_cmd_6"]}
        orchestrator_guard._save_state(self.base_payload, state_after_inv5)

        payload_proto_zero = dict(self.base_payload, invocationNum=0)
        orchestrator_guard.handle_pre_invocation(payload_proto_zero)

        state_proto_cleared = orchestrator_guard._load_state(self.base_payload)
        self.assertEqual(state_proto_cleared.get("current_invocation_num"), 5)
        self.assertNotIn("0", state_proto_cleared.get("direct_budget", {}))
        self.assertIn("6", state_proto_cleared.get("direct_budget", {}))

        # Also test with protojson string "0"
        state_proto_cleared["direct_budget"]["0"] = {"commands": ["old_cmd_0"]}
        orchestrator_guard._save_state(self.base_payload, state_proto_cleared)
        payload_proto_str_zero = dict(self.base_payload, invocationNum="0")
        orchestrator_guard.handle_pre_invocation(payload_proto_str_zero)
        state_str_cleared = orchestrator_guard._load_state(self.base_payload)
        self.assertEqual(state_str_cleared.get("current_invocation_num"), 5)
        self.assertNotIn("0", state_str_cleared.get("direct_budget", {}))

        # Calling handle_pre_invocation with invocationNum omitted resets current_invocation_num to 0
        # and clears direct_budget key "0"
        state_after["direct_budget"]["0"] = {"commands": ["old_cmd"]}
        orchestrator_guard._save_state(self.base_payload, state_after)

        payload_omit_inv = dict(self.base_payload)
        payload_omit_inv.pop("invocationNum", None)
        orchestrator_guard.handle_pre_invocation(payload_omit_inv)

        state_cleared = orchestrator_guard._load_state(self.base_payload)
        self.assertNotIn("0", state_cleared.get("direct_budget", {}))

    def test_compound_git_and_pytest_command_not_exempt(self) -> None:
        orchestrator_guard.handle_pre_invocation(self.base_payload)

        dummy_file = orchestrator_guard.PROJECT_ROOT / "test_dummy_source.py"
        try:
            write_payload = dict(
                self.base_payload,
                toolCall={
                    "name": "write_file",
                    "args": {
                        "Path": str(dummy_file),
                        "Content": "x = 1\n",
                    },
                },
            )
            res_write = orchestrator_guard.handle_pre_tool(write_payload)
            self.assertEqual(res_write.get("decision"), "allow")

            compound_payload_1 = dict(
                self.base_payload,
                toolCall={
                    "id": "compound_1",
                    "name": "run_command",
                    "args": {
                        "CommandLine": "git status && pytest",
                        "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                    },
                },
            )
            res_cmd_1 = orchestrator_guard.handle_pre_tool(compound_payload_1)
            self.assertEqual(res_cmd_1.get("decision"), "allow")

            compound_payload_2 = dict(
                self.base_payload,
                toolCall={
                    "id": "compound_2",
                    "name": "run_command",
                    "args": {
                        "CommandLine": "git status && pytest",
                        "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                    },
                },
            )
            res_cmd_2 = orchestrator_guard.handle_pre_tool(compound_payload_2)
            self.assertEqual(res_cmd_2.get("decision"), "deny")
            self.assertIn(
                "Direct workspace edit followed by iterative test execution detected",
                res_cmd_2.get("reason", ""),
            )
        finally:
            if dummy_file.exists():
                dummy_file.unlink()

        clean_payload_inv = dict(self.base_payload, invocationNum=99)
        orchestrator_guard.handle_pre_invocation(clean_payload_inv)

        clean_cmd_payload = dict(
            clean_payload_inv,
            toolCall={
                "id": "clean_cmd_1",
                "name": "run_command",
                "args": {
                    "CommandLine": "git status && pytest",
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                },
            },
        )
        res_clean = orchestrator_guard.handle_pre_tool(clean_cmd_payload)
        self.assertEqual(res_clean.get("decision"), "allow")
        state = orchestrator_guard._load_state(clean_payload_inv)
        self.assertIn("git status && pytest", state["direct_budget"]["99"]["commands"])

    def test_grep_search_on_log_or_transcript_blocked_for_root(self) -> None:
        orchestrator_guard.handle_pre_invocation(self.base_payload)

        payload_log = dict(
            self.base_payload,
            toolCall={
                "name": "grep_search",
                "args": {
                    "SearchPath": "/tmp/debug.log",
                    "Query": "ERROR",
                },
            },
        )
        res_log = orchestrator_guard.handle_pre_tool(payload_log)
        self.assertEqual(res_log.get("decision"), "deny")
        self.assertIn("Context separation violation", res_log.get("reason", ""))
        self.assertIn("debug.log", res_log.get("reason", ""))

        payload_transcript = dict(
            self.base_payload,
            toolCall={
                "name": "grep_search",
                "args": {
                    "SearchPath": "transcript.jsonl",
                    "Query": "conversationId",
                },
            },
        )
        res_transcript = orchestrator_guard.handle_pre_tool(payload_transcript)
        self.assertEqual(res_transcript.get("decision"), "deny")
        self.assertIn("Context separation violation", res_transcript.get("reason", ""))

        payload_normal = dict(
            self.base_payload,
            toolCall={
                "name": "grep_search",
                "args": {
                    "SearchPath": str(orchestrator_guard.PROJECT_ROOT / "tests"),
                    "Query": "test",
                },
            },
        )
        res_normal = orchestrator_guard.handle_pre_tool(payload_normal)
        self.assertEqual(res_normal.get("decision"), "allow")

    def test_transcript_offset_cursor_and_doc_uuid_ignored(self) -> None:
        transcript_file = Path(self.artifact_dir) / "transcript_cursor.jsonl"
        payload = dict(
            self.base_payload,
            transcriptPath=str(transcript_file),
        )
        state = orchestrator_guard._load_state(payload)
        state["pending_spawns"] = [
            {"role": "Worker 1", "type": "implementer", "model": "flash"}
        ]
        orchestrator_guard._save_state(payload, state)

        doc_cid = "aaaa1111-aaaa-1111-aaaa-1111aaaa1111"
        line_doc = json.dumps({
            "stepId": "00000000-0000-0000-0000-000000000001",
            "name": "view_file",
            "args": {"AbsolutePath": "/workspace/docs/hooks.md"},
            "content": f"Created the following subagents: conversationId: {doc_cid}",
        }) + "\n"
        transcript_file.write_text(line_doc, encoding="utf-8")

        changed = orchestrator_guard._sync_subagents_from_transcript(payload, state)
        self.assertTrue(changed)
        offset_after_first = state.get("transcript_offset", 0)
        self.assertGreater(offset_after_first, 0)
        self.assertNotIn(doc_cid, state.get("subagents", {}))
        self.assertEqual(len(state.get("pending_spawns", [])), 1)

        real_cid = "bbbb2222-bbbb-2222-bbbb-2222bbbb2222"
        line_real = json.dumps({
            "stepId": "00000000-0000-0000-0000-000000000002",
            "name": "invoke_subagent",
            "result": {
                "conversationId": real_cid,
            },
        }) + "\n"
        with open(transcript_file, "a", encoding="utf-8") as fh:
            fh.write(line_real)

        changed_real = orchestrator_guard._sync_subagents_from_transcript(payload, state)
        self.assertTrue(changed_real)
        offset_after_second = state.get("transcript_offset", 0)
        self.assertGreater(offset_after_second, offset_after_first)
        self.assertIn(real_cid, state.get("subagents", {}))
        self.assertEqual(len(state.get("pending_spawns", [])), 0)

        changed_third = orchestrator_guard._sync_subagents_from_transcript(payload, state)
        self.assertFalse(changed_third)
        self.assertEqual(state.get("transcript_offset"), offset_after_second)

        # Test singular "Created the following subagent:" transcript line pairs pending spawn
        singular_cid = "cccc3333-cccc-3333-cccc-3333cccc3333"
        state["pending_spawns"] = [
            {"role": "Worker 2", "type": "scout", "model": "flash_lite"}
        ]
        line_singular = f"Created the following subagent: conversationId: {singular_cid}\n"
        with open(transcript_file, "a", encoding="utf-8") as fh:
            fh.write(line_singular)

        changed_singular = orchestrator_guard._sync_subagents_from_transcript(payload, state)
        self.assertTrue(changed_singular)
        self.assertIn(singular_cid, state.get("subagents", {}))
        self.assertEqual(state.get("subagents", {})[singular_cid]["model"], "flash_lite")
        self.assertEqual(len(state.get("pending_spawns", [])), 0)

    def test_self_subagent_meta_cached_from_sibling_brain(self) -> None:
        app_data = Path(self.artifact_dir) / "appdata"
        parent_brain = app_data / "brain" / "parent-orch-cid"
        sub_brain = app_data / "brain" / "sub-worker-cid"
        parent_brain.mkdir(parents=True, exist_ok=True)
        sub_brain.mkdir(parents=True, exist_ok=True)

        worker_cid = "sub-worker-cid-5555"
        parent_payload = dict(
            self.base_payload,
            conversationId="parent-orch-cid",
            artifactDirectoryPath=str(parent_brain),
        )
        orchestrator_guard.register_subagent(
            parent_payload,
            conversation_id=worker_cid,
            type_name="research",
            model="flash",
            role="Worker Researcher",
        )

        sub_payload = dict(
            self.base_payload,
            conversationId=worker_cid,
            artifactDirectoryPath=str(sub_brain),
            modelName="gemini-3-flash",
        )
        sub_state = orchestrator_guard._load_state(sub_payload)
        self.assertNotIn("self_subagent_meta", sub_state)

        is_known = orchestrator_guard._is_known_subagent(sub_payload, sub_state)
        self.assertTrue(is_known)

        self.assertIn("self_subagent_meta", sub_state)
        self.assertEqual(sub_state["self_subagent_meta"]["type_name"], "research")

        persisted_sub_state = orchestrator_guard._load_state(sub_payload)
        self.assertIn("self_subagent_meta", persisted_sub_state)
        self.assertEqual(
            persisted_sub_state["self_subagent_meta"]["type_name"], "research"
        )

        self.assertTrue(
            orchestrator_guard._is_research_subagent(sub_payload, persisted_sub_state)
        )

    def test_commented_out_assertion_tampering_blocked_in_edit_and_diff(self) -> None:
        test_file = str(orchestrator_guard.PROJECT_ROOT / "tests" / "test_sample.py")
        commented_assert_edit = dict(
            self.base_payload,
            toolCall={
                "name": "edit_file",
                "args": {
                    "Path": test_file,
                    "Target": "        self.assertEqual(a, b)\n",
                    "Replacement": "        # self.assertEqual(a, b)\n",
                },
            },
        )
        res_edit = orchestrator_guard.handle_pre_tool(commented_assert_edit)
        self.assertEqual(res_edit["decision"], "deny")
        self.assertIn("reduces assertion count", res_edit["reason"])

        diff_commented_assert = (
            "--- a/tests/test_auth.py\n"
            "+++ b/tests/test_auth.py\n"
            "@@ -10,1 +10,1 @@\n"
            "-    self.assertEqual(token, 'abc')\n"
            "+    # self.assertEqual(token, 'abc')\n"
        )
        res_diff = orchestrator_guard.audit_diff_text(diff_commented_assert)
        self.assertIsNotNone(res_diff)
        self.assertIn("Diff audit detected net removal of 1 assertion(s)", res_diff)

    def test_step_idx_deduplication_and_scout_does_not_clear_dirty_receipt(self) -> None:
        orchestrator_guard.handle_pre_invocation(self.base_payload)

        # 1. Duplicate PreToolUse calls with same stepIdx (and no toolCall.id) count once
        step_payload = dict(
            self.base_payload,
            stepIdx=42,
            toolCall={
                "name": "run_command",
                "args": {
                    "CommandLine": "echo 'step test'",
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                },
            },
        )
        res1 = orchestrator_guard.handle_pre_tool(step_payload)
        self.assertEqual(res1["decision"], "allow")
        res2 = orchestrator_guard.handle_pre_tool(step_payload)
        self.assertEqual(res2["decision"], "allow")

        state = orchestrator_guard._load_state(self.base_payload)
        self.assertEqual(len(state["direct_budget"]["1"]["commands"]), 1)

        # 2. handle_post_tool for a scout invoke_subagent does NOT clear a dirty test_receipt
        receipt = orchestrator_guard._get_test_receipt(state, self.base_payload)
        receipt["dirty"] = True
        receipt["verified"] = False
        orchestrator_guard._save_state(self.base_payload, state)

        scout_post_payload = dict(
            self.base_payload,
            toolCall={
                "id": "scout_post_call",
                "name": "invoke_subagent",
                "args": {
                    "TypeName": "scout",
                    "Role": "Read Scout",
                },
            },
            error="",
        )
        orchestrator_guard.handle_post_tool(scout_post_payload)

        state_after_scout = orchestrator_guard._load_state(self.base_payload)
        receipt_after_scout = orchestrator_guard._get_test_receipt(
            state_after_scout, self.base_payload
        )
        self.assertTrue(receipt_after_scout["dirty"])
        self.assertFalse(receipt_after_scout["verified"])

        # 3. echo pytest does not clear a dirty test_receipt
        echo_post_payload = dict(
            self.base_payload,
            toolCall={
                "id": "echo_post_call",
                "name": "run_command",
                "args": {
                    "CommandLine": "echo pytest",
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                },
            },
            error="",
        )
        orchestrator_guard.handle_post_tool(echo_post_payload)

        state_after_echo = orchestrator_guard._load_state(self.base_payload)
        receipt_after_echo = orchestrator_guard._get_test_receipt(
            state_after_echo, self.base_payload
        )
        self.assertTrue(receipt_after_echo["dirty"])
        self.assertFalse(receipt_after_echo["verified"])

    def test_cli_doctor_and_status_modes(self) -> None:
        status_res = orchestrator_guard._cli_status(self.base_payload)
        self.assertIn("projectRoot", status_res)
        self.assertIn("stateFile", status_res)
        self.assertIn("active", status_res)
        self.assertIn("currentInvocation", status_res)
        self.assertIn("directBudget", status_res)
        self.assertIn("testReceipt", status_res)
        self.assertIn("circuitBreaker", status_res)

        doctor_res = orchestrator_guard._cli_doctor()
        self.assertIn("healthy", doctor_res)
        self.assertIn("pluginHooksPresent", doctor_res)
        self.assertIn("issues", doctor_res)
        self.assertTrue(doctor_res["pluginHooksPresent"])
        self.assertIsInstance(doctor_res["issues"], list)

    def test_inactive_session_on_normal_chat(self) -> None:
        normal_payload = dict(
            self.base_payload,
            lastUserInput="normal chat message",
            toolCall={
                "name": "read_file",
                "args": {"Path": str(orchestrator_guard.PROJECT_ROOT / "README.md")},
            },
        )
        # handle_pre_invocation returns {}
        res_pre_inv = orchestrator_guard.handle_pre_invocation(normal_payload)
        self.assertEqual(res_pre_inv, {})

        state = orchestrator_guard._load_state(normal_payload)
        self.assertFalse(state.get("active"))
        self.assertTrue(state.get("session_evaluated"))

        # handle_pre_tool returns {"decision": "allow"}
        res_pre_tool = orchestrator_guard.handle_pre_tool(normal_payload)
        self.assertEqual(res_pre_tool, {"decision": "allow"})

        # handle_post_tool returns {}
        res_post_tool = orchestrator_guard.handle_post_tool(normal_payload)
        self.assertEqual(res_post_tool, {})

        # handle_stop returns {}
        stop_payload = dict(
            normal_payload,
            fullyIdle=True,
            terminationReason="NO_TOOL_CALL",
        )
        res_stop = orchestrator_guard.handle_stop(stop_payload)
        self.assertEqual(res_stop, {})

    def test_slash_command_activation_and_persistence_across_turns(self) -> None:
        slash_payload = dict(
            self.base_payload,
            lastUserInput="/philharmonic implement feature X",
            invocationNum=1,
        )
        res_inv1 = orchestrator_guard.handle_pre_invocation(slash_payload)
        self.assertEqual(res_inv1, {})

        state = orchestrator_guard._load_state(slash_payload)
        self.assertTrue(state.get("active"))
        self.assertTrue(state.get("session_evaluated"))

        # Subsequent turn with normal message preserves activation
        turn2_payload = dict(
            self.base_payload,
            lastUserInput="now run the tests",
            invocationNum=2,
        )
        res_inv2 = orchestrator_guard.handle_pre_invocation(turn2_payload)
        self.assertEqual(res_inv2, {})

        state2 = orchestrator_guard._load_state(turn2_payload)
        self.assertTrue(state2.get("active"))

        # In active session, pre_tool allows valid tool calls and enforces limits
        tool_payload = dict(
            turn2_payload,
            toolCall={
                "name": "read_file",
                "args": {"Path": str(orchestrator_guard.PROJECT_ROOT / "README.md")},
            },
        )
        res_tool = orchestrator_guard.handle_pre_tool(tool_payload)
        self.assertEqual(res_tool, {"decision": "allow"})

    def test_reading_skill_file_activates_inactive_session(self) -> None:
        normal_payload = dict(
            self.base_payload,
            lastUserInput="please look up some instructions",
        )
        orchestrator_guard.handle_pre_invocation(normal_payload)
        state_before = orchestrator_guard._load_state(normal_payload)
        self.assertFalse(state_before.get("active"))

        # Agent reads the philharmonic SKILL.md file
        read_skill_payload = dict(
            self.base_payload,
            toolCall={
                "name": "read_file",
                "args": {
                    "Path": str(
                        orchestrator_guard.PROJECT_ROOT
                        / ".agents"
                        / "plugins"
                        / "philharmonic_orchestra"
                        / "skills"
                        / "philharmonic"
                        / "SKILL.md"
                    )
                },
            },
        )
        res_pre_tool = orchestrator_guard.handle_pre_tool(read_skill_payload)
        self.assertEqual(res_pre_tool, {"decision": "allow"})

        state_after = orchestrator_guard._load_state(self.base_payload)
        self.assertTrue(state_after.get("active"))
        self.assertTrue(state_after.get("session_evaluated"))

    def test_cli_activate_and_deactivate_modes(self) -> None:
        # CLI activate
        act_res = orchestrator_guard._cli_activate(self.base_payload)
        self.assertEqual(act_res["status"], "activated")
        self.assertTrue(act_res["active"])
        self.assertIn("state_path", act_res)

        status_res = orchestrator_guard._cli_status(self.base_payload)
        self.assertTrue(status_res["active"])

        # CLI deactivate
        deact_res = orchestrator_guard._cli_deactivate(self.base_payload)
        self.assertEqual(deact_res["status"], "deactivated")
        self.assertFalse(deact_res["active"])

        status_res2 = orchestrator_guard._cli_status(self.base_payload)
        self.assertFalse(status_res2["active"])

    def test_philharmonic_skill_file_exists_and_valid_frontmatter(self) -> None:
        skill_path = (
            orchestrator_guard.PROJECT_ROOT
            / ".agents"
            / "plugins"
            / "philharmonic_orchestra"
            / "skills"
            / "philharmonic"
            / "SKILL.md"
        )
        self.assertTrue(skill_path.exists())
        content = skill_path.read_text(encoding="utf-8")
        self.assertTrue(content.startswith("---"))
        self.assertIn("name: philharmonic", content)
        self.assertIn("description:", content)
        self.assertIn("## On-Demand Activation", content)
        self.assertIn("## 1. Lazy Subagent Registration", content)
        self.assertIn("## 2. Token-Optimal Dispatch and Model Selection Rules", content)
        self.assertIn("## 3. Structured Subagent Prompt Contract", content)
        self.assertIn("## 4. Reactive Coordination and Context Hygiene", content)
        self.assertIn("## 5. Empirical Test Receipts, Circuit Breaker, and Diff Audit", content)

    def test_remote_vm_tool_schemas_support(self) -> None:
        # 1. lookup_skill(skill_name="philharmonic") activates an inactive session
        normal_payload = dict(
            self.base_payload,
            lastUserInput="hello remote vm",
        )
        orchestrator_guard.handle_pre_invocation(normal_payload)
        self.assertFalse(orchestrator_guard._load_state(normal_payload).get("active"))

        lookup_payload = dict(
            self.base_payload,
            toolCall={
                "name": "lookup_skill",
                "args": {"skill_name": "philharmonic"},
            },
        )
        res_lookup = orchestrator_guard.handle_pre_tool(lookup_payload)
        self.assertEqual(res_lookup["decision"], "allow")
        self.assertTrue(orchestrator_guard._load_state(self.base_payload).get("active"))

        # 1b. lookup_skills(skill_names=["philharmonic"]) also activates an inactive session
        orchestrator_guard._cli_deactivate(self.base_payload)
        self.assertFalse(orchestrator_guard._load_state(self.base_payload).get("active"))
        lookup_multi_payload = dict(
            self.base_payload,
            toolCall={
                "name": "lookup_skills",
                "args": {"skill_names": ["other_skill", "philharmonic"]},
            },
        )
        res_lookup_multi = orchestrator_guard.handle_pre_tool(lookup_multi_payload)
        self.assertEqual(res_lookup_multi["decision"], "allow")
        self.assertTrue(orchestrator_guard._load_state(self.base_payload).get("active"))

        # 2. snake_case define_subagent forces enable_subagent_tools=False
        define_payload = dict(
            self.base_payload,
            toolCall={
                "name": "define_subagent",
                "args": {
                    "name": "scout",
                    "enable_write_tools": False,
                    "enable_subagent_tools": True,
                },
            },
        )
        res_define = orchestrator_guard.handle_pre_tool(define_payload)
        self.assertEqual(res_define["decision"], "allow")
        self.assertEqual(res_define["overwrite"], {"enable_subagent_tools": False})

        # 3. Subagents array in invoke_subagent rewrites inherit models and blocks pro
        subagents_pro_payload = dict(
            self.base_payload,
            toolCall={
                "id": "remote_sub_pro",
                "name": "invoke_subagent",
                "args": {
                    "Subagents": [
                        {
                            "TypeName": "implementer",
                            "Role": "Pro Worker",
                            "Model": "pro",
                            "Prompt": VALID_PROMPT,
                        }
                    ]
                },
            },
        )
        res_pro = orchestrator_guard.handle_pre_tool(subagents_pro_payload)
        self.assertEqual(res_pro["decision"], "deny")
        self.assertIn("Model='pro'", res_pro["reason"])

        subagents_inherit_payload = dict(
            self.base_payload,
            toolCall={
                "id": "remote_sub_inherit",
                "name": "invoke_subagent",
                "args": {
                    "Subagents": [
                        {
                            "TypeName": "scout",
                            "Role": "Remote Scout",
                            "Model": "inherit",
                            "Prompt": VALID_PROMPT,
                        },
                        {
                            "TypeName": "implementer",
                            "Role": "Remote Implementer",
                            "Model": "inherit",
                            "Prompt": VALID_PROMPT,
                        },
                    ]
                },
            },
        )
        res_inherit = orchestrator_guard.handle_pre_tool(subagents_inherit_payload)
        self.assertEqual(res_inherit["decision"], "allow")
        self.assertEqual(
            res_inherit["overwrite"]["Subagents"][0]["Model"], "flash_lite"
        )
        self.assertEqual(
            res_inherit["overwrite"]["Subagents"][1]["Model"], "flash"
        )
        state_after_spawn = orchestrator_guard._load_state(self.base_payload)
        self.assertEqual(len(state_after_spawn.get("pending_spawns", [])), 2)

        # 3b. handle_post_tool with Subagents list rolls back all pending spawns on error
        post_err_payload = dict(
            subagents_inherit_payload,
            error="remote subagent spawn failed",
        )
        orchestrator_guard.handle_post_tool(post_err_payload)
        state_after_rollback = orchestrator_guard._load_state(self.base_payload)
        self.assertEqual(len(state_after_rollback.get("pending_spawns", [])), 0)

        # 4. view_file, write_to_file, and replace_file_content enforce budget, dirty receipt, and tamper checks
        test_file = str(orchestrator_guard.PROJECT_ROOT / "tests" / "test_remote.py")
        tamper_replace = dict(
            self.base_payload,
            toolCall={
                "name": "replace_file_content",
                "args": {
                    "TargetFile": test_file,
                    "TargetContent": "    assert x == 1\n",
                    "ReplacementContent": "    # assert x == 1\n",
                },
            },
        )
        res_tamper = orchestrator_guard.handle_pre_tool(tamper_replace)
        self.assertEqual(res_tamper["decision"], "deny")
        self.assertIn("reduces assertion count", res_tamper["reason"])

        tamper_write_to_file = dict(
            self.base_payload,
            toolCall={
                "name": "write_to_file",
                "args": {
                    "TargetFile": test_file,
                    "CodeContent": "def test_x():\n    assert True\n",
                },
            },
        )
        res_tamper_wtf = orchestrator_guard.handle_pre_tool(tamper_write_to_file)
        self.assertEqual(res_tamper_wtf["decision"], "deny")
        self.assertIn("assert True", res_tamper_wtf["reason"])

        code_file = str(orchestrator_guard.PROJECT_ROOT / "remote_code.py")
        res_write_code = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "name": "write_to_file",
                    "args": {
                        "TargetFile": code_file,
                        "CodeContent": "val = 123\n",
                    },
                },
            )
        )
        self.assertEqual(res_write_code["decision"], "allow")
        state_dirty = orchestrator_guard._load_state(self.base_payload)
        self.assertTrue(
            orchestrator_guard._get_test_receipt(state_dirty, self.base_payload).get("dirty")
        )

        # 4b. handle_post_tool with Subagents list keeps dirty=True for scout/research only,
        # and clears dirty=False when an implementer entry succeeds
        orchestrator_guard.handle_post_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "remote_post_readonly",
                    "name": "invoke_subagent",
                    "args": {
                        "Subagents": [
                            {"TypeName": "scout", "Role": "Read Scout"},
                            {"TypeName": "research", "Role": "Doc Researcher"},
                        ]
                    },
                },
                error="",
            )
        )
        state_still_dirty = orchestrator_guard._load_state(self.base_payload)
        self.assertTrue(
            orchestrator_guard._get_test_receipt(state_still_dirty, self.base_payload).get("dirty")
        )

        orchestrator_guard.handle_post_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "remote_post_impl",
                    "name": "invoke_subagent",
                    "args": {
                        "Subagents": [
                            {"TypeName": "scout", "Role": "Read Scout"},
                            {"TypeName": "implementer", "Role": "Code Implementer"},
                        ]
                    },
                },
                error="",
            )
        )
        state_verified = orchestrator_guard._load_state(self.base_payload)
        receipt_verified = orchestrator_guard._get_test_receipt(
            state_verified, self.base_payload
        )
        self.assertFalse(receipt_verified.get("dirty"))
        self.assertTrue(receipt_verified.get("verified"))

        f1 = str(orchestrator_guard.PROJECT_ROOT / "remote_a.py")
        f2 = str(orchestrator_guard.PROJECT_ROOT / "remote_b.py")
        f3 = str(orchestrator_guard.PROJECT_ROOT / "remote_c.py")
        for f in (f1, f2):
            r = orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={"name": "view_file", "args": {"AbsolutePath": f}},
                )
            )
            self.assertEqual(r["decision"], "allow")
        r3 = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={"name": "view_file", "args": {"AbsolutePath": f3}},
            )
        )
        self.assertEqual(r3["decision"], "deny")
        self.assertIn("read limit (2 workspace files) exceeded", r3["reason"])

    def test_redirection_splitting_and_regex_literal_paths_in_workspace(self) -> None:
        self.assertEqual(
            orchestrator_guard._split_shell_commands("git diff 2>&1"),
            ["git diff 2>&1"],
        )
        self.assertTrue(orchestrator_guard._is_git_only_command("git diff 2>&1"))
        self.assertFalse(
            orchestrator_guard._is_git_only_command("git diff 2>&1 && pytest")
        )

        roots = orchestrator_guard._allowed_roots(self.base_payload)
        cwd = str(orchestrator_guard.PROJECT_ROOT)
        self.assertTrue(
            orchestrator_guard._command_stays_in_workspace(
                "awk '/pattern/' README.md", cwd, roots
            )
        )
        self.assertTrue(
            orchestrator_guard._command_stays_in_workspace(
                "sed '/foo/d' README.md", cwd, roots
            )
        )
        self.assertFalse(
            orchestrator_guard._command_stays_in_workspace(
                "cat /etc/passwd", cwd, roots
            )
        )

    def test_tripped_circuit_breaker_blocks_compound_git_command(self) -> None:
        state = orchestrator_guard._load_state(self.base_payload)
        cb = orchestrator_guard._get_circuit_breaker(state)
        cb["failure_streak"] = 2
        cb["tripped"] = True
        cb["last_error"] = "exit 1"
        orchestrator_guard._save_state(self.base_payload, state)

        res_compound = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "name": "run_command",
                    "args": {
                        "CommandLine": "git status && pytest",
                        "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                    },
                },
            )
        )
        self.assertEqual(res_compound["decision"], "deny")
        self.assertIn("2-turn circuit breaker tripped", res_compound["reason"])

        res_git_status = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "name": "run_command",
                    "args": {
                        "CommandLine": "git status",
                        "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                    },
                },
            )
        )
        self.assertEqual(res_git_status["decision"], "allow")

        res_git_diff = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "name": "run_command",
                    "args": {
                        "CommandLine": "git diff 2>&1",
                        "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                    },
                },
            )
        )
        self.assertEqual(res_git_diff["decision"], "allow")

    def test_parallel_sibling_brain_subagents_claim_distinct_pending_spawns(self) -> None:
        app_data = Path(self.artifact_dir) / "appdata_parallel"
        parent_brain = app_data / "brain" / "parent-cid"
        sub1_brain = app_data / "brain" / "sub-1-cid"
        sub2_brain = app_data / "brain" / "sub-2-cid"
        parent_brain.mkdir(parents=True, exist_ok=True)
        sub1_brain.mkdir(parents=True, exist_ok=True)
        sub2_brain.mkdir(parents=True, exist_ok=True)

        parent_payload = dict(
            self.base_payload,
            conversationId="parent-cid",
            artifactDirectoryPath=str(parent_brain),
            toolCall={
                "id": "parallel_spawn_1",
                "name": "invoke_subagent",
                "args": {
                    "Subagents": [
                        {
                            "TypeName": "scout",
                            "Role": "First Scout",
                            "Model": "flash_lite",
                            "Prompt": VALID_PROMPT,
                        },
                        {
                            "TypeName": "implementer",
                            "Role": "Second Implementer",
                            "Model": "flash",
                            "Prompt": VALID_PROMPT,
                        },
                    ]
                },
            },
        )
        self.assertEqual(
            orchestrator_guard.handle_pre_tool(parent_payload)["decision"], "allow"
        )

        sub1_payload = dict(
            self.base_payload,
            conversationId="sub-1-cid",
            artifactDirectoryPath=str(sub1_brain),
        )
        sub1_state = orchestrator_guard._load_state(sub1_payload)
        self.assertTrue(orchestrator_guard._is_known_subagent(sub1_payload, sub1_state))
        self.assertEqual(sub1_state["self_subagent_meta"]["type_name"], "scout")

        sub2_payload = dict(
            self.base_payload,
            conversationId="sub-2-cid",
            artifactDirectoryPath=str(sub2_brain),
        )
        sub2_state = orchestrator_guard._load_state(sub2_payload)
        self.assertTrue(orchestrator_guard._is_known_subagent(sub2_payload, sub2_state))
        self.assertEqual(sub2_state["self_subagent_meta"]["type_name"], "implementer")

        parent_state_after = orchestrator_guard._load_state(parent_payload)
        self.assertEqual(len(parent_state_after.get("pending_spawns", [])), 0)
        self.assertEqual(
            parent_state_after["subagents"]["sub-1-cid"]["type_name"], "scout"
        )
        self.assertEqual(
            parent_state_after["subagents"]["sub-2-cid"]["type_name"], "implementer"
        )

    def test_scout_readonly_enforced_in_define_and_pre_tool(self) -> None:
        define_pascal = dict(
            self.base_payload,
            toolCall={
                "name": "define_subagent",
                "args": {
                    "Name": "scout",
                    "EnableWriteTools": True,
                    "EnableSubagentTools": True,
                },
            },
        )
        res_def1 = orchestrator_guard.handle_pre_tool(define_pascal)
        self.assertEqual(res_def1["decision"], "allow")
        self.assertEqual(
            res_def1["overwrite"],
            {"EnableSubagentTools": False, "EnableWriteTools": False},
        )

        define_snake = dict(
            self.base_payload,
            toolCall={
                "name": "define_subagent",
                "args": {
                    "name": "scout",
                    "enable_write_tools": True,
                    "enable_subagent_tools": False,
                },
            },
        )
        res_def2 = orchestrator_guard.handle_pre_tool(define_snake)
        self.assertEqual(res_def2["decision"], "allow")
        self.assertEqual(res_def2["overwrite"], {"enable_write_tools": False})

        scout_cid = "scout-readonly-worker-01"
        orchestrator_guard.register_subagent(
            self.base_payload,
            conversation_id=scout_cid,
            type_name="scout",
            model="flash_lite",
        )
        scratch_md = str(Path(self.artifact_dir) / "scratch" / "notes.md")
        res_scout_write = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                conversationId=scout_cid,
                toolCall={
                    "name": "write_file",
                    "args": {"Path": scratch_md, "Content": "hello"},
                },
            )
        )
        self.assertEqual(res_scout_write["decision"], "deny")
        self.assertIn("Scout subagents", res_scout_write["reason"])

    def test_blocked_or_failed_subagent_output_does_not_clear_dirty_receipt(self) -> None:
        state = orchestrator_guard._load_state(self.base_payload)
        receipt = orchestrator_guard._get_test_receipt(state)
        receipt["dirty"] = True
        receipt["verified"] = False
        orchestrator_guard._save_state(self.base_payload, state)

        blocked_report = (
            "STATUS: BLOCKED\n"
            "FILES_CHANGED: NONE\n"
            "VERIFICATION: pytest failed twice.\n"
            "NOTES: Failing test in auth module at file:///tmp/scratch/fail.log"
        )
        orchestrator_guard.handle_post_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "impl_blocked_call",
                    "name": "invoke_subagent",
                    "args": {"TypeName": "implementer"},
                },
                output=blocked_report,
                error="",
            )
        )
        state_after = orchestrator_guard._load_state(self.base_payload)
        receipt_after = orchestrator_guard._get_test_receipt(state_after)
        self.assertTrue(receipt_after["dirty"])
        self.assertFalse(receipt_after["verified"])

    def test_prune_state_bounds_deduplication_lists_and_cli_reset(self) -> None:
        state = orchestrator_guard._load_state(self.base_payload)
        limit = orchestrator_guard.MAX_STATE_LIST_ENTRIES
        state["isolation_reminded_invocations"] = [f"inv:{i}" for i in range(limit + 25)]
        state["recorded_spawn_calls"] = [f"spawn:{i}" for i in range(limit + 25)]
        state["stop_reminded_executions"] = [f"stop:{i}" for i in range(limit + 25)]
        cb = orchestrator_guard._get_circuit_breaker(state)
        cb["seen_failures"] = [f"fail:{i}" for i in range(limit + 25)]
        orchestrator_guard._save_state(self.base_payload, state)

        reloaded = orchestrator_guard._load_state(self.base_payload)
        self.assertEqual(len(reloaded["isolation_reminded_invocations"]), limit)
        self.assertEqual(len(reloaded["recorded_spawn_calls"]), limit)
        self.assertEqual(len(reloaded["stop_reminded_executions"]), limit)
        self.assertEqual(
            len(orchestrator_guard._get_circuit_breaker(reloaded)["seen_failures"]),
            limit,
        )
        self.assertEqual(
            reloaded["recorded_spawn_calls"][-1], f"spawn:{limit + 24}"
        )

        reset_res = orchestrator_guard._cli_reset(self.base_payload)
        self.assertEqual(reset_res["status"], "reset")
        after_reset = orchestrator_guard._load_state(self.base_payload)
        self.assertEqual(after_reset["recorded_spawn_calls"], [])

    def test_cross_brain_subagent_receipt_propagation_and_cb_invocation_reset(
        self,
    ) -> None:
        app_data = Path(self.artifact_dir) / "appdata_receipt_prop"
        parent_brain = app_data / "brain" / "parent-root-cid"
        sub_brain = app_data / "brain" / "sub-impl-cid"
        parent_brain.mkdir(parents=True, exist_ok=True)
        sub_brain.mkdir(parents=True, exist_ok=True)

        parent_payload = dict(
            self.base_payload,
            conversationId="parent-root-cid",
            artifactDirectoryPath=str(parent_brain),
            invocationNum=2,
        )
        orchestrator_guard.handle_pre_invocation(parent_payload)
        orchestrator_guard.register_subagent(
            parent_payload,
            conversation_id="sub-impl-cid",
            type_name="implementer",
            model="flash",
            role="Cross-Brain Implementer",
        )

        code_path = str(orchestrator_guard.PROJECT_ROOT / "src_cross_brain.py")
        sub_edit_payload = dict(
            self.base_payload,
            conversationId="sub-impl-cid",
            artifactDirectoryPath=str(sub_brain),
            toolCall={
                "name": "write_file",
                "args": {"Path": code_path, "Content": "x = 1\n"},
            },
        )
        self.assertEqual(
            orchestrator_guard.handle_pre_tool(sub_edit_payload)["decision"], "allow"
        )

        parent_state = orchestrator_guard._load_state(parent_payload)
        parent_receipt = orchestrator_guard._get_test_receipt(parent_state)
        self.assertTrue(parent_receipt["dirty"])
        self.assertFalse(parent_receipt["verified"])

        parent_stop_1 = dict(
            parent_payload,
            fullyIdle=True,
            terminationReason="NO_TOOL_CALL",
            executionNum=1,
        )
        res_stop_1 = orchestrator_guard.handle_stop(parent_stop_1)
        self.assertEqual(res_stop_1["decision"], "continue")
        self.assertIn("Empirical test receipt missing", res_stop_1["reason"])

        sub_test_post = dict(
            self.base_payload,
            conversationId="sub-impl-cid",
            artifactDirectoryPath=str(sub_brain),
            toolCall={
                "id": "sub_pytest_pass",
                "name": "run_command",
                "args": {
                    "CommandLine": "pytest tests/",
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                },
            },
            error="",
        )
        orchestrator_guard.handle_post_tool(sub_test_post)

        parent_state_2 = orchestrator_guard._load_state(parent_payload)
        parent_receipt_2 = orchestrator_guard._get_test_receipt(parent_state_2)
        self.assertFalse(parent_receipt_2["dirty"])
        self.assertTrue(parent_receipt_2["verified"])

        parent_stop_2 = dict(parent_stop_1, executionNum=2)
        self.assertEqual(
            orchestrator_guard.handle_stop(parent_stop_2)["decision"], "allow"
        )

        # Trip circuit breaker on invocation 2, then verify advancing to invocation 3 resets it
        cb = orchestrator_guard._get_circuit_breaker(parent_state_2)
        cb["failure_streak"] = 2
        cb["tripped"] = True
        cb["last_error"] = "failed twice on inv 2"
        orchestrator_guard._save_state(parent_payload, parent_state_2)

        parent_inv3 = dict(parent_payload, invocationNum=3)
        orchestrator_guard.handle_pre_invocation(parent_inv3)
        parent_state_3 = orchestrator_guard._load_state(parent_inv3)
        cb_3 = orchestrator_guard._get_circuit_breaker(parent_state_3)
        self.assertFalse(cb_3["tripped"])
        self.assertEqual(cb_3["failure_streak"], 0)

    def test_allowed_files_header_enforces_subagent_file_allowlist(self) -> None:
        app_data = Path(self.artifact_dir) / "appdata_allowed_files"
        parent_brain = app_data / "brain" / "parent-allow-cid"
        sub1_brain = app_data / "brain" / "sub-allow-1"
        sub2_brain = app_data / "brain" / "sub-allow-2"
        parent_brain.mkdir(parents=True, exist_ok=True)
        sub1_brain.mkdir(parents=True, exist_ok=True)
        sub2_brain.mkdir(parents=True, exist_ok=True)

        prompt_1 = (
            "TASK: Implement feature A.\n"
            "ALLOWED_FILES: module_a.py, tests/test_a.py\n\n"
            "RULES:\n"
            "1. Open `module_a.py`.\n"
            "2. Do not change any file outside `ALLOWED_FILES`.\n\n"
            "REPORT FORMAT:\n"
            "STATUS: COMPLETED | BLOCKED | FAILED\n"
            "FILES_CHANGED: module_a.py\n"
            "VERIFICATION: pytest\n"
            "NOTES: done\n"
        )
        prompt_2 = (
            "TASK: Implement feature B.\n"
            "ALLOWED_FILES: module_b.py\n\n"
            "RULES:\n"
            "1. Open `module_b.py`.\n"
            "2. Do not change any file outside `ALLOWED_FILES`.\n\n"
            "REPORT FORMAT:\n"
            "STATUS: COMPLETED | BLOCKED | FAILED\n"
            "FILES_CHANGED: module_b.py\n"
            "VERIFICATION: pytest\n"
            "NOTES: done\n"
        )

        parent_payload = dict(
            self.base_payload,
            conversationId="parent-allow-cid",
            artifactDirectoryPath=str(parent_brain),
            toolCall={
                "id": "spawn_allow_pair",
                "name": "invoke_subagent",
                "args": {
                    "Subagents": [
                        {
                            "TypeName": "implementer",
                            "Role": "Worker A",
                            "Model": "flash",
                            "Prompt": prompt_1,
                        },
                        {
                            "TypeName": "implementer",
                            "Role": "Worker B",
                            "Model": "flash",
                            "Prompt": prompt_2,
                        },
                    ]
                },
            },
        )
        self.assertEqual(
            orchestrator_guard.handle_pre_tool(parent_payload)["decision"], "allow"
        )

        mod_a = str(orchestrator_guard.PROJECT_ROOT / "module_a.py")
        mod_b = str(orchestrator_guard.PROJECT_ROOT / "module_b.py")
        mod_c = str(orchestrator_guard.PROJECT_ROOT / "module_c.py")
        scratch_log = str(sub1_brain / "scratch" / "fail.log")

        # Subagent 1 is allowed to edit module_a.py and write scratch/fail.log, but denied on module_b.py
        res_sub1_ok = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                conversationId="sub-allow-1",
                artifactDirectoryPath=str(sub1_brain),
                toolCall={
                    "name": "write_file",
                    "args": {"Path": mod_a, "Content": "a = 1\n"},
                },
            )
        )
        self.assertEqual(res_sub1_ok["decision"], "allow")

        res_sub1_scratch = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                conversationId="sub-allow-1",
                artifactDirectoryPath=str(sub1_brain),
                toolCall={
                    "name": "write_to_file",
                    "args": {"TargetFile": scratch_log, "CodeContent": "error\n"},
                },
            )
        )
        self.assertEqual(res_sub1_scratch["decision"], "allow")

        res_sub1_deny = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                conversationId="sub-allow-1",
                artifactDirectoryPath=str(sub1_brain),
                toolCall={
                    "name": "replace_file_content",
                    "args": {"TargetFile": mod_b},
                },
            )
        )
        self.assertEqual(res_sub1_deny["decision"], "deny")
        self.assertIn("File allowlist violation", res_sub1_deny["reason"])

        # Subagent 2 is allowed to edit module_b.py, but denied on module_a.py
        res_sub2_ok = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                conversationId="sub-allow-2",
                artifactDirectoryPath=str(sub2_brain),
                toolCall={
                    "name": "write_file",
                    "args": {"Path": mod_b, "Content": "b = 2\n"},
                },
            )
        )
        self.assertEqual(res_sub2_ok["decision"], "allow")

        res_sub2_deny = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                conversationId="sub-allow-2",
                artifactDirectoryPath=str(sub2_brain),
                toolCall={
                    "name": "write_file",
                    "args": {"Path": mod_a, "Content": "a = 2\n"},
                },
            )
        )
        self.assertEqual(res_sub2_deny["decision"], "deny")
        self.assertIn("File allowlist violation", res_sub2_deny["reason"])

        # send_message updates sub-allow-1's ALLOWED_FILES to include module_c.py
        send_update = dict(
            self.base_payload,
            conversationId="parent-allow-cid",
            artifactDirectoryPath=str(parent_brain),
            toolCall={
                "name": "send_message",
                "args": {
                    "Recipient": "sub-allow-1",
                    "Message": (
                        "TASK: Also update module_c.py.\n"
                        "ALLOWED_FILES: module_a.py, module_c.py\n"
                    ),
                },
            },
        )
        self.assertEqual(
            orchestrator_guard.handle_pre_tool(send_update)["decision"], "allow"
        )

        res_sub1_mod_c = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                conversationId="sub-allow-1",
                artifactDirectoryPath=str(sub1_brain),
                toolCall={
                    "name": "write_file",
                    "args": {"Path": mod_c, "Content": "c = 3\n"},
                },
            )
        )
        self.assertEqual(res_sub1_mod_c["decision"], "allow")

        # ALLOWED_FILES: NONE blocks all workspace edits while allowing scratch writes
        send_none = dict(
            self.base_payload,
            conversationId="parent-allow-cid",
            artifactDirectoryPath=str(parent_brain),
            toolCall={
                "name": "send_message",
                "args": {
                    "Recipient": "sub-allow-1",
                    "Message": "TASK: Read-only check.\nALLOWED_FILES: NONE\n",
                },
            },
        )
        self.assertEqual(
            orchestrator_guard.handle_pre_tool(send_none)["decision"], "allow"
        )

        res_sub1_none_deny = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                conversationId="sub-allow-1",
                artifactDirectoryPath=str(sub1_brain),
                toolCall={
                    "name": "write_file",
                    "args": {"Path": mod_a, "Content": "a = 99\n"},
                },
            )
        )
        self.assertEqual(res_sub1_none_deny["decision"], "deny")
        self.assertIn("NONE", res_sub1_none_deny["reason"])

    def test_thin_entrypoint_line_budget_and_modular_guard_exports(self) -> None:
        import guard
        guard_py = SCRIPT_DIR / "orchestrator_guard.py"
        lines = guard_py.read_text(encoding="utf-8").splitlines()
        self.assertLess(len(lines), 80)
        self.assertGreater(len(lines), 10)
        for sym in guard.__all__:
            self.assertTrue(
                hasattr(orchestrator_guard, sym),
                f"Symbol '{sym}' missing from orchestrator_guard",
            )

    def test_c1_no_wildcard_command_override_and_redirection_bare_root_symlink_guards(
        self,
    ) -> None:
        import os

        cmd = "git status"
        res = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={"name": "run_command", "args": {"CommandLine": cmd}},
            )
        )
        self.assertEqual(res.get("decision"), "allow")
        overrides = res.get("permissionOverrides", [])
        self.assertNotIn("command(*)", overrides)
        self.assertIn(f"command({cmd})", overrides)
        self.assertTrue(any(str(o).startswith("file(") for o in overrides))

        roots = [Path(self.artifact_dir).resolve()]
        isolated_payload = dict(
            self.base_payload,
            workspacePaths=[str(self.artifact_dir)],
        )

        self.assertFalse(
            orchestrator_guard._command_stays_in_workspace(
                "cat >/etc/passwd", cwd=str(self.artifact_dir), roots=roots
            )
        )
        self.assertFalse(
            orchestrator_guard._command_stays_in_workspace(
                "echo hi>>/etc/hosts", cwd=str(self.artifact_dir), roots=roots
            )
        )
        res_cat = orchestrator_guard.handle_pre_tool(
            dict(
                isolated_payload,
                toolCall={
                    "name": "run_command",
                    "args": {
                        "CommandLine": "cat >/etc/passwd",
                        "Cwd": str(self.artifact_dir),
                    },
                },
            )
        )
        self.assertEqual(res_cat.get("decision"), "ask")

        self.assertFalse(
            orchestrator_guard._command_stays_in_workspace(
                "ls /", cwd=str(self.artifact_dir), roots=roots
            )
        )
        res_root = orchestrator_guard.handle_pre_tool(
            dict(
                isolated_payload,
                toolCall={
                    "name": "run_command",
                    "args": {"CommandLine": "ls /", "Cwd": str(self.artifact_dir)},
                },
            )
        )
        self.assertEqual(res_root.get("decision"), "ask")

        self.assertFalse(
            orchestrator_guard._command_stays_in_workspace(
                "ls $HOME", cwd=str(self.artifact_dir), roots=roots
            )
        )
        res_home = orchestrator_guard.handle_pre_tool(
            dict(
                isolated_payload,
                toolCall={
                    "name": "run_command",
                    "args": {"CommandLine": "ls $HOME", "Cwd": str(self.artifact_dir)},
                },
            )
        )
        self.assertEqual(res_home.get("decision"), "ask")

        oldpwd_orig = os.environ.get("OLDPWD")
        try:
            os.environ["OLDPWD"] = "/var/log"
            self.assertFalse(
                orchestrator_guard._command_stays_in_workspace(
                    "ls $OLDPWD", cwd=str(self.artifact_dir), roots=roots
                )
            )
            res_oldpwd = orchestrator_guard.handle_pre_tool(
                dict(
                    isolated_payload,
                    toolCall={
                        "name": "run_command",
                        "args": {
                            "CommandLine": "ls $OLDPWD",
                            "Cwd": str(self.artifact_dir),
                        },
                    },
                )
            )
            self.assertEqual(res_oldpwd.get("decision"), "ask")
        finally:
            if oldpwd_orig is not None:
                os.environ["OLDPWD"] = oldpwd_orig
            else:
                os.environ.pop("OLDPWD", None)

        symlink_path = Path(self.artifact_dir) / "link_to_hosts"
        if not symlink_path.exists():
            symlink_path.symlink_to("/etc/hosts")
        self.assertFalse(
            orchestrator_guard._command_stays_in_workspace(
                "cat link_to_hosts", cwd=str(self.artifact_dir), roots=roots
            )
        )
        res_sym = orchestrator_guard.handle_pre_tool(
            dict(
                isolated_payload,
                toolCall={
                    "name": "run_command",
                    "args": {
                        "CommandLine": "cat link_to_hosts",
                        "Cwd": str(self.artifact_dir),
                    },
                },
            )
        )
        self.assertEqual(res_sym.get("decision"), "ask")

    def test_c2_allowed_files_blocks_nested_basename_collision_and_shell_writes(
        self,
    ) -> None:
        orchestrator_guard.register_subagent(
            self.base_payload,
            conversation_id="imp-c2",
            type_name="implementer",
            model="flash",
            allowed_files=["module_a.py"],
        )
        imp_payload = dict(self.base_payload, conversationId="imp-c2")

        res_root_file = orchestrator_guard.handle_pre_tool(
            dict(
                imp_payload,
                toolCall={
                    "name": "write_file",
                    "args": {
                        "Path": str(orchestrator_guard.PROJECT_ROOT / "module_a.py"),
                        "Content": "x = 1\n",
                    },
                },
            )
        )
        self.assertEqual(res_root_file.get("decision"), "allow")

        res_nested_file = orchestrator_guard.handle_pre_tool(
            dict(
                imp_payload,
                toolCall={
                    "name": "write_file",
                    "args": {
                        "Path": str(
                            orchestrator_guard.PROJECT_ROOT
                            / "nested_dir"
                            / "module_a.py"
                        ),
                        "Content": "x = 1\n",
                    },
                },
            )
        )
        self.assertEqual(res_nested_file.get("decision"), "deny")
        self.assertIn("File allowlist violation", res_nested_file.get("reason", ""))

        res_cmd_echo = orchestrator_guard.handle_pre_tool(
            dict(
                imp_payload,
                toolCall={
                    "name": "run_command",
                    "args": {"CommandLine": "echo hacked > module_b.py"},
                },
            )
        )
        self.assertEqual(res_cmd_echo.get("decision"), "deny")
        self.assertIn("File allowlist violation", res_cmd_echo.get("reason", ""))

        res_cmd_rm = orchestrator_guard.handle_pre_tool(
            dict(
                imp_payload,
                toolCall={
                    "name": "run_command",
                    "args": {"CommandLine": "rm module_b.py"},
                },
            )
        )
        self.assertEqual(res_cmd_rm.get("decision"), "deny")
        self.assertIn("File allowlist violation", res_cmd_rm.get("reason", ""))

        orchestrator_guard.register_subagent(
            self.base_payload,
            conversation_id="scout-c2",
            type_name="scout",
            model="flash_lite",
        )
        scout_payload = dict(self.base_payload, conversationId="scout-c2")

        res_scout_write = orchestrator_guard.handle_pre_tool(
            dict(
                scout_payload,
                toolCall={
                    "name": "run_command",
                    "args": {"CommandLine": "echo x > out.txt"},
                },
            )
        )
        self.assertEqual(res_scout_write.get("decision"), "deny")

        res_scout_rm = orchestrator_guard.handle_pre_tool(
            dict(
                scout_payload,
                toolCall={
                    "name": "run_command",
                    "args": {"CommandLine": "rm README.md"},
                },
            )
        )
        self.assertEqual(res_scout_rm.get("decision"), "deny")

        res_scout_git = orchestrator_guard.handle_pre_tool(
            dict(
                scout_payload,
                toolCall={
                    "name": "run_command",
                    "args": {"CommandLine": "git status"},
                },
            )
        )
        self.assertEqual(res_scout_git.get("decision"), "allow")

        res_scout_ls = orchestrator_guard.handle_pre_tool(
            dict(
                scout_payload,
                toolCall={
                    "name": "run_command",
                    "args": {"CommandLine": "ls"},
                },
            )
        )
        self.assertEqual(res_scout_ls.get("decision"), "allow")

    def test_c3_nonzero_command_exit_output_fails_receipt_and_trips_cb_while_allowing_subagent_blocked_report(
        self,
    ) -> None:
        orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "name": "write_file",
                    "args": {
                        "Path": str(orchestrator_guard.PROJECT_ROOT / "code_c3.py"),
                        "Content": "val = 42\n",
                    },
                },
            )
        )
        state_after_write = orchestrator_guard._load_state(self.base_payload)
        receipt_after_write = orchestrator_guard._get_test_receipt(state_after_write)
        self.assertTrue(receipt_after_write.get("dirty"))

        call_id1 = "post-call-c3-1"
        orchestrator_guard.handle_post_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": call_id1,
                    "name": "run_command",
                    "args": {"CommandLine": "pytest tests/"},
                },
                error="",
                output="Command exited with code 1\n1 failed",
            )
        )
        state_post1 = orchestrator_guard._load_state(self.base_payload)
        rec1 = orchestrator_guard._get_test_receipt(state_post1)
        cb1 = orchestrator_guard._get_circuit_breaker(state_post1)
        self.assertTrue(rec1.get("dirty"))
        self.assertFalse(rec1.get("verified"))
        self.assertEqual(cb1.get("failure_streak"), 1)

        call_id2 = "post-call-c3-2"
        orchestrator_guard.handle_post_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": call_id2,
                    "name": "run_command",
                    "args": {"CommandLine": "pytest tests/"},
                },
                error="",
                output="Command exited with code 2\nfatal error",
            )
        )
        state_post2 = orchestrator_guard._load_state(self.base_payload)
        cb2 = orchestrator_guard._get_circuit_breaker(state_post2)
        self.assertTrue(cb2.get("tripped"))

        res_root_send = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "name": "send_message",
                    "args": {"Recipient": "sub-worker", "Message": "try again"},
                },
            )
        )
        self.assertEqual(res_root_send.get("decision"), "deny")

        orchestrator_guard.register_subagent(
            self.base_payload,
            conversation_id="sub-c3",
            type_name="implementer",
            model="flash",
        )
        sub_payload = dict(self.base_payload, conversationId="sub-c3")
        res_sub_blocked = orchestrator_guard.handle_pre_tool(
            dict(
                sub_payload,
                toolCall={
                    "name": "send_message",
                    "args": {
                        "Recipient": self.base_payload["conversationId"],
                        "Message": (
                            "STATUS: BLOCKED\n"
                            "FILES_CHANGED: NONE\n"
                            "VERIFICATION: pytest tests/ exited with 1\n"
                            f"NOTES: Tests failed repeatedly file://{self.artifact_dir}/scratch/fail.log\n"
                        ),
                    },
                },
            )
        )
        self.assertEqual(res_sub_blocked.get("decision"), "allow")

    def test_c4_cross_brain_isolation_preserves_unrelated_active_sessions(
        self,
    ) -> None:
        base_dir = Path(self.artifact_dir)
        ws_a = base_dir / "ws_a"
        ws_b = base_dir / "ws_b"
        ws_a.mkdir(parents=True, exist_ok=True)
        ws_b.mkdir(parents=True, exist_ok=True)

        brain_dir = base_dir / "brain"
        brain_a = brain_dir / "session-a"
        brain_b = brain_dir / "session-b"
        brain_sub = brain_dir / "sub-a"
        brain_a.mkdir(parents=True, exist_ok=True)
        brain_b.mkdir(parents=True, exist_ok=True)
        brain_sub.mkdir(parents=True, exist_ok=True)

        payload_a = {
            "conversationId": "session-a",
            "artifactDirectoryPath": str(brain_a),
            "workspacePaths": [str(ws_a.resolve())],
        }
        payload_b = {
            "conversationId": "session-b",
            "artifactDirectoryPath": str(brain_b),
            "workspacePaths": [str(ws_b.resolve())],
        }

        st_a = orchestrator_guard._load_state(payload_a)
        st_a["active"] = True
        st_a["session_evaluated"] = True
        st_a["root_conversation_id"] = "session-a"
        st_a["workspace_roots"] = [str(ws_a.resolve())]
        st_a["test_receipt"] = {"dirty": True, "verified": False}
        st_a["pending_spawns"] = [
            {"call_id": "spawn-a", "type": "implementer", "model": "flash"}
        ]
        orchestrator_guard._save_state(payload_a, st_a)

        st_b = orchestrator_guard._load_state(payload_b)
        st_b["active"] = True
        st_b["session_evaluated"] = True
        st_b["root_conversation_id"] = "session-b"
        st_b["workspace_roots"] = [str(ws_b.resolve())]
        st_b["test_receipt"] = {"dirty": True, "verified": False}
        st_b["pending_spawns"] = [
            {"call_id": "spawn-b", "type": "implementer", "model": "flash"}
        ]
        orchestrator_guard._save_state(payload_b, st_b)

        sub_payload = {
            "conversationId": "sub-a",
            "workspacePaths": [str(ws_a.resolve())],
            "artifactDirectoryPath": str(brain_sub),
            "modelName": "Gemini 4 Flash",
            "invocationNum": 1,
        }
        st_sub = orchestrator_guard._load_state(sub_payload)
        self.assertTrue(orchestrator_guard._is_known_subagent(sub_payload, st_sub))

        post_passing = dict(
            sub_payload,
            toolCall={
                "name": "run_command",
                "args": {"CommandLine": "pytest tests/test_feature.py"},
            },
            error="",
            output="1 passed in 0.02s",
        )
        orchestrator_guard.handle_post_tool(post_passing)

        st_a_final = orchestrator_guard._load_state(payload_a)
        rec_a = orchestrator_guard._get_test_receipt(st_a_final)
        self.assertFalse(rec_a.get("dirty"))
        self.assertTrue(rec_a.get("verified"))

        st_b_final = orchestrator_guard._load_state(payload_b)
        rec_b = orchestrator_guard._get_test_receipt(st_b_final)
        self.assertTrue(rec_b.get("dirty"))
        self.assertFalse(rec_b.get("verified"))
        self.assertEqual(len(st_b_final.get("pending_spawns", [])), 1)
        self.assertEqual(st_b_final["pending_spawns"][0].get("call_id"), "spawn-b")

    def test_h1_multiprocess_concurrent_state_transactions_and_corrupt_recovery(
        self,
    ) -> None:
        import subprocess

        worker_code = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import orchestrator_guard
payload = {"artifactDirectoryPath": sys.argv[2], "conversationId": "root-mp"}
orchestrator_guard.register_subagent(
    payload,
    conversation_id=sys.argv[3],
    type_name="scout",
    model="flash_lite",
)
"""
        procs = []
        for i in range(6):
            p = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    worker_code,
                    str(SCRIPT_DIR),
                    self.artifact_dir,
                    f"worker-{i}",
                ]
            )
            procs.append(p)

        for p in procs:
            exit_code = p.wait()
            self.assertEqual(exit_code, 0)

        final_state = orchestrator_guard._load_state(
            {"artifactDirectoryPath": self.artifact_dir, "conversationId": "root-mp"}
        )
        self.assertEqual(len(final_state.get("subagents", {})), 6)

        corrupt_temp = tempfile.TemporaryDirectory()
        try:
            corrupt_payload = {
                "artifactDirectoryPath": corrupt_temp.name,
                "conversationId": "corrupt-cid",
            }
            corrupt_file = orchestrator_guard._state_file_path(corrupt_payload)
            corrupt_file.write_text("{corrupt", encoding="utf-8")
            recovered = orchestrator_guard._load_state(corrupt_payload)
            self.assertTrue(recovered.get("active"))
            self.assertTrue(recovered.get("session_evaluated"))
        finally:
            corrupt_temp.cleanup()

    def test_h2_async_spawn_ack_keeps_dirty_receipt_until_verified(self) -> None:
        orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "name": "write_file",
                    "args": {
                        "Path": str(orchestrator_guard.PROJECT_ROOT / "mut_h2.py"),
                        "Content": "y = 99\n",
                    },
                },
            )
        )
        st_pre = orchestrator_guard._load_state(self.base_payload)
        rec_pre = orchestrator_guard._get_test_receipt(st_pre)
        self.assertTrue(rec_pre.get("dirty"))

        ack_output = (
            'Created the following subagent:\n'
            '{"conversationId": "12345678-1234-1234-1234-1234567890ab"}'
        )
        orchestrator_guard.handle_post_tool(
            dict(
                self.base_payload,
                toolCall={
                    "name": "invoke_subagent",
                    "args": {"TypeName": "implementer"},
                },
                output=ack_output,
            )
        )

        st_post = orchestrator_guard._load_state(self.base_payload)
        rec_post = orchestrator_guard._get_test_receipt(st_post)
        self.assertTrue(rec_post.get("dirty"))
        self.assertFalse(rec_post.get("verified"))

    def test_h3_deleted_test_file_diff_multireplace_and_block_comment_tampering(
        self,
    ) -> None:
        deleted_diff = (
            "--- a/tests/test_old.py\n"
            "+++ /dev/null\n"
            "@@ -1,2 +0,0 @@\n"
            "-def test_old():\n"
            "-    assert 1 == 1\n"
        )
        verdict = orchestrator_guard.audit_diff_text(deleted_diff)
        self.assertIsNotNone(verdict)
        self.assertIn("Diff audit detected", verdict or "")

        res_multi = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "name": "multi_replace_file_content",
                    "args": {
                        "TargetFile": "tests/test_x.py",
                        "ReplacementChunks": [
                            {
                                "TargetContent": "def test_a():\n    assert a == 1\n",
                                "ReplacementContent": "def test_a():\n    pass\n",
                            }
                        ],
                    },
                },
            )
        )
        self.assertEqual(res_multi.get("decision"), "deny")
        self.assertIn("Test tampering blocked", res_multi.get("reason", ""))

        res_docstring = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "name": "edit_file",
                    "args": {
                        "Path": "tests/test_x.py",
                        "Target": "def test_b():\n    assert x == 1\n",
                        "Replacement": 'def test_b():\n    """\n    assert x == 1\n    """\n',
                    },
                },
            )
        )
        self.assertEqual(res_docstring.get("decision"), "deny")
        self.assertIn("Test tampering blocked", res_docstring.get("reason", ""))

        res_c_comment = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "name": "edit_file",
                    "args": {
                        "Path": "tests/test_x.py",
                        "Target": "def test_b():\n    assert x == 1\n",
                        "Replacement": "def test_b():\n    /* assert x == 1 */\n",
                    },
                },
            )
        )
        self.assertEqual(res_c_comment.get("decision"), "deny")
        self.assertIn("Test tampering blocked", res_c_comment.get("reason", ""))

    def test_h4_h5_m3_subagent_send_message_validation_unsynced_allowlist_and_manage_subagents(
        self,
    ) -> None:
        brain_base = Path(self.artifact_dir) / "brain"
        parent_brain = brain_base / "parent-h4"
        sub_brain = brain_base / "sub-h4"
        parent_brain.mkdir(parents=True, exist_ok=True)
        sub_brain.mkdir(parents=True, exist_ok=True)

        parent_payload = dict(
            self.base_payload,
            conversationId="parent-h4",
            artifactDirectoryPath=str(parent_brain),
        )
        orchestrator_guard.register_subagent(
            parent_payload,
            conversation_id="sub-h4",
            type_name="implementer",
            model="flash",
        )

        sub_payload = dict(
            self.base_payload,
            conversationId="sub-h4",
            artifactDirectoryPath=str(sub_brain),
            modelName="Gemini 4 Flash",
        )

        bad_report = "STATUS: COMPLETED\nFILES_CHANGED: NONE\n"
        res_bad = orchestrator_guard.handle_pre_tool(
            dict(
                sub_payload,
                toolCall={
                    "name": "send_message",
                    "args": {"Recipient": "parent-h4", "Message": bad_report},
                },
            )
        )
        self.assertEqual(res_bad.get("decision"), "deny")
        self.assertIn("Subagent report format error", res_bad.get("reason", ""))

        good_report = (
            "STATUS: COMPLETED\n"
            "FILES_CHANGED: NONE\n"
            "VERIFICATION: pytest tests/ passed\n"
            "NOTES: Finished all tasks cleanly.\n"
        )
        res_good = orchestrator_guard.handle_pre_tool(
            dict(
                sub_payload,
                toolCall={
                    "name": "send_message",
                    "args": {"Recipient": "parent-h4", "Message": good_report},
                },
            )
        )
        self.assertEqual(res_good.get("decision"), "allow")

        res_stop = orchestrator_guard.handle_stop(sub_payload)
        self.assertEqual(res_stop.get("decision"), "allow")

        unsynced_brain = brain_base / "unsynced-recipient"
        unsynced_brain.mkdir(parents=True, exist_ok=True)
        unsynced_payload = {
            "conversationId": "unsynced-recipient",
            "artifactDirectoryPath": str(unsynced_brain),
        }
        orchestrator_guard._save_state(
            unsynced_payload, orchestrator_guard._load_state(unsynced_payload)
        )

        msg_allowed = (
            "TASK: Refactor module\n"
            "ALLOWED_FILES: updated.py\n"
            "RULES:\n1. Open updated.py\n"
            "REPORT FORMAT:\nSTATUS: COMPLETED\n"
        )
        orchestrator_guard.handle_pre_tool(
            dict(
                parent_payload,
                toolCall={
                    "name": "send_message",
                    "args": {
                        "Recipient": "unsynced-recipient",
                        "Message": msg_allowed,
                    },
                },
            )
        )
        unsynced_st = orchestrator_guard._load_state(unsynced_payload)
        self.assertEqual(
            unsynced_st.get("self_subagent_meta", {}).get("allowed_files"),
            ["updated.py"],
        )

        def_res = orchestrator_guard._handle_define_subagent(
            {"Name": "implementer", "EnableMCPTools": True}
        )
        self.assertEqual(def_res.get("decision"), "allow")
        self.assertEqual(def_res.get("overwrite"), {"EnableMCPTools": False})

        simple_prompt = "1. Open a.py\n2. Read x\n3. Do not edit any file"
        self.assertTrue(
            orchestrator_guard._is_simple_readonly_task("research", simple_prompt)
        )

        st_manage = orchestrator_guard._load_state(parent_payload)
        st_manage["subagents"] = {
            "branch-1": {"active_branch": True},
            "branch-2": {"active_branch": True},
        }
        orchestrator_guard._save_state(parent_payload, st_manage)

        orchestrator_guard._handle_manage_subagents(
            parent_payload, {"action": "kill", "conversation_ids": ["branch-1"]}
        )
        st_m1 = orchestrator_guard._load_state(parent_payload)
        self.assertFalse(st_m1["subagents"]["branch-1"]["active_branch"])
        self.assertTrue(st_m1["subagents"]["branch-2"]["active_branch"])

        orchestrator_guard._handle_manage_subagents(
            parent_payload, {"Action": "kill_all"}
        )
        st_m2 = orchestrator_guard._load_state(parent_payload)
        self.assertFalse(st_m2["subagents"]["branch-2"]["active_branch"])

    def test_l1_cli_main_subprocess_modes_transcript_tail_and_git_diff_fallback(
        self,
    ) -> None:
        import subprocess

        tpath = Path(self.artifact_dir) / "transcript.jsonl"
        with open(tpath, "w", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"type": "USER_MESSAGE", "content": "hello worker"}) + "\n"
            )
            fh.write(
                json.dumps(
                    {
                        "type": "PLANNER_RESPONSE",
                        "content": "STATUS: COMPLETED\nFILES_CHANGED: NONE\nVERIFICATION: pass\nNOTES: done\n",
                        "tool_calls": [],
                    }
                )
                + "\n"
            )
        extracted = orchestrator_guard._extract_last_model_response(
            {"transcriptPath": str(tpath)}
        )
        self.assertIsNotNone(extracted)
        self.assertIn("STATUS: COMPLETED", extracted or "")

        diff_text = orchestrator_guard._get_workspace_git_diff(
            {"workspacePaths": [str(orchestrator_guard.PROJECT_ROOT)]}
        )
        self.assertIsInstance(diff_text, str)

        entrypoint = SCRIPT_DIR / "orchestrator_guard.py"
        cli_modes = [
            ("doctor", {}),
            ("activate", self.base_payload),
            ("status", self.base_payload),
            ("pre-invocation", self.base_payload),
            (
                "pre-tool",
                dict(
                    self.base_payload,
                    toolCall={
                        "name": "read_file",
                        "args": {
                            "Path": str(orchestrator_guard.PROJECT_ROOT / "README.md")
                        },
                    },
                ),
            ),
            (
                "pre-tool",
                dict(
                    self.base_payload,
                    toolCall={
                        "name": "write_file",
                        "args": {
                            "Path": str(Path(self.artifact_dir) / "scratch.txt"),
                            "Content": "test\n",
                        },
                    },
                ),
            ),
            (
                "pre-tool",
                dict(
                    self.base_payload,
                    toolCall={
                        "name": "run_command",
                        "args": {"CommandLine": "git status"},
                    },
                ),
            ),
            (
                "post-tool",
                dict(
                    self.base_payload,
                    toolCall={
                        "name": "run_command",
                        "args": {"CommandLine": "git status"},
                    },
                ),
            ),
            ("stop", self.base_payload),
            ("deactivate", self.base_payload),
            ("reset", self.base_payload),
        ]
        for mode, payload in cli_modes:
            res = subprocess.run(
                [sys.executable, str(entrypoint), mode],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(res.returncode, 0)
            parsed = json.loads(res.stdout)
            self.assertIsInstance(parsed, dict)

    def test_cli_modes_do_not_hang_on_open_unclosed_stdin_pipe(self) -> None:
        import os
        import subprocess

        entrypoint = SCRIPT_DIR / "orchestrator_guard.py"
        default_state_dir = orchestrator_guard.PROJECT_ROOT / ".agents" / ".state"
        default_state_file = default_state_dir / "orchestrator_state.json"
        default_lock_file = default_state_dir / "orchestrator_state.lock"
        had_state = default_state_file.exists()
        orig_state_bytes = default_state_file.read_bytes() if had_state else None
        had_lock = default_lock_file.exists()

        r_fd, w_fd = os.pipe()
        try:
            for mode in ("status", "activate", "deactivate", "reset"):
                res = subprocess.run(
                    [sys.executable, str(entrypoint), mode],
                    stdin=r_fd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=3,
                    check=True,
                )
                self.assertEqual(res.returncode, 0)
                parsed = json.loads(res.stdout)
                self.assertIsInstance(parsed, dict)
                if mode == "status":
                    self.assertEqual(
                        parsed.get("stateFile"), str(default_state_file)
                    )
                elif mode == "activate":
                    self.assertEqual(parsed.get("status"), "activated")
                    self.assertTrue(parsed.get("active"))
                elif mode == "deactivate":
                    self.assertEqual(parsed.get("status"), "deactivated")
                    self.assertFalse(parsed.get("active"))
                elif mode == "reset":
                    self.assertEqual(parsed.get("status"), "reset")
        finally:
            os.close(r_fd)
            os.close(w_fd)
            if had_state and orig_state_bytes is not None:
                default_state_file.write_bytes(orig_state_bytes)
            elif default_state_file.exists():
                default_state_file.unlink()
            if not had_lock and default_lock_file.exists():
                default_lock_file.unlink()

    def test_cli_status_honors_piped_json_and_stringio_stdin(self) -> None:
        import io
        import subprocess
        from unittest import mock
        from guard import cli as guard_cli

        entrypoint = SCRIPT_DIR / "orchestrator_guard.py"
        custom_dir = Path(self.artifact_dir) / "piped_cli_brain"
        custom_dir.mkdir(parents=True, exist_ok=True)
        piped_payload = {"artifactDirectoryPath": str(custom_dir)}

        act_res = subprocess.run(
            [sys.executable, str(entrypoint), "activate"],
            input=json.dumps(piped_payload),
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        self.assertEqual(json.loads(act_res.stdout).get("active"), True)

        status_res = subprocess.run(
            [sys.executable, str(entrypoint), "status"],
            input=json.dumps(piped_payload),
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        status_data = json.loads(status_res.stdout)
        self.assertEqual(
            status_data.get("stateFile"),
            str(custom_dir / "orchestrator_state.json"),
        )
        self.assertTrue(status_data.get("active"))

        stdout_buf = io.StringIO()
        with mock.patch.object(
            sys, "argv", ["orchestrator_guard.py", "status"]
        ), mock.patch.object(
            sys, "stdin", io.StringIO(json.dumps(piped_payload))
        ), mock.patch.object(
            sys, "stdout", stdout_buf
        ):
            guard_cli.main()

        inproc_status = json.loads(stdout_buf.getvalue())
        self.assertEqual(
            inproc_status.get("stateFile"),
            str(custom_dir / "orchestrator_state.json"),
        )
        self.assertTrue(inproc_status.get("active"))

        with mock.patch.object(sys, "stdin", io.StringIO("[1, 2, 3]")):
            self.assertEqual(guard_cli._read_stdin_payload(is_cli_mode=True), {})


if __name__ == "__main__":
    unittest.main()
