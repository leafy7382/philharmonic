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
        self.assertIn("Direct execution fast-path read budget", res_c["reason"])

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
        self.assertIn("Direct execution fast-path read budget", r3["reason"])

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

    def test_replan_status_and_root_replan_circuit_breaker(self) -> None:
        # 1. validate_subagent_report for NEEDS_REPLAN
        valid_replan = (
            "STATUS: NEEDS_REPLAN\n"
            "FILES_CHANGED: NONE\n"
            "VERIFICATION: NONE\n"
            "NOTES: Core assumption about database schema was invalidated."
        )
        self.assertIsNone(orchestrator_guard.validate_subagent_report(valid_replan))

        invalid_replan = (
            "STATUS: NEEDS_REPLAN\n"
            "FILES_CHANGED: NONE\n"
            "VERIFICATION: NONE\n"
            "NOTES: invalid"
        )
        err = orchestrator_guard.validate_subagent_report(invalid_replan)
        self.assertIsNotNone(err)
        self.assertIn("NEEDS_REPLAN", err)
        self.assertIn("at least 3 words", err)

        # 2. Tripped subagent allowed to send STATUS: NEEDS_REPLAN
        sub_cid = "sub-cb-replan-111"
        sub_brain = Path(self.artifact_dir).parent / sub_cid
        sub_brain.mkdir(parents=True, exist_ok=True)
        orchestrator_guard.register_subagent(
            self.base_payload,
            conversation_id=sub_cid,
            type_name="implementer",
            model="flash",
        )
        state = orchestrator_guard._load_state(self.base_payload)
        cb = orchestrator_guard._get_circuit_breaker(state, self.base_payload)
        cb["tripped"] = True
        cb["failure_streak"] = 2
        cb["last_error"] = "Command exited with code 1"
        orchestrator_guard._save_state(self.base_payload, state)

        sub_payload = dict(
            self.base_payload,
            conversationId=sub_cid,
            artifactDirectoryPath=str(sub_brain),
            toolCall={
                "name": "send_message",
                "args": {
                    "Recipient": self.base_payload["conversationId"],
                    "Message": valid_replan,
                },
            },
        )
        res_sub_replan = orchestrator_guard.handle_pre_tool(sub_payload)
        self.assertEqual(res_sub_replan["decision"], "allow")

        # 3. Tripped root denied without REPLAN: header
        root_dispatch_no_replan = dict(
            self.base_payload,
            toolCall={
                "name": "invoke_subagent",
                "args": {
                    "TypeName": "implementer",
                    "Prompt": "TASK: Fix bug\nRULES:\n1. Open foo.py\n2. Run pytest\nREPORT FORMAT:\nSTATUS: COMPLETED\nFILES_CHANGED: foo.py\nVERIFICATION: pytest ok\nNOTES: done",
                },
            },
        )
        res_root_deny = orchestrator_guard.handle_pre_tool(root_dispatch_no_replan)
        self.assertEqual(res_root_deny["decision"], "deny")
        self.assertIn("circuit breaker tripped", res_root_deny["reason"])
        self.assertIn("REPLAN:", res_root_deny["reason"])

        # Tripped root allowed once with REPLAN: header (resets CB)
        root_dispatch_with_replan = dict(
            self.base_payload,
            toolCall={
                "name": "invoke_subagent",
                "args": {
                    "TypeName": "implementer",
                    "Prompt": "TASK: Fix bug with revised strategy\nREPLAN: Switched to migration approach\nRULES:\n1. Open foo.py\n2. Run pytest\nREPORT FORMAT:\nSTATUS: COMPLETED\nFILES_CHANGED: foo.py\nVERIFICATION: pytest ok\nNOTES: done",
                },
            },
        )
        res_root_allow = orchestrator_guard.handle_pre_tool(root_dispatch_with_replan)
        self.assertEqual(res_root_allow["decision"], "allow")
        state_after = orchestrator_guard._load_state(self.base_payload)
        cb_after = orchestrator_guard._get_circuit_breaker(state_after, self.base_payload)
        self.assertFalse(cb_after["tripped"])
        self.assertEqual(cb_after["failure_streak"], 0)
        self.assertEqual(cb_after["replan_count"], 1)

        # Trip CB again
        cb_after["tripped"] = True
        cb_after["failure_streak"] = 2
        orchestrator_guard._save_state(self.base_payload, state_after)

        # Denied on 2nd trip because replan_count >= MAX_REPLAN_ATTEMPTS (1)
        res_root_deny2 = orchestrator_guard.handle_pre_tool(root_dispatch_with_replan)
        self.assertEqual(res_root_deny2["decision"], "deny")
        self.assertIn("circuit breaker tripped", res_root_deny2["reason"])

    def test_line_byte_read_budget_and_list_dir_exemption(self) -> None:
        file_paths = []
        for i in range(5):
            p = orchestrator_guard.PROJECT_ROOT / f"tmp_small_{i}.txt"
            p.write_text("line 1\nline 2\nline 3\nline 4\nline 5\n", encoding="utf-8")
            file_paths.append(str(p))

        try:
            # 1. Reading 4 small real files succeeds (5 lines each = 20 lines << 400 lines, 4 files < 8 files)
            for f in file_paths[:4]:
                res = orchestrator_guard.handle_pre_tool(
                    dict(
                        self.base_payload,
                        toolCall={"name": "read_file", "args": {"Path": f}},
                    )
                )
                self.assertEqual(res["decision"], "allow")

            # 2. StartLine / EndLine slicing accumulates accurately
            big_p = orchestrator_guard.PROJECT_ROOT / "tmp_big_sliced.txt"
            big_p.write_text("\n".join(f"line {j}" for j in range(1, 601)) + "\n", encoding="utf-8")
            big_path = str(big_p)
            file_paths.append(big_path)

            # Read 200 lines (lines 1..200) -> allows
            res_slice1 = orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={"name": "read_file", "args": {"Path": big_path, "StartLine": 1, "EndLine": 200}},
                )
            )
            self.assertEqual(res_slice1["decision"], "allow")

            # Read another 200 lines (lines 201..400) -> exceeds 400 lines!
            res_slice2 = orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={"name": "read_file", "args": {"Path": big_path, "StartLine": 201, "EndLine": 400}},
                )
            )
            self.assertEqual(res_slice2["decision"], "deny")
            self.assertIn("Direct execution fast-path read budget", res_slice2["reason"])
            self.assertIn("400 lines", res_slice2["reason"])

            # 3. Reading > 16 KB single file is blocked
            huge_p = orchestrator_guard.PROJECT_ROOT / "tmp_huge_bytes.txt"
            huge_p.write_text("A" * 20000 + "\n", encoding="utf-8")
            huge_path = str(huge_p)
            file_paths.append(huge_path)

            state = orchestrator_guard._load_state(self.base_payload)
            state["direct_budget"].clear()
            orchestrator_guard._save_state(self.base_payload, state)

            res_huge = orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={"name": "read_file", "args": {"Path": huge_path}},
                )
            )
            self.assertEqual(res_huge["decision"], "deny")
            self.assertIn("16384 bytes", res_huge["reason"])

            # 4. list_dir is free (0 budget cost)
            for _ in range(15):
                res_ld = orchestrator_guard.handle_pre_tool(
                    dict(
                        self.base_payload,
                        toolCall={"name": "list_dir", "args": {"Path": str(orchestrator_guard.PROJECT_ROOT)}},
                    )
                )
                self.assertEqual(res_ld["decision"], "allow")

            # But list_dir on .log is blocked
            res_ld_log = orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={"name": "list_dir", "args": {"Path": str(Path(self.artifact_dir) / "scratch" / "run.log")}},
                )
            )
            self.assertEqual(res_ld_log["decision"], "deny")
            self.assertIn("Context separation violation", res_ld_log["reason"])

        finally:
            for fp in file_paths:
                try:
                    p = Path(fp)
                    if p.exists():
                        p.unlink()
                except OSError:
                    pass

    def test_grounded_spec_file_validation(self) -> None:
        spec_path = Path(self.artifact_dir) / "scratch" / "spec_task_test.md"
        spec_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Valid spec with SIGNATURES, PITFALLS/INVARIANTS, and VERIFICATION
        valid_spec_content = (
            "# Grounded Specification\n\n"
            "## SIGNATURES\n"
            "def solve(a: int) -> int: ...\n\n"
            "## INVARIANTS_AND_PITFALLS\n"
            "Keep a > 0.\n\n"
            "## VERIFICATION\n"
            "Run pytest tests/test_solve.py\n"
        )
        spec_path.write_text(valid_spec_content, encoding="utf-8")

        prompt_valid = (
            f"TASK: Implement solve function\n"
            f"SPEC_FILE: {spec_path}\n"
            f"RULES:\n1. Open spec\n2. Run pytest\n"
            f"REPORT FORMAT:\nSTATUS: COMPLETED\nFILES_CHANGED: foo.py\nVERIFICATION: pytest ok\nNOTES: done"
        )
        res_valid = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "name": "invoke_subagent",
                    "args": {"TypeName": "implementer", "Prompt": prompt_valid},
                },
            )
        )
        self.assertEqual(res_valid["decision"], "allow")
        state = orchestrator_guard._load_state(self.base_payload)
        pending = state.get("pending_spawns", [])
        self.assertTrue(any(p.get("spec_file") == str(spec_path.resolve()) for p in pending))

        # 2. Missing spec file is denied for implementer
        missing_spec_prompt = (
            "TASK: Implement feature\n"
            "SPEC_FILE: /tmp/scratch/non_existent_spec_xyz.md\n"
            "RULES:\n1. Open spec\n2. Run pytest\n"
            "REPORT FORMAT:\nSTATUS: COMPLETED\nFILES_CHANGED: foo.py\nVERIFICATION: pytest ok\nNOTES: done"
        )
        res_missing = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "name": "invoke_subagent",
                    "args": {"TypeName": "implementer", "Prompt": missing_spec_prompt},
                },
            )
        )
        self.assertEqual(res_missing["decision"], "deny")
        self.assertIn("does not exist", res_missing["reason"])

        # 3. Spec file missing required sections is denied for implementer
        stub_spec_path = Path(self.artifact_dir) / "scratch" / "spec_stub.md"
        stub_spec_path.write_text("# Incomplete Spec\nJust some notes without sections.\n", encoding="utf-8")
        stub_prompt = (
            f"TASK: Implement feature\n"
            f"SPEC_FILE: {stub_spec_path}\n"
            f"RULES:\n1. Open spec\n2. Run pytest\n"
            f"REPORT FORMAT:\nSTATUS: COMPLETED\nFILES_CHANGED: foo.py\nVERIFICATION: pytest ok\nNOTES: done"
        )
        res_stub = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "name": "invoke_subagent",
                    "args": {"TypeName": "implementer", "Prompt": stub_prompt},
                },
            )
        )
        self.assertEqual(res_stub["decision"], "deny")
        self.assertIn("missing required grounded sections", res_stub["reason"])

    def test_staged_checkpoints_and_checkpoint_status(self) -> None:
        state = orchestrator_guard._load_state(self.base_payload)
        receipt = orchestrator_guard._get_test_receipt(state, self.base_payload)
        receipt["dirty"] = True
        receipt["verified"] = False
        orchestrator_guard._save_state(self.base_payload, state)

        # STAGE: 1/2 in invoke_subagent
        stage_prompt = (
            "TASK: Multi-stage refactor\n"
            "ALLOWED_FILES: foo.py\n"
            "STAGE: 1/2\n"
            "RULES:\n1. Edit foo.py\n2. Run tests\n"
            "REPORT FORMAT:\nSTATUS: COMPLETED\nFILES_CHANGED: foo.py\nVERIFICATION: pytest ok\nNOTES: done"
        )
        res_spawn = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "name": "invoke_subagent",
                    "args": {"TypeName": "implementer", "Prompt": stage_prompt},
                },
            )
        )
        self.assertEqual(res_spawn["decision"], "allow")
        state = orchestrator_guard._load_state(self.base_payload)
        pending = state.get("pending_spawns", [])
        self.assertTrue(any(p.get("stage") == "1/2" for p in pending))

        # STATUS: CHECKPOINT report passes validation
        checkpoint_report = (
            "STATUS: CHECKPOINT\n"
            "FILES_CHANGED: foo.py\n"
            "VERIFICATION: pytest tests/test_foo.py passed\n"
            "NOTES: Completed stage 1; ready for stage 2."
        )
        self.assertIsNone(orchestrator_guard.validate_subagent_report(checkpoint_report))

        # invoke_subagent output with STATUS: CHECKPOINT does NOT prematurely clear dirty receipt
        post_res = orchestrator_guard.handle_post_tool(
            dict(
                self.base_payload,
                toolCall={
                    "name": "invoke_subagent",
                    "args": {"TypeName": "implementer", "Prompt": stage_prompt},
                },
                output=checkpoint_report,
                error="",
            )
        )
        self.assertEqual(post_res, {})
        state_post = orchestrator_guard._load_state(self.base_payload)
        receipt_post = orchestrator_guard._get_test_receipt(state_post, self.base_payload)
        self.assertTrue(receipt_post.get("dirty"))
        self.assertFalse(receipt_post.get("verified"))

        # send_message with STAGE: 2/2 and expanded ALLOWED_FILES:
        sub_cid = "sub-staged-stage2-cid"
        sub_brain = Path(self.artifact_dir).parent / sub_cid
        sub_brain.mkdir(parents=True, exist_ok=True)
        orchestrator_guard.register_subagent(
            self.base_payload,
            conversation_id=sub_cid,
            type_name="implementer",
            model="flash",
            allowed_files=["foo.py"],
        )
        res_msg = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "name": "send_message",
                    "args": {
                        "Recipient": sub_cid,
                        "Message": "STAGE: 2/2\nALLOWED_FILES: foo.py, bar.py\nPlease proceed with stage 2.",
                    },
                },
            )
        )
        self.assertEqual(res_msg["decision"], "allow")
        state_msg = orchestrator_guard._load_state(self.base_payload)
        sub_info = state_msg.get("subagents", {}).get(sub_cid, {})
        self.assertEqual(sub_info.get("stage"), "2/2")
        self.assertIn("2/2", sub_info.get("stage_history", []))
        self.assertEqual(sub_info.get("allowed_files"), ["foo.py", "bar.py"])

    def test_compact_scout_report_shorthand(self) -> None:
        compact_report = (
            "STATUS: COMPLETED\n"
            "NOTES: Found symbol in foo.py"
        )
        # 1. validate_subagent_report allows compact for scout, rejects for implementer
        self.assertIsNone(orchestrator_guard.validate_subagent_report(compact_report, is_scout=True))
        err_impl = orchestrator_guard.validate_subagent_report(compact_report, is_scout=False)
        self.assertIsNotNone(err_impl)
        self.assertIn("Expected 4 header lines", err_impl)

        # 2. scout sending 2-header compact report via send_message is allowed
        brain_base = Path(self.artifact_dir) / "brain"
        parent_brain = brain_base / "parent-scout-test"
        scout_cid = "sub-scout-compact-99"
        impl_cid = "sub-impl-compact-fail"
        scout_brain = brain_base / scout_cid
        impl_brain = brain_base / impl_cid
        parent_brain.mkdir(parents=True, exist_ok=True)
        scout_brain.mkdir(parents=True, exist_ok=True)
        impl_brain.mkdir(parents=True, exist_ok=True)

        parent_payload = dict(
            self.base_payload,
            conversationId="parent-scout-test",
            artifactDirectoryPath=str(parent_brain),
        )
        orchestrator_guard._cli_activate(parent_payload)
        orchestrator_guard.register_subagent(
            parent_payload,
            conversation_id=scout_cid,
            type_name="scout",
            model="flash_lite",
        )
        orchestrator_guard.register_subagent(
            parent_payload,
            conversation_id=impl_cid,
            type_name="implementer",
            model="flash",
        )
        scout_payload = dict(
            self.base_payload,
            conversationId=scout_cid,
            artifactDirectoryPath=str(scout_brain),
            toolCall={
                "name": "send_message",
                "args": {
                    "Recipient": "parent-scout-test",
                    "Message": compact_report,
                },
            },
        )
        res_scout_msg = orchestrator_guard.handle_pre_tool(scout_payload)
        self.assertEqual(res_scout_msg["decision"], "allow")

        # 3. implementer sending 2-header compact report via send_message is denied
        impl_payload = dict(
            self.base_payload,
            conversationId=impl_cid,
            artifactDirectoryPath=str(impl_brain),
            toolCall={
                "name": "send_message",
                "args": {
                    "Recipient": "parent-scout-test",
                    "Message": compact_report,
                },
            },
        )
        res_impl_msg = orchestrator_guard.handle_pre_tool(impl_payload)
        self.assertEqual(res_impl_msg["decision"], "deny")
        self.assertIn("Subagent report format error", res_impl_msg["reason"])

        # 4. handle_stop allows compact report for scout
        stop_payload = dict(
            self.base_payload,
            conversationId=scout_cid,
            artifactDirectoryPath=str(scout_brain),
            lastAssistantMessage=compact_report,
            fullyIdle=True,
        )
        res_stop = orchestrator_guard.handle_stop(stop_payload)
        self.assertEqual(res_stop.get("decision"), "allow")

    def test_system_orchestrator_exit_plan_mode_coexistence(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)
        plan_content = (
            "TASK: Multi-file refactor\n"
            "ALLOWED_FILES: file_x.py, file_y.py, file_z.py\n"
            "1. Edit all three files\n"
            "2. Run tests"
        )
        exit_plan_payload = dict(
            self.base_payload,
            toolCall={
                "name": "exit_plan_mode",
                "args": {"plan": plan_content},
            },
        )
        res_exit = orchestrator_guard.handle_pre_tool(exit_plan_payload)
        self.assertEqual(res_exit["decision"], "allow")

        state = orchestrator_guard._load_state(self.base_payload)
        self.assertTrue(state.get("plan_execute_phase"))
        self.assertFalse(state.get("plan_execute_consumed"))
        self.assertEqual(state.get("plan_allowed_files"), ["file_x.py", "file_y.py", "file_z.py"])

        # In coexistence mode, root can edit > 2 files in plan_allowed_files
        for fname in ["file_x.py", "file_y.py", "file_z.py"]:
            fpath = str(orchestrator_guard.PROJECT_ROOT / fname)
            res_edit = orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={"name": "edit_file", "args": {"Path": fpath, "Target": "", "Replacement": ""}},
                )
            )
            self.assertEqual(res_edit["decision"], "allow")

        # Editing out-of-plan file is denied
        out_fpath = str(orchestrator_guard.PROJECT_ROOT / "out_of_plan.py")
        res_out = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={"name": "edit_file", "args": {"Path": out_fpath, "Target": "", "Replacement": ""}},
            )
        )
        self.assertEqual(res_out["decision"], "deny")
        self.assertIn("File allowlist violation", res_out["reason"])

        # Running multiple commands is allowed
        for cmd in ["echo 1", "echo 2", "echo 3", "echo 4"]:
            res_cmd = orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={"name": "run_command", "args": {"CommandLine": cmd}},
                )
            )
            self.assertEqual(res_cmd["decision"], "allow")

        # But reading .log is STILL blocked in plan_execute_phase
        log_fpath = str(Path(self.artifact_dir) / "scratch" / "test.log")
        res_log = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={"name": "read_file", "args": {"Path": log_fpath}},
            )
        )
        self.assertEqual(res_log["decision"], "deny")
        self.assertIn("Context separation violation", res_log["reason"])

        # PreInvocation resets plan_execute_phase on [plan review]
        review_payload = dict(
            self.base_payload,
            lastUserInput="Please do [plan review] now",
        )
        orchestrator_guard.handle_pre_invocation(review_payload)
        state_after_review = orchestrator_guard._load_state(self.base_payload)
        self.assertFalse(state_after_review.get("plan_execute_phase"))
        self.assertIsNone(state_after_review.get("plan_allowed_files"))

    def test_safe_git_env_prefixes_and_unsafe_env_rejection(self) -> None:
        # Safe git prefixes accepted by _is_git_only_command
        safe_cmds = [
            "GIT_EDITOR=true git status",
            "GIT_SEQUENCE_EDITOR=: git diff",
            "GIT_PAGER=cat git log",
            "LC_ALL=C GIT_TERMINAL_PROMPT=0 git status",
            "PAGER='' git diff",
            'LANG="en_US.UTF-8" GIT_EDITOR="/bin/true" git status',
            "GIT_PAGER=\"\" git status",
        ]
        for cmd in safe_cmds:
            self.assertTrue(
                orchestrator_guard._is_git_only_command(cmd),
                f"Expected safe command to be accepted: {cmd}",
            )

        # Unsafe env prefixes rejected by _is_git_only_command
        unsafe_cmds = [
            "GIT_SSH_COMMAND=evil git status",
            "LD_PRELOAD=/tmp/lib.so git status",
            "GIT_EDITOR=/tmp/evil.sh git status",
            "GIT_EDITOR=vim git status",
            "PATH=/tmp git status",
            "GIT_EXEC_PATH=/tmp git status",
        ]
        for cmd in unsafe_cmds:
            self.assertFalse(
                orchestrator_guard._is_git_only_command(cmd),
                f"Expected unsafe command to be rejected: {cmd}",
            )

        # Tripped circuit breaker allows safe git inspections but denies unsafe commands
        state = orchestrator_guard._load_state(self.base_payload)
        cb = orchestrator_guard._get_circuit_breaker(state)
        cb["failure_streak"] = 2
        cb["tripped"] = True
        cb["last_error"] = "exit 1"
        orchestrator_guard._save_state(self.base_payload, state)

        res_safe = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "name": "run_command",
                    "args": {
                        "CommandLine": "GIT_EDITOR=true git status",
                        "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                    },
                },
            )
        )
        self.assertEqual(res_safe["decision"], "allow")

        res_unsafe = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "name": "run_command",
                    "args": {
                        "CommandLine": "GIT_SSH_COMMAND=evil git status",
                        "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                    },
                },
            )
        )
        self.assertEqual(res_unsafe["decision"], "deny")
        self.assertIn("2-turn circuit breaker tripped", res_unsafe["reason"])

        # Grounded spec with underscore headers is accepted
        cb["tripped"] = False
        cb["failure_streak"] = 0
        orchestrator_guard._save_state(self.base_payload, state)

        spec_file = Path(self.artifact_dir) / "scratch" / "spec_underscores.md"
        spec_file.parent.mkdir(parents=True, exist_ok=True)
        spec_file.write_text(
            "# Grounded Spec with Underscores\n\n"
            "## SIGNATURES_AND_SNIPPETS\n"
            "def foo() -> None: pass\n\n"
            "## INVARIANTS_AND_PITFALLS\n"
            "- invariant 1\n\n"
            "## VERIFICATION_EXPECTATIONS\n"
            "- run pytest\n"
        )
        sub_dispatch = dict(
            self.base_payload,
            toolCall={
                "name": "invoke_subagent",
                "args": {
                    "TypeName": "implementer",
                    "Prompt": (
                        "TASK: Implement underscores test.\n"
                        f"SPEC_FILE: {spec_file}\n"
                        "ALLOWED_FILES: tests/test_sample.py\n\n"
                        "RULES:\n1. Open file.\n2. Edit.\n3. Run pytest.\n\n"
                        "REPORT FORMAT:\nSTATUS: COMPLETED\n"
                        "FILES_CHANGED: tests/test_sample.py\n"
                        "VERIFICATION: pytest ok\n"
                        "NOTES: done\n"
                    ),
                },
            },
        )
        res_dispatch = orchestrator_guard.handle_pre_tool(sub_dispatch)
        self.assertEqual(res_dispatch["decision"], "allow")

    def test_sandbox_boundary_errors_exempt_from_circuit_breaker(self) -> None:
        state = orchestrator_guard._load_state(self.base_payload)
        cb = orchestrator_guard._get_circuit_breaker(state)
        cb["failure_streak"] = 0
        cb["tripped"] = False
        orchestrator_guard._save_state(self.base_payload, state)

        # 1. Error with .git/index.lock: Read-only file system does NOT increment failure streak
        post_ro_lock = dict(
            self.base_payload,
            toolCall={
                "id": "cmd_ro_lock",
                "name": "run_command",
                "args": {
                    "CommandLine": "git checkout -b branch1",
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                },
            },
            error="fatal: Unable to create '/path/.git/index.lock': Read-only file system",
        )
        orchestrator_guard.handle_post_tool(post_ro_lock)
        state_after_ro = orchestrator_guard._load_state(self.base_payload)
        cb_ro = orchestrator_guard._get_circuit_breaker(state_after_ro)
        self.assertEqual(cb_ro.get("failure_streak", 0), 0)
        self.assertFalse(cb_ro.get("tripped", False))

        # 2. Output with sandbox banner does NOT increment failure streak
        post_sandbox_banner = dict(
            self.base_payload,
            toolCall={
                "id": "cmd_banner",
                "name": "run_command",
                "args": {
                    "CommandLine": "git push origin branch1",
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                },
            },
            output="There were sandbox errors that may or may not be related to the failure\nexit status 1",
            error="Command exited with code 1",
        )
        orchestrator_guard.handle_post_tool(post_sandbox_banner)
        state_after_banner = orchestrator_guard._load_state(self.base_payload)
        cb_banner = orchestrator_guard._get_circuit_breaker(state_after_banner)
        self.assertEqual(cb_banner.get("failure_streak", 0), 0)
        self.assertFalse(cb_banner.get("tripped", False))

        # 3. Payload with sandboxError=True does NOT increment failure streak
        post_sb_err = dict(
            self.base_payload,
            toolCall={
                "id": "cmd_sberr",
                "name": "run_command",
                "args": {
                    "CommandLine": "npm test",
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                },
            },
            sandboxError=True,
            error="Process failed in sandbox",
        )
        orchestrator_guard.handle_post_tool(post_sb_err)
        state_after_sberr = orchestrator_guard._load_state(self.base_payload)
        cb_sberr = orchestrator_guard._get_circuit_breaker(state_after_sberr)
        self.assertEqual(cb_sberr.get("failure_streak", 0), 0)
        self.assertFalse(cb_sberr.get("tripped", False))

        # 4. Read-only FETCH_HEAD does NOT increment failure streak
        post_fetch_ro = dict(
            self.base_payload,
            toolCall={
                "id": "cmd_fetch_ro",
                "name": "run_command",
                "args": {
                    "CommandLine": "git fetch origin",
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                },
            },
            output="fatal: cannot open .git/FETCH_HEAD: Read-only file system\n",
            error="Command exited with code 128",
        )
        orchestrator_guard.handle_post_tool(post_fetch_ro)
        state_after_fetch = orchestrator_guard._load_state(self.base_payload)
        cb_fetch = orchestrator_guard._get_circuit_breaker(state_after_fetch)
        self.assertEqual(cb_fetch.get("failure_streak", 0), 0)
        self.assertFalse(cb_fetch.get("tripped", False))

        # 5. Genuine code failures DO increment failure streak and trip breaker at streak 2
        post_real_1 = dict(
            self.base_payload,
            toolCall={
                "id": "cmd_real_1",
                "name": "run_command",
                "args": {
                    "CommandLine": "pytest tests/",
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                },
            },
            error="exit status 1",
        )
        orchestrator_guard.handle_post_tool(post_real_1)
        state_after_real_1 = orchestrator_guard._load_state(self.base_payload)
        cb_real_1 = orchestrator_guard._get_circuit_breaker(state_after_real_1)
        self.assertEqual(cb_real_1.get("failure_streak", 0), 1)
        self.assertFalse(cb_real_1.get("tripped", False))

        post_real_2 = dict(
            self.base_payload,
            toolCall={
                "id": "cmd_real_2",
                "name": "run_command",
                "args": {
                    "CommandLine": "pytest tests/",
                    "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                },
            },
            error="exit status 1",
        )
        orchestrator_guard.handle_post_tool(post_real_2)
        state_after_real_2 = orchestrator_guard._load_state(self.base_payload)
        cb_real_2 = orchestrator_guard._get_circuit_breaker(state_after_real_2)
        self.assertEqual(cb_real_2.get("failure_streak", 0), 2)
        self.assertTrue(cb_real_2.get("tripped", False))

    def test_phase1_define_subagent_verifier_and_reviewer_permissions(self) -> None:
        # 1. PascalCase define_subagent strips write, MCP, and subagent tools for verifier & reviewer
        for role_name in ("verifier", "reviewer"):
            res = orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={
                        "name": "define_subagent",
                        "args": {
                            "Name": role_name,
                            "EnableWriteTools": True,
                            "EnableMCPTools": True,
                            "EnableSubagentTools": True,
                        },
                    },
                )
            )
            self.assertEqual(res["decision"], "allow")
            self.assertEqual(
                res["overwrite"],
                {
                    "EnableSubagentTools": False,
                    "EnableWriteTools": False,
                    "EnableMCPTools": False,
                },
            )

        # 2. snake_case define_subagent strips write, MCP, and subagent tools for verifier & reviewer
        for role_name in ("verifier", "reviewer"):
            res_snake = orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={
                        "name": "define_subagent",
                        "args": {
                            "name": role_name,
                            "enable_write_tools": True,
                            "enable_mcp_tools": True,
                            "enable_subagent_tools": True,
                        },
                    },
                )
            )
            self.assertEqual(res_snake["decision"], "allow")
            self.assertEqual(
                res_snake["overwrite"],
                {
                    "enable_subagent_tools": False,
                    "enable_write_tools": False,
                    "enable_mcp_tools": False,
                },
            )

    def test_phase1_verifier_reviewer_model_policy_and_regressions(self) -> None:
        short_readonly_prompt = (
            "TASK: Inspect module changes.\n"
            "REVIEW_LENS: security\n"
            "RULES:\n1. Read changed files.\n2. Run git diff.\n"
            "REPORT FORMAT:\nSTATUS: COMPLETED\nFILES_CHANGED: NONE\nVERIFICATION: ok\nNOTES: done"
        )
        # Verifier and reviewer model policy: omitted -> flash, flash -> flash, pro -> pro,
        # inherit -> inherit, flash_lite -> flash, short readonly prompt NOT downgraded to flash_lite.
        for type_name in ("verifier", "reviewer"):
            # Omitted Model -> overwritten to flash
            res_omitted = orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={
                        "id": f"m_omit_{type_name}",
                        "name": "invoke_subagent",
                        "args": {
                            "TypeName": type_name,
                            "Prompt": short_readonly_prompt,
                        },
                    },
                )
            )
            self.assertEqual(res_omitted["decision"], "allow")
            self.assertEqual(res_omitted["overwrite"]["Model"], "flash")

            # Explicit flash -> kept as flash (no overwrite)
            res_flash = orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={
                        "id": f"m_flash_{type_name}",
                        "name": "invoke_subagent",
                        "args": {
                            "TypeName": type_name,
                            "Model": "flash",
                            "Prompt": short_readonly_prompt.replace("security", f"flash-{type_name}"),
                        },
                    },
                )
            )
            self.assertEqual(res_flash, {"decision": "allow"})

            # Explicit pro -> allowed & preserved
            res_pro = orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={
                        "id": f"m_pro_{type_name}",
                        "name": "invoke_subagent",
                        "args": {
                            "TypeName": type_name,
                            "Model": "pro",
                            "Prompt": short_readonly_prompt.replace("security", f"pro-{type_name}"),
                        },
                    },
                )
            )
            self.assertEqual(res_pro, {"decision": "allow"})

            # Explicit inherit -> allowed & preserved
            res_inherit = orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={
                        "id": f"m_inh_{type_name}",
                        "name": "invoke_subagent",
                        "args": {
                            "TypeName": type_name,
                            "Model": "inherit",
                            "Prompt": short_readonly_prompt.replace("security", f"inh-{type_name}"),
                        },
                    },
                )
            )
            self.assertEqual(res_inherit, {"decision": "allow"})

            # Explicit flash_lite -> rewritten to flash
            res_lite = orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={
                        "id": f"m_lite_{type_name}",
                        "name": "invoke_subagent",
                        "args": {
                            "TypeName": type_name,
                            "Model": "flash_lite",
                            "Prompt": short_readonly_prompt.replace("security", f"lite-{type_name}"),
                        },
                    },
                )
            )
            self.assertEqual(res_lite["decision"], "allow")
            self.assertEqual(res_lite["overwrite"]["Model"], "flash")

        # Regression guard: scout, research, implementer still deny pro and rewrite inherit
        for std_type in ("scout", "research", "implementer"):
            res_std_pro = orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={
                        "id": f"std_pro_{std_type}",
                        "name": "invoke_subagent",
                        "args": {
                            "TypeName": std_type,
                            "Model": "pro",
                            "Prompt": VALID_PROMPT,
                        },
                    },
                )
            )
            self.assertEqual(res_std_pro["decision"], "deny")
            self.assertIn("Model='pro'", res_std_pro["reason"])

            res_std_inh = orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={
                        "id": f"std_inh_{std_type}",
                        "name": "invoke_subagent",
                        "args": {
                            "TypeName": std_type,
                            "Model": "inherit",
                            "Prompt": VALID_PROMPT,
                        },
                    },
                )
            )
            self.assertEqual(res_std_inh["decision"], "allow")
            expected_default = "flash_lite" if std_type == "scout" else "flash"
            self.assertEqual(res_std_inh["overwrite"]["Model"], expected_default)

    def test_phase1_verifier_reviewer_strict_readonly_and_dirty_receipt(self) -> None:
        brain_base = Path(self.artifact_dir) / "brain"
        parent_brain = brain_base / "parent-ro-test"
        verifier_cid = "sub-verifier-ro-01"
        reviewer_cid = "sub-reviewer-ro-02"
        for d in (parent_brain, brain_base / verifier_cid, brain_base / reviewer_cid):
            d.mkdir(parents=True, exist_ok=True)

        parent_payload = dict(
            self.base_payload,
            conversationId="parent-ro-test",
            artifactDirectoryPath=str(parent_brain),
        )
        orchestrator_guard._cli_activate(parent_payload)
        orchestrator_guard.register_subagent(
            parent_payload,
            conversation_id=verifier_cid,
            type_name="verifier",
            model="flash",
            status="active",
        )
        orchestrator_guard.register_subagent(
            parent_payload,
            conversation_id=reviewer_cid,
            type_name="reviewer",
            model="pro",
            status="active",
            review_lens="security",
        )

        target_code = str(orchestrator_guard.PROJECT_ROOT / "src" / "app.py")
        for sub_cid, sub_type in ((verifier_cid, "verifier"), (reviewer_cid, "reviewer")):
            sub_payload = dict(
                self.base_payload,
                conversationId=sub_cid,
                artifactDirectoryPath=str(brain_base / sub_cid),
            )
            # MUTATION_TOOLS blocked
            for tool_name, tool_args in (
                ("write_file", {"Path": target_code, "Content": "x = 1"}),
                ("edit_file", {"Path": target_code, "Target": "a", "Replacement": "b"}),
                ("replace_file_content", {"TargetFile": target_code, "TargetContent": "a", "ReplacementContent": "b"}),
            ):
                res_mut = orchestrator_guard.handle_pre_tool(
                    dict(sub_payload, toolCall={"name": tool_name, "args": tool_args})
                )
                self.assertEqual(res_mut["decision"], "deny")
                self.assertIn("strictly read-only", res_mut["reason"])

            # Mutating shell commands blocked
            for bad_cmd in ("echo hi > out.txt", "rm -f foo.py", "sed -i 's/a/b/' foo.py", "touch new.py"):
                res_cmd = orchestrator_guard.handle_pre_tool(
                    dict(
                        sub_payload,
                        toolCall={
                            "name": "run_command",
                            "args": {"CommandLine": bad_cmd, "Cwd": str(orchestrator_guard.PROJECT_ROOT)},
                        },
                    )
                )
                self.assertEqual(res_cmd["decision"], "deny")
                self.assertIn("strictly read-only", res_cmd["reason"])

            # read_file and git diff allowed
            res_read = orchestrator_guard.handle_pre_tool(
                dict(
                    sub_payload,
                    toolCall={
                        "name": "read_file",
                        "args": {"Path": str(orchestrator_guard.PROJECT_ROOT / "README.md"), "StartLine": 1, "EndLine": 10},
                    },
                )
            )
            self.assertEqual(res_read["decision"], "allow")
            res_git = orchestrator_guard.handle_pre_tool(
                dict(
                    sub_payload,
                    toolCall={
                        "name": "run_command",
                        "args": {"CommandLine": "git diff HEAD", "Cwd": str(orchestrator_guard.PROJECT_ROOT)},
                    },
                )
            )
            self.assertEqual(res_git["decision"], "allow")

        # Synchronous invoke_subagent completion for verifier/reviewer NEVER clears dirty test_receipt
        parent_state = orchestrator_guard._load_state(parent_payload)
        parent_receipt = orchestrator_guard._get_test_receipt(parent_state, parent_payload)
        parent_receipt["dirty"] = True
        parent_receipt["verified"] = False
        orchestrator_guard._save_state(parent_payload, parent_state)

        for ro_type in ("verifier", "reviewer"):
            orchestrator_guard.handle_post_tool(
                dict(
                    parent_payload,
                    toolCall={
                        "id": f"sync_{ro_type}",
                        "name": "invoke_subagent",
                        "args": {"TypeName": ro_type, "Role": ro_type.title()},
                    },
                    output="STATUS: COMPLETED\nFILES_CHANGED: NONE\nVERIFICATION: inspected\nNOTES: VERDICT: PASS",
                    error="",
                )
            )
            st_after = orchestrator_guard._load_state(parent_payload)
            rec_after = orchestrator_guard._get_test_receipt(st_after, parent_payload)
            self.assertTrue(rec_after["dirty"])
            self.assertFalse(rec_after["verified"])

    def test_phase1_verifier_reviewer_require_4_header_report(self) -> None:
        compact_report = "STATUS: COMPLETED\nNOTES: Looks good to me"
        valid_4_header = (
            "STATUS: COMPLETED\n"
            "FILES_CHANGED: NONE\n"
            "VERIFICATION: Ran git diff and inspected tests\n"
            "NOTES: VERDICT: APPROVE"
        )
        brain_base = Path(self.artifact_dir) / "brain"
        parent_brain = brain_base / "parent-report-test"
        parent_brain.mkdir(parents=True, exist_ok=True)
        parent_payload = dict(
            self.base_payload,
            conversationId="parent-report-test",
            artifactDirectoryPath=str(parent_brain),
        )
        orchestrator_guard._cli_activate(parent_payload)

        for idx, role_type in enumerate(("verifier", "reviewer"), start=1):
            sub_cid = f"sub-{role_type}-report-{idx}"
            (brain_base / sub_cid).mkdir(parents=True, exist_ok=True)
            orchestrator_guard.register_subagent(
                parent_payload,
                conversation_id=sub_cid,
                type_name=role_type,
                model="flash",
                status="active",
                review_lens="security" if role_type == "reviewer" else None,
            )
            sub_payload = dict(
                self.base_payload,
                conversationId=sub_cid,
                artifactDirectoryPath=str(brain_base / sub_cid),
            )
            # 2-header shorthand rejected in send_message
            res_msg = orchestrator_guard.handle_pre_tool(
                dict(
                    sub_payload,
                    toolCall={
                        "name": "send_message",
                        "args": {"Recipient": "parent-report-test", "Message": compact_report},
                    },
                )
            )
            self.assertEqual(res_msg["decision"], "deny")
            self.assertIn("Expected 4 header lines", res_msg["reason"])

            # 2-header shorthand triggers continue in handle_stop
            res_stop_bad = orchestrator_guard.handle_stop(
                dict(
                    sub_payload,
                    executionNum=1,
                    lastAssistantMessage=compact_report,
                    fullyIdle=True,
                )
            )
            self.assertEqual(res_stop_bad["decision"], "continue")
            self.assertIn("Expected 4 header lines", res_stop_bad["reason"])

            # 4-header report passes handle_stop and marks status completed in parent brain
            res_stop_ok = orchestrator_guard.handle_stop(
                dict(
                    sub_payload,
                    executionNum=2,
                    lastAssistantMessage=valid_4_header,
                    fullyIdle=True,
                )
            )
            self.assertEqual(res_stop_ok["decision"], "allow")
            parent_state = orchestrator_guard._load_state(parent_payload)
            self.assertEqual(parent_state["subagents"][sub_cid]["status"], "completed")

    def test_phase1_freshness_gate_mutation_epoch_root_and_implementer(self) -> None:
        brain_base = Path(self.artifact_dir) / "brain"
        parent_brain = brain_base / "parent-epoch-test"
        impl_cid = "sub-impl-epoch-01"
        verifier_cid = "sub-verifier-epoch-02"
        reviewer_cid = "sub-reviewer-epoch-03"
        for d in (
            parent_brain,
            brain_base / impl_cid,
            brain_base / verifier_cid,
            brain_base / reviewer_cid,
        ):
            d.mkdir(parents=True, exist_ok=True)

        parent_payload = dict(
            self.base_payload,
            conversationId="parent-epoch-test",
            artifactDirectoryPath=str(parent_brain),
        )
        orchestrator_guard._cli_activate(parent_payload)
        orchestrator_guard.handle_pre_invocation(parent_payload)

        orchestrator_guard.register_subagent(
            parent_payload,
            conversation_id=impl_cid,
            type_name="implementer",
            model="flash",
            status="active",
        )
        orchestrator_guard.register_subagent(
            parent_payload,
            conversation_id=verifier_cid,
            type_name="verifier",
            model="flash",
            status="active",
            spawned_mutation_epoch=0,
        )
        orchestrator_guard.register_subagent(
            parent_payload,
            conversation_id=reviewer_cid,
            type_name="reviewer",
            model="pro",
            status="active",
            spawned_mutation_epoch=0,
            review_lens="security",
        )

        # Clarifying send_message BEFORE any code mutation is allowed
        for target_cid in (verifier_cid, reviewer_cid):
            res_before = orchestrator_guard.handle_pre_tool(
                dict(
                    parent_payload,
                    toolCall={
                        "name": "send_message",
                        "args": {"Recipient": target_cid, "Message": "Please also check input bounds."},
                    },
                )
            )
            self.assertEqual(res_before["decision"], "allow")

        # Implementer mutates workspace file -> propagates dirty=True and bumps parent mutation_epoch
        impl_payload = dict(
            self.base_payload,
            conversationId=impl_cid,
            artifactDirectoryPath=str(brain_base / impl_cid),
            toolCall={
                "name": "edit_file",
                "args": {
                    "Path": str(orchestrator_guard.PROJECT_ROOT / "src_mod.py"),
                    "Target": "a",
                    "Replacement": "b",
                },
            },
        )
        res_impl_edit = orchestrator_guard.handle_pre_tool(impl_payload)
        self.assertEqual(res_impl_edit["decision"], "allow")

        parent_state = orchestrator_guard._load_state(parent_payload)
        self.assertEqual(
            orchestrator_guard._get_test_receipt(parent_state, parent_payload).get("mutation_epoch"),
            1,
        )

        # send_message to stale verifier & reviewer is now DENIED
        for target_cid, r_type in ((verifier_cid, "verifier"), (reviewer_cid, "reviewer")):
            res_stale = orchestrator_guard.handle_pre_tool(
                dict(
                    parent_payload,
                    toolCall={
                        "name": "send_message",
                        "args": {"Recipient": target_cid, "Message": "Re-verify after fix."},
                    },
                )
            )
            self.assertEqual(res_stale["decision"], "deny")
            self.assertIn(f"Stale {r_type}", res_stale["reason"])

        # Spawn fresh verifier at epoch 1 -> send_message allowed until root edit bumps epoch to 2
        fresh_verifier_cid = "sub-verifier-fresh-04"
        orchestrator_guard.register_subagent(
            parent_payload,
            conversation_id=fresh_verifier_cid,
            type_name="verifier",
            model="flash",
            status="active",
            spawned_mutation_epoch=1,
        )
        res_fresh_ok = orchestrator_guard.handle_pre_tool(
            dict(
                parent_payload,
                toolCall={
                    "name": "send_message",
                    "args": {"Recipient": fresh_verifier_cid, "Message": "Clarification at epoch 1."},
                },
            )
        )
        self.assertEqual(res_fresh_ok["decision"], "allow")

        # Root edit bumps mutation_epoch to 2 -> fresh_verifier becomes stale too
        res_root_edit = orchestrator_guard.handle_pre_tool(
            dict(
                parent_payload,
                toolCall={
                    "name": "write_file",
                    "args": {
                        "Path": str(orchestrator_guard.PROJECT_ROOT / "root_mod.py"),
                        "Content": "x = 2\n",
                    },
                },
            )
        )
        self.assertEqual(res_root_edit["decision"], "allow")
        status_out = orchestrator_guard._cli_status(parent_payload)
        self.assertEqual(status_out["testReceipt"]["mutationEpoch"], 2)

        res_stale_root = orchestrator_guard.handle_pre_tool(
            dict(
                parent_payload,
                toolCall={
                    "name": "send_message",
                    "args": {"Recipient": fresh_verifier_cid, "Message": "Verify root edit."},
                },
            )
        )
        self.assertEqual(res_stale_root["decision"], "deny")
        self.assertIn("Stale verifier", res_stale_root["reason"])

    def test_phase1_disjoint_implementer_lanes_and_lifecycle_release(self) -> None:
        def _impl_entry(allowed_line: str, workspace: str = "inherit", role: str = "Lane") -> dict:
            header = f"ALLOWED_FILES: {allowed_line}\n" if allowed_line else ""
            return {
                "TypeName": "implementer",
                "Role": role,
                "Model": "flash",
                "Workspace": workspace,
                "Prompt": (
                    f"TASK: Implement {role}.\n"
                    f"{header}"
                    "RULES:\n1. Edit assigned files.\n2. Run tests.\n"
                    "REPORT FORMAT:\nSTATUS: COMPLETED\nFILES_CHANGED: a.py\nVERIFICATION: ok\nNOTES: done"
                ),
            }

        # 1. Atomic Subagents batch: exact & directory-prefix overlap denied WITHOUT mutating pending_spawns
        res_batch_overlap = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "batch_overlap_1",
                    "name": "invoke_subagent",
                    "args": {
                        "Subagents": [
                            _impl_entry("web/", role="Lane Web"),
                            _impl_entry("web/App.tsx", role="Lane App"),
                        ]
                    },
                },
            )
        )
        self.assertEqual(res_batch_overlap["decision"], "deny")
        self.assertIn("Overlapping ALLOWED_FILES", res_batch_overlap["reason"])
        st_atomic = orchestrator_guard._load_state(self.base_payload)
        self.assertEqual(st_atomic.get("pending_spawns"), [])
        self.assertEqual(st_atomic.get("recorded_spawn_calls"), [])

        # 2. Disjoint lanes in a batch are allowed
        res_disjoint = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "batch_disjoint_1",
                    "name": "invoke_subagent",
                    "args": {
                        "Subagents": [
                            _impl_entry("web/", role="Lane Web"),
                            _impl_entry("api/", role="Lane API"),
                            _impl_entry("docs/", role="Lane Docs"),
                        ]
                    },
                },
            )
        )
        self.assertEqual(res_disjoint["decision"], "allow")

        # 3. Local CLI sequential pending_spawns overlap is denied
        res_pending_conflict = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "cli_conflict_1",
                    "name": "invoke_subagent",
                    "args": _impl_entry("api/routes.py", role="Lane API Routes"),
                },
            )
        )
        self.assertEqual(res_pending_conflict["decision"], "deny")
        self.assertIn("Overlapping ALLOWED_FILES", res_pending_conflict["reason"])

        # 4. Synchronous PostToolUse completion releases pending_spawns lane locks
        orchestrator_guard.handle_post_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "batch_disjoint_1",
                    "name": "invoke_subagent",
                    "args": {
                        "Subagents": [
                            _impl_entry("web/", role="Lane Web"),
                            _impl_entry("api/", role="Lane API"),
                            _impl_entry("docs/", role="Lane Docs"),
                        ]
                    },
                },
                output="STATUS: COMPLETED\nFILES_CHANGED: web/a.ts\nVERIFICATION: ok\nNOTES: done",
                error="",
            )
        )
        res_after_post = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "cli_after_post_1",
                    "name": "invoke_subagent",
                    "args": _impl_entry("api/routes.py", role="Lane API Routes"),
                },
            )
        )
        self.assertEqual(res_after_post["decision"], "allow")

        # 5. Active subagents overlap check + release via handle_stop and manage_subagents(kill)
        brain_base = Path(self.artifact_dir) / "brain"
        parent_brain = brain_base / "parent-lane-test"
        lane_a = "sub-lane-a-01"
        lane_b = "sub-lane-b-02"
        for d in (parent_brain, brain_base / lane_a, brain_base / lane_b):
            d.mkdir(parents=True, exist_ok=True)
        parent_payload = dict(
            self.base_payload,
            conversationId="parent-lane-test",
            artifactDirectoryPath=str(parent_brain),
        )
        orchestrator_guard._cli_activate(parent_payload)
        orchestrator_guard.register_subagent(
            parent_payload,
            conversation_id=lane_a,
            type_name="implementer",
            model="flash",
            allowed_files=["src/a.py"],
            status="active",
        )
        res_sub_overlap = orchestrator_guard.handle_pre_tool(
            dict(
                parent_payload,
                toolCall={
                    "id": "sub_overlap_1",
                    "name": "invoke_subagent",
                    "args": _impl_entry("src/a.py", role="Conflicting Worker"),
                },
            )
        )
        self.assertEqual(res_sub_overlap["decision"], "deny")
        self.assertIn("Overlapping ALLOWED_FILES", res_sub_overlap["reason"])

        # Overlapping files ARE allowed when isolated in separate branch/share workspaces
        orchestrator_guard.register_subagent(
            parent_payload,
            conversation_id=lane_b,
            type_name="implementer",
            model="flash",
            workspace="branch",
            allowed_files=["src/shared.py"],
            status="active",
        )
        res_isolated = orchestrator_guard.handle_pre_tool(
            dict(
                parent_payload,
                toolCall={
                    "id": "sub_isolated_share",
                    "name": "invoke_subagent",
                    "args": _impl_entry("src/shared.py", workspace="share", role="Share Worker"),
                },
            )
        )
        self.assertEqual(res_isolated["decision"], "allow")

        # Omitted ALLOWED_FILES stays backward-compatible
        res_omitted = orchestrator_guard.handle_pre_tool(
            dict(
                parent_payload,
                toolCall={
                    "id": "sub_omitted_allowed",
                    "name": "invoke_subagent",
                    "args": _impl_entry("", role="Unrestricted Legacy Worker"),
                },
            )
        )
        self.assertEqual(res_omitted["decision"], "allow")

        # Sequential reuse allowed after handle_stop marks lane_a completed
        res_stop_a = orchestrator_guard.handle_stop(
            dict(
                self.base_payload,
                conversationId=lane_a,
                artifactDirectoryPath=str(brain_base / lane_a),
                lastAssistantMessage=(
                    "STATUS: COMPLETED\nFILES_CHANGED: src/a.py\nVERIFICATION: pytest ok\nNOTES: done"
                ),
                fullyIdle=True,
            )
        )
        self.assertEqual(res_stop_a["decision"], "allow")
        res_reuse_a = orchestrator_guard.handle_pre_tool(
            dict(
                parent_payload,
                toolCall={
                    "id": "sub_reuse_after_stop",
                    "name": "invoke_subagent",
                    "args": _impl_entry("src/a.py", role="Follow-up Worker"),
                },
            )
        )
        self.assertEqual(res_reuse_a["decision"], "allow")

        # Reactivating lane_a via send_message while another active worker holds src/c.py is blocked
        lane_c = "sub-lane-c-03"
        orchestrator_guard.register_subagent(
            parent_payload,
            conversation_id=lane_c,
            type_name="implementer",
            model="flash",
            allowed_files=["src/c.py"],
            status="active",
        )
        res_reactivate_conflict = orchestrator_guard.handle_pre_tool(
            dict(
                parent_payload,
                toolCall={
                    "name": "send_message",
                    "args": {
                        "Recipient": lane_a,
                        "Message": "ALLOWED_FILES: src/c.py\nContinue work on c.py.",
                    },
                },
            )
        )
        self.assertEqual(res_reactivate_conflict["decision"], "deny")
        self.assertIn("Overlapping ALLOWED_FILES", res_reactivate_conflict["reason"])

        # manage_subagents(Action='kill') marks lane_c killed and releases src/c.py
        orchestrator_guard.handle_pre_tool(
            dict(
                parent_payload,
                toolCall={
                    "name": "manage_subagents",
                    "args": {"Action": "kill", "ConversationIds": [lane_c]},
                },
            )
        )
        res_reactivate_ok = orchestrator_guard.handle_pre_tool(
            dict(
                parent_payload,
                toolCall={
                    "name": "send_message",
                    "args": {
                        "Recipient": lane_a,
                        "Message": "ALLOWED_FILES: src/c.py\nContinue work on c.py.",
                    },
                },
            )
        )
        self.assertEqual(res_reactivate_ok["decision"], "allow")

    def test_phase1_reviewer_lens_enforcement_and_round2_release(self) -> None:
        def _rev_entry(lens_line: str, role: str = "Reviewer") -> dict:
            header = f"REVIEW_LENS: {lens_line}\n" if lens_line is not None else ""
            return {
                "TypeName": "reviewer",
                "Role": role,
                "Model": "flash",
                "Prompt": (
                    f"TASK: Review changes.\n"
                    f"{header}"
                    "RULES:\n1. Inspect diff.\n2. Report findings.\n"
                    "REPORT FORMAT:\nSTATUS: COMPLETED\nFILES_CHANGED: NONE\nVERIFICATION: git diff\nNOTES: VERDICT: APPROVE"
                ),
            }

        # 1. Missing or blank REVIEW_LENS denied
        for bad_lens in (None, '""', "---"):
            res_missing = orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={
                        "id": f"rev_bad_{bad_lens}",
                        "name": "invoke_subagent",
                        "args": _rev_entry(bad_lens),
                    },
                )
            )
            self.assertEqual(res_missing["decision"], "deny")
            self.assertIn("REVIEW_LENS", res_missing["reason"])

        # Blank REVIEW_LENS followed by newline must not match across newline onto next line
        for idx, blank_prompt in enumerate(
            (
                "TASK: Review changes.\nREVIEW_LENS:\nRULES:\n1. Review.\n",
                "REVIEW_LENS:   \nTASK: Review changes.\nRULES:\n1. Review.\n",
            )
        ):
            res_blank = orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={
                        "id": f"rev_blank_{idx}",
                        "name": "invoke_subagent",
                        "args": {
                            "TypeName": "reviewer",
                            "Role": "Reviewer",
                            "Model": "flash",
                            "Prompt": (
                                f"{blank_prompt}"
                                "REPORT FORMAT:\nSTATUS: COMPLETED\nFILES_CHANGED: NONE\nVERIFICATION: git diff\nNOTES: VERDICT: APPROVE"
                            ),
                        },
                    },
                )
            )
            self.assertEqual(res_blank["decision"], "deny")
            self.assertIn("REVIEW_LENS", res_blank["reason"])

        # 2. Duplicate normalized lens in atomic Subagents batch denied without mutating pending_spawns
        res_dup_batch = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "rev_dup_batch",
                    "name": "invoke_subagent",
                    "args": {
                        "Subagents": [
                            _rev_entry("edge cases", role="Reviewer 1"),
                            _rev_entry("edge_cases", role="Reviewer 2"),
                        ]
                    },
                },
            )
        )
        self.assertEqual(res_dup_batch["decision"], "deny")
        self.assertIn("Duplicate REVIEW_LENS 'edge-cases'", res_dup_batch["reason"])
        st_atomic = orchestrator_guard._load_state(self.base_payload)
        self.assertEqual(st_atomic.get("pending_spawns"), [])

        # 3. Distinct lenses allowed in parallel batch
        res_distinct = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "rev_distinct_batch",
                    "name": "invoke_subagent",
                    "args": {
                        "Subagents": [
                            _rev_entry("ux", role="UX Reviewer"),
                            _rev_entry("simplicity", role="Simplicity Reviewer"),
                            _rev_entry("edge-cases", role="Edge Cases Reviewer"),
                            _rev_entry("security", role="Security Reviewer"),
                        ]
                    },
                },
            )
        )
        self.assertEqual(res_distinct["decision"], "allow")

        # Duplicate against pending_spawns denied
        res_dup_pending = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "rev_dup_pending",
                    "name": "invoke_subagent",
                    "args": _rev_entry("Edge Cases", role="Dup Reviewer"),
                },
            )
        )
        self.assertEqual(res_dup_pending["decision"], "deny")
        self.assertIn("Duplicate REVIEW_LENS 'edge-cases'", res_dup_pending["reason"])

        # 4. Round-1 active reviewer blocks duplicate lens; round-2 allowed after handle_stop completes round-1
        brain_base = Path(self.artifact_dir) / "brain"
        parent_brain = brain_base / "parent-rev-round2"
        rev1_cid = "sub-rev-sec-r1"
        for d in (parent_brain, brain_base / rev1_cid):
            d.mkdir(parents=True, exist_ok=True)
        parent_payload = dict(
            self.base_payload,
            conversationId="parent-rev-round2",
            artifactDirectoryPath=str(parent_brain),
        )
        orchestrator_guard._cli_activate(parent_payload)
        orchestrator_guard.register_subagent(
            parent_payload,
            conversation_id=rev1_cid,
            type_name="reviewer",
            model="pro",
            status="active",
            review_lens="security",
        )
        status_r1 = orchestrator_guard._cli_status(parent_payload)
        self.assertEqual(status_r1["activeSubagents"][rev1_cid]["status"], "active")
        self.assertEqual(status_r1["activeSubagents"][rev1_cid]["reviewLens"], "security")

        res_dup_active = orchestrator_guard.handle_pre_tool(
            dict(
                parent_payload,
                toolCall={
                    "id": "rev_dup_active",
                    "name": "invoke_subagent",
                    "args": _rev_entry("security", role="Round 2 Early"),
                },
            )
        )
        self.assertEqual(res_dup_active["decision"], "deny")
        self.assertIn("Duplicate REVIEW_LENS 'security'", res_dup_active["reason"])

        # Round-1 reviewer completes via handle_stop
        res_stop_r1 = orchestrator_guard.handle_stop(
            dict(
                self.base_payload,
                conversationId=rev1_cid,
                artifactDirectoryPath=str(brain_base / rev1_cid),
                lastAssistantMessage=(
                    "STATUS: COMPLETED\nFILES_CHANGED: NONE\nVERIFICATION: git diff\nNOTES: VERDICT: REQUEST_CHANGES"
                ),
                fullyIdle=True,
            )
        )
        self.assertEqual(res_stop_r1["decision"], "allow")

        # Round-2 fresh reviewer reusing "security" is now allowed
        res_r2 = orchestrator_guard.handle_pre_tool(
            dict(
                parent_payload,
                toolCall={
                    "id": "rev_r2_ok",
                    "name": "invoke_subagent",
                    "args": _rev_entry("security", role="Round 2 Fresh Security"),
                },
            )
        )
        self.assertEqual(res_r2["decision"], "allow")

    def test_orchestrator_guard_failsafe_catches_import_indentation_error(self) -> None:
        import shutil
        import subprocess

        scripts_copy = Path(self.artifact_dir) / "scripts_copy"
        guard_copy = scripts_copy / "guard"
        guard_copy.mkdir(parents=True, exist_ok=True)
        src_script = Path(orchestrator_guard.__file__).resolve()
        dst_script = scripts_copy / "orchestrator_guard.py"
        shutil.copy2(src_script, dst_script)
        (guard_copy / "__init__.py").write_text(
            "def broken():\n  pass\n   bad_indent = 1\n",
            encoding="utf-8",
        )

        for mode in ("pre-tool", "stop", "post-tool"):
            proc = subprocess.run(
                [sys.executable, str(dst_script), mode],
                input="{}",
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, f"mode={mode} stderr={proc.stderr}")
            out = json.loads(proc.stdout.strip())
            if mode in ("pre-tool", "stop"):
                self.assertEqual(out.get("decision"), "allow")
                self.assertIn("Guard fail-safe:", out.get("reason", ""))
            else:
                self.assertEqual(out, {})

    def test_plan_execute_phase_persists_across_steps_and_subagent_wakeups(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)
        plan_text = "ALLOWED_FILES: file_a.py, file_b.py, file_c.py\n1. Implement feature"
        res_exit = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                lastUserInput="Implement feature",
                toolCall={"name": "exit_plan_mode", "args": {"plan": plan_text}},
            )
        )
        self.assertEqual(res_exit["decision"], "allow")
        st = orchestrator_guard._load_state(self.base_payload)
        self.assertTrue(st.get("plan_execute_phase"))
        self.assertEqual(st.get("plan_execute_user_input"), "Implement feature")
        self.assertFalse(st.get("plan_execute_turn_finished"))

        # Repeated PreInvocation calls with identical lastUserInput preserve plan_execute_phase
        for inv_num in (2, 3):
            orchestrator_guard.handle_pre_invocation(
                dict(self.base_payload, invocationNum=inv_num, lastUserInput="Implement feature")
            )
            st_step = orchestrator_guard._load_state(self.base_payload)
            self.assertTrue(st_step.get("plan_execute_phase"))
            self.assertTrue(st_step.get("plan_execute_consumed"))
            self.assertEqual(
                st_step.get("plan_allowed_files"),
                ["file_a.py", "file_b.py", "file_c.py"],
            )

        # Active subagent prevents idle stop from marking plan_execute_turn_finished
        orchestrator_guard.register_subagent(
            self.base_payload,
            conversation_id="sub-active-plan-1",
            type_name="implementer",
            model="flash",
            status="active",
        )
        res_stop_active = orchestrator_guard.handle_stop(
            dict(self.base_payload, fullyIdle=True)
        )
        self.assertEqual(res_stop_active["decision"], "allow")
        st_after_active_stop = orchestrator_guard._load_state(self.base_payload)
        self.assertFalse(st_after_active_stop.get("plan_execute_turn_finished"))
        orchestrator_guard.handle_pre_invocation(
            dict(self.base_payload, invocationNum=4, lastUserInput="Implement feature")
        )
        self.assertTrue(
            orchestrator_guard._load_state(self.base_payload).get("plan_execute_phase")
        )

        # Active pending_spawn also preserves plan_execute_phase across idle stop & wakeup
        st_pending = orchestrator_guard._load_state(self.base_payload)
        st_pending["subagents"]["sub-active-plan-1"]["status"] = "completed"
        st_pending["pending_spawns"] = [
            {"type": "implementer", "role": "worker", "status": "active"}
        ]
        orchestrator_guard._save_state(self.base_payload, st_pending)
        orchestrator_guard.handle_stop(dict(self.base_payload, fullyIdle=True))
        self.assertFalse(
            orchestrator_guard._load_state(self.base_payload).get("plan_execute_turn_finished")
        )
        orchestrator_guard.handle_pre_invocation(
            dict(self.base_payload, invocationNum=5, lastUserInput="Implement feature")
        )
        self.assertTrue(
            orchestrator_guard._load_state(self.base_payload).get("plan_execute_phase")
        )

        # A new lastUserInput resets plan_execute_phase and plan_allowed_files
        orchestrator_guard.handle_pre_invocation(
            dict(self.base_payload, invocationNum=6, lastUserInput="Next task")
        )
        st_new_input = orchestrator_guard._load_state(self.base_payload)
        self.assertFalse(st_new_input.get("plan_execute_phase"))
        self.assertIsNone(st_new_input.get("plan_allowed_files"))
        self.assertIsNone(st_new_input.get("plan_execute_user_input"))
        self.assertFalse(st_new_input.get("plan_execute_turn_finished"))

        # Idle root handle_stop with no active subagents marks turn finished; same lastUserInput then resets
        st_clear = orchestrator_guard._load_state(self.base_payload)
        st_clear["pending_spawns"] = []
        orchestrator_guard._save_state(self.base_payload, st_clear)
        orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                lastUserInput="Implement feature 2",
                toolCall={"name": "exit_plan_mode", "args": {"plan": plan_text}},
            )
        )
        orchestrator_guard.handle_pre_invocation(
            dict(self.base_payload, invocationNum=7, lastUserInput="Implement feature 2")
        )
        orchestrator_guard.handle_stop(dict(self.base_payload, fullyIdle=True))
        st_turn_done = orchestrator_guard._load_state(self.base_payload)
        self.assertTrue(st_turn_done.get("plan_execute_turn_finished"))
        orchestrator_guard.handle_pre_invocation(
            dict(self.base_payload, invocationNum=8, lastUserInput="Implement feature 2")
        )
        st_after_done = orchestrator_guard._load_state(self.base_payload)
        self.assertFalse(st_after_done.get("plan_execute_phase"))
        self.assertIsNone(st_after_done.get("plan_allowed_files"))

        # [plan review] still resets immediately
        orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                lastUserInput="Implement feature 3",
                toolCall={"name": "exit_plan_mode", "args": {"plan": plan_text}},
            )
        )
        orchestrator_guard.handle_pre_invocation(
            dict(self.base_payload, invocationNum=9, lastUserInput="Please run [plan review]")
        )
        st_review = orchestrator_guard._load_state(self.base_payload)
        self.assertFalse(st_review.get("plan_execute_phase"))
        self.assertIsNone(st_review.get("plan_allowed_files"))

    def test_phase2_share_workspace_tracking_stop_gate_and_kill_cleanup(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)
        share_cid = "11111111-2222-3333-4444-555555555555"
        orchestrator_guard.register_subagent(
            self.base_payload,
            conversation_id=share_cid,
            type_name="implementer",
            model="flash",
            workspace="share",
            role="Share Worker",
        )
        st = orchestrator_guard._load_state(self.base_payload)
        self.assertFalse(st["subagents"][share_cid]["active_branch"])
        self.assertTrue(st["subagents"][share_cid]["active_worktree"])

        # handle_stop blocks when a "share" worktree subagent is active
        res_stop = orchestrator_guard.handle_stop(
            dict(self.base_payload, terminationReason="NO_TOOL_CALL", fullyIdle=True)
        )
        self.assertEqual(res_stop["decision"], "continue")
        self.assertIn("Uncleaned branched subagent worktrees remain", res_stop["reason"])
        self.assertIn(share_cid, res_stop["reason"])

        # manage_subagents(kill) clears active_worktree on subagents and pending_spawns
        orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "name": "manage_subagents",
                    "args": {"Action": "kill", "ConversationIds": [share_cid]},
                },
            )
        )
        st_killed = orchestrator_guard._load_state(self.base_payload)
        self.assertFalse(st_killed["subagents"][share_cid]["active_worktree"])
        self.assertFalse(st_killed["subagents"][share_cid]["active_branch"])

        res_stop_after_kill = orchestrator_guard.handle_stop(
            dict(
                self.base_payload,
                executionNum=2,
                terminationReason="NO_TOOL_CALL",
                fullyIdle=True,
            )
        )
        self.assertEqual(res_stop_after_kill["decision"], "allow")

        # Also verify pending_spawns with Workspace="share" blocks Stop until kill_all
        spawn_res = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "share_pending_1",
                    "name": "invoke_subagent",
                    "args": {
                        "TypeName": "implementer",
                        "Role": "Share Pending",
                        "Model": "flash",
                        "Workspace": "share",
                        "Prompt": VALID_PROMPT,
                    },
                },
            )
        )
        self.assertEqual(spawn_res["decision"], "allow")
        st_pending = orchestrator_guard._load_state(self.base_payload)
        self.assertTrue(st_pending["pending_spawns"][0]["active_worktree"])
        self.assertFalse(st_pending["pending_spawns"][0]["active_branch"])

        res_stop_pending = orchestrator_guard.handle_stop(
            dict(
                self.base_payload,
                executionNum=3,
                terminationReason="NO_TOOL_CALL",
                fullyIdle=True,
            )
        )
        self.assertEqual(res_stop_pending["decision"], "continue")
        self.assertIn("Share Pending", res_stop_pending["reason"])

        orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={"name": "manage_subagents", "args": {"Action": "kill_all"}},
            )
        )
        res_stop_final = orchestrator_guard.handle_stop(
            dict(
                self.base_payload,
                executionNum=4,
                terminationReason="NO_TOOL_CALL",
                fullyIdle=True,
            )
        )
        self.assertEqual(res_stop_final["decision"], "allow")

    def test_phase2_best_of_n_mode_cohort_workspace_model_and_atomic_batch(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)

        # 1. MODE: BEST_OF_N with Workspace="inherit" is denied
        res_inherit = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "bon_inherit",
                    "name": "invoke_subagent",
                    "args": {
                        "TypeName": "implementer",
                        "Role": "Candidate Inherit",
                        "Workspace": "inherit",
                        "Prompt": "MODE: BEST_OF_N\nALLOWED_FILES: src/a.py\n" + VALID_PROMPT,
                    },
                },
            )
        )
        self.assertEqual(res_inherit["decision"], "deny")
        self.assertIn("MODE: BEST_OF_N requires isolated Workspace='branch' or Workspace='share'", res_inherit["reason"])

        # 2. Standard non-BEST_OF_N implementer still denies Model="pro"
        res_non_bon_pro = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "non_bon_pro",
                    "name": "invoke_subagent",
                    "args": {
                        "TypeName": "implementer",
                        "Role": "Standard Pro",
                        "Model": "pro",
                        "Workspace": "branch",
                        "Prompt": VALID_PROMPT,
                    },
                },
            )
        )
        self.assertEqual(res_non_bon_pro["decision"], "deny")
        self.assertIn("Model='pro'", res_non_bon_pro["reason"])

        # 3. MODE: BEST_OF_N without COHORT auto-generates best-of-n-<inv_key>,
        #    defaults omitted/inherit Model to flash, and allows explicit Model="pro"
        res_auto_cohort = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                invocationNum=7,
                toolCall={
                    "id": "bon_auto_batch",
                    "name": "invoke_subagent",
                    "args": {
                        "Subagents": [
                            {
                                "TypeName": "implementer",
                                "Role": "Cand A",
                                "Model": "pro",
                                "Workspace": "branch",
                                "Prompt": "MODE: BEST_OF_N\nALLOWED_FILES: src/a.py\n" + VALID_PROMPT,
                            },
                            {
                                "TypeName": "implementer",
                                "Role": "Cand B",
                                "Model": "inherit",
                                "Workspace": "share",
                                "Prompt": "MODE: best-of-n\nALLOWED_FILES: src/a.py\n" + VALID_PROMPT,
                            },
                        ]
                    },
                },
            )
        )
        self.assertEqual(res_auto_cohort["decision"], "allow")
        overwritten = res_auto_cohort.get("overwrite", {}).get("Subagents", [])
        self.assertEqual(overwritten[0]["Model"], "pro")
        self.assertEqual(overwritten[1]["Model"], "flash")

        st_auto = orchestrator_guard._load_state(self.base_payload)
        self.assertIn("best-of-n-7", st_auto["cohorts"])
        self.assertEqual(st_auto["cohorts"]["best-of-n-7"]["total_spawned"], 2)
        self.assertEqual(st_auto["pending_spawns"][0]["cohort"], "best-of-n-7")
        self.assertEqual(st_auto["pending_spawns"][0]["mode"], "best_of_n")
        self.assertTrue(st_auto["pending_spawns"][1]["active_worktree"])

        # 4. Explicit COHORT alone implies mode="best_of_n"
        res_cohort_only = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                invocationNum=8,
                toolCall={
                    "id": "bon_cohort_only",
                    "name": "invoke_subagent",
                    "args": {
                        "TypeName": "implementer",
                        "Role": "Cand C",
                        "Workspace": "share",
                        "Prompt": "COHORT: Cache_Fix\nALLOWED_FILES: src/a.py\n" + VALID_PROMPT,
                    },
                },
            )
        )
        self.assertEqual(res_cohort_only["decision"], "allow")
        st_c = orchestrator_guard._load_state(self.base_payload)
        self.assertIn("cache-fix", st_c["cohorts"])
        self.assertEqual(st_c["cohorts"]["cache-fix"]["total_spawned"], 1)

        # 5. Atomic batch rollback: valid entry #1 + invalid entry #2 leaves state untouched
        before_pending = len(st_c["pending_spawns"])
        before_calls = len(st_c["recorded_spawn_calls"])
        res_atomic_fail = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                invocationNum=9,
                toolCall={
                    "id": "bon_atomic_fail",
                    "name": "invoke_subagent",
                    "args": {
                        "Subagents": [
                            {
                                "TypeName": "implementer",
                                "Role": "Good Entry",
                                "Model": "pro",
                                "Workspace": "branch",
                                "Prompt": "COHORT: atomic-cohort\nEVAL_CMD: python3 scripts/eval.py\n" + VALID_PROMPT,
                            },
                            {
                                "TypeName": "implementer",
                                "Role": "Bad Inherit Entry",
                                "Workspace": "inherit",
                                "Prompt": "COHORT: atomic-cohort\n" + VALID_PROMPT,
                            },
                        ]
                    },
                },
            )
        )
        self.assertEqual(res_atomic_fail["decision"], "deny")
        st_after_fail = orchestrator_guard._load_state(self.base_payload)
        self.assertNotIn("atomic-cohort", st_after_fail["cohorts"])
        self.assertEqual(len(st_after_fail["pending_spawns"]), before_pending)
        self.assertEqual(len(st_after_fail["recorded_spawn_calls"]), before_calls)
        self.assertIsNone(st_after_fail.get("active_eval_cmd"))

    def test_phase2_cohort_circuit_breaker_per_candidate_and_async_sync(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)

        # Spawn 5 candidates in cohort "five-way"
        for i in range(1, 6):
            res = orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={
                        "id": f"five_{i}",
                        "name": "invoke_subagent",
                        "args": {
                            "TypeName": "implementer",
                            "Role": f"Cand {i}",
                            "Workspace": "share",
                            "Prompt": f"MODE: BEST_OF_N\nCOHORT: five-way\n{VALID_PROMPT}",
                        },
                    },
                )
            )
            self.assertEqual(res["decision"], "allow")

        # 2/5 candidate failures must NOT trip the global circuit breaker
        orchestrator_guard.handle_post_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "five_1",
                    "name": "invoke_subagent",
                    "args": {"TypeName": "implementer", "Prompt": "COHORT: five-way\n" + VALID_PROMPT},
                },
                error="candidate 1 crashed",
            )
        )
        orchestrator_guard.handle_post_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "five_2",
                    "name": "invoke_subagent",
                    "args": {"TypeName": "implementer", "Prompt": "COHORT: five-way\n" + VALID_PROMPT},
                },
                output="STATUS: FAILED\nFILES_CHANGED: NONE\nVERIFICATION: pytest\nNOTES: failed tests see file:///tmp/fail.log",
            )
        )
        st_mid = orchestrator_guard._load_state(self.base_payload)
        self.assertFalse(st_mid["circuit_breaker"]["tripped"])
        self.assertEqual(st_mid["circuit_breaker"]["failure_streak"], 0)
        self.assertEqual(len(st_mid["cohorts"]["five-way"]["failed"]), 2)

        # Spawn a 2-candidate cohort "two-way" and fail 2/2 -> trips global breaker
        for i in range(1, 3):
            orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={
                        "id": f"two_{i}",
                        "name": "invoke_subagent",
                        "args": {
                            "TypeName": "implementer",
                            "Role": f"Two {i}",
                            "Workspace": "branch",
                            "Prompt": f"MODE: BEST_OF_N\nCOHORT: two-way\n{VALID_PROMPT}",
                        },
                    },
                )
            )
        for i in range(1, 3):
            orchestrator_guard.handle_post_tool(
                dict(
                    self.base_payload,
                    toolCall={
                        "id": f"two_{i}",
                        "name": "invoke_subagent",
                        "args": {"TypeName": "implementer", "Prompt": "COHORT: two-way\n" + VALID_PROMPT},
                    },
                    error=f"two_{i} failed",
                )
            )
        st_tripped = orchestrator_guard._load_state(self.base_payload)
        self.assertTrue(st_tripped["circuit_breaker"]["tripped"])
        self.assertGreaterEqual(st_tripped["circuit_breaker"]["failure_streak"], 2)
        self.assertEqual(len(st_tripped["cohorts"]["two-way"]["failed"]), 2)

        # Async brain sibling sync + local candidate 2-turn circuit breaker
        brain_base = Path(self.artifact_dir) / "brain"
        parent_dir = brain_base / "parent-bon"
        cand1_dir = brain_base / "cand-async-1"
        cand2_dir = brain_base / "cand-async-2"
        parent_dir.mkdir(parents=True, exist_ok=True)
        cand1_dir.mkdir(parents=True, exist_ok=True)
        cand2_dir.mkdir(parents=True, exist_ok=True)

        parent_payload = {
            "conversationId": "parent-bon",
            "workspacePaths": [str(orchestrator_guard.PROJECT_ROOT)],
            "artifactDirectoryPath": str(parent_dir),
            "invocationNum": 1,
            "executionNum": 1,
        }
        orchestrator_guard._cli_activate(parent_payload)
        orchestrator_guard.handle_pre_invocation(parent_payload)
        for cid_idx, c_cid in enumerate(("cand-async-1", "cand-async-2"), start=1):
            orchestrator_guard.register_subagent(
                parent_payload,
                conversation_id=c_cid,
                type_name="implementer",
                model="flash",
                workspace="share",
                role=f"Async {cid_idx}",
                status="active",
                mode="best_of_n",
                cohort="async-cohort",
            )
        p_state = orchestrator_guard._load_state(parent_payload)
        p_state["cohorts"]["async-cohort"] = {
            "total_spawned": 2,
            "succeeded": [],
            "failed": [],
            "created_at": 100.0,
        }
        orchestrator_guard._save_state(parent_payload, p_state)

        # Candidate 1 trips its own local 2-turn circuit breaker
        cand1_payload = {
            "conversationId": "cand-async-1",
            "workspacePaths": [str(orchestrator_guard.PROJECT_ROOT)],
            "artifactDirectoryPath": str(cand1_dir),
            "isSubagent": True,
            "invocationNum": 1,
            "executionNum": 1,
        }
        for step in (1, 2):
            orchestrator_guard.handle_post_tool(
                dict(
                    cand1_payload,
                    toolCall={
                        "id": f"c1_cmd_{step}",
                        "name": "run_command",
                        "args": {"CommandLine": "pytest tests/missing.py"},
                    },
                    exitCode=1,
                )
            )
        st_cand1 = orchestrator_guard._load_state(cand1_payload)
        self.assertTrue(st_cand1["circuit_breaker"]["tripped"])

        # Candidate 1 stops -> parent cohort records cand-async-1 as failed, parent breaker stays untripped (1/2)
        res_c1_stop = orchestrator_guard.handle_stop(
            dict(
                cand1_payload,
                fullyIdle=True,
                lastAssistantMessage=(
                    "STATUS: BLOCKED\nFILES_CHANGED: NONE\nVERIFICATION: pytest failed\n"
                    "NOTES: Candidate failed twice see file:///tmp/c1.log"
                ),
            )
        )
        self.assertEqual(res_c1_stop["decision"], "allow")
        p_after_c1 = orchestrator_guard._load_state(parent_payload)
        self.assertEqual(p_after_c1["cohorts"]["async-cohort"]["failed"], ["cand-async-1"])
        self.assertFalse(p_after_c1["circuit_breaker"]["tripped"])

        # Candidate 2 succeeds cleanly -> parent cohort records succeeded
        cand2_payload = {
            "conversationId": "cand-async-2",
            "workspacePaths": [str(orchestrator_guard.PROJECT_ROOT)],
            "artifactDirectoryPath": str(cand2_dir),
            "isSubagent": True,
            "invocationNum": 1,
            "executionNum": 1,
        }
        res_c2_stop = orchestrator_guard.handle_stop(
            dict(
                cand2_payload,
                fullyIdle=True,
                lastAssistantMessage=(
                    "STATUS: COMPLETED\nFILES_CHANGED: src/a.py\nVERIFICATION: pytest passed\n"
                    "NOTES: SCORE: 1.0"
                ),
            )
        )
        self.assertEqual(res_c2_stop["decision"], "allow")
        p_after_c2 = orchestrator_guard._load_state(parent_payload)
        self.assertEqual(p_after_c2["cohorts"]["async-cohort"]["succeeded"], ["cand-async-2"])
        self.assertFalse(p_after_c2["circuit_breaker"]["tripped"])

        # State pruning caps cohorts and succeeded/failed lists
        prune_state = orchestrator_guard._default_state()
        for idx in range(orchestrator_guard.MAX_STATE_INVOCATION_HISTORY + 5):
            prune_state["cohorts"][f"cohort-{idx}"] = {
                "total_spawned": 1,
                "succeeded": [f"s-{j}" for j in range(orchestrator_guard.MAX_STATE_LIST_ENTRIES + 5)],
                "failed": [f"f-{j}" for j in range(orchestrator_guard.MAX_STATE_LIST_ENTRIES + 5)],
                "created_at": float(idx),
            }
        orchestrator_guard._prune_state(prune_state)
        self.assertEqual(len(prune_state["cohorts"]), orchestrator_guard.MAX_STATE_INVOCATION_HISTORY)
        self.assertNotIn("cohort-0", prune_state["cohorts"])
        newest_cohort = prune_state["cohorts"][f"cohort-{orchestrator_guard.MAX_STATE_INVOCATION_HISTORY + 4}"]
        self.assertEqual(len(newest_cohort["succeeded"]), orchestrator_guard.MAX_STATE_LIST_ENTRIES)
        self.assertEqual(len(newest_cohort["failed"]), orchestrator_guard.MAX_STATE_LIST_ENTRIES)

    def test_phase2_candidate_receipt_isolation(self) -> None:
        brain_base = Path(self.artifact_dir) / "brain_iso"
        parent_dir = brain_base / "parent-iso"
        cand_dir = brain_base / "cand-iso"
        parent_dir.mkdir(parents=True, exist_ok=True)
        cand_dir.mkdir(parents=True, exist_ok=True)

        parent_payload = {
            "conversationId": "parent-iso",
            "workspacePaths": [str(orchestrator_guard.PROJECT_ROOT)],
            "artifactDirectoryPath": str(parent_dir),
            "invocationNum": 1,
            "executionNum": 1,
        }
        orchestrator_guard._cli_activate(parent_payload)
        orchestrator_guard.handle_pre_invocation(parent_payload)
        p_state = orchestrator_guard._load_state(parent_payload)
        p_state["test_receipt"] = {
            "dirty": True,
            "verified": False,
            "last_mutated_file": "src/main.py",
            "last_test_cmd": "",
            "mutation_epoch": 4,
        }
        orchestrator_guard._save_state(parent_payload, p_state)

        orchestrator_guard.register_subagent(
            parent_payload,
            conversation_id="cand-iso",
            type_name="implementer",
            model="flash",
            workspace="branch",
            role="Iso Candidate",
            status="active",
            mode="best_of_n",
            cohort="iso-cohort",
        )

        # Candidate mutates a workspace code file -> parent dirty/epoch stays unchanged
        cand_payload = {
            "conversationId": "cand-iso",
            "workspacePaths": [str(orchestrator_guard.PROJECT_ROOT)],
            "artifactDirectoryPath": str(cand_dir),
            "isSubagent": True,
            "invocationNum": 1,
            "executionNum": 1,
        }
        target_py = orchestrator_guard.PROJECT_ROOT / "src" / "example_iso.py"
        res_edit = orchestrator_guard.handle_pre_tool(
            dict(
                cand_payload,
                toolCall={
                    "id": "cand_edit_1",
                    "name": "edit_file",
                    "args": {"Path": str(target_py), "Target": "a", "Replacement": "b"},
                },
            )
        )
        self.assertEqual(res_edit["decision"], "allow")
        p_after_edit = orchestrator_guard._load_state(parent_payload)
        self.assertEqual(p_after_edit["test_receipt"]["mutation_epoch"], 4)
        self.assertEqual(p_after_edit["test_receipt"]["last_mutated_file"], "src/main.py")

        # Candidate runs passing test -> clears candidate's receipt, NOT parent's dirty receipt
        orchestrator_guard.handle_post_tool(
            dict(
                cand_payload,
                toolCall={
                    "id": "cand_test_1",
                    "name": "run_command",
                    "args": {"CommandLine": "python3 -m unittest discover -s tests"},
                },
                exitCode=0,
            )
        )
        c_after_test = orchestrator_guard._load_state(cand_payload)
        self.assertFalse(c_after_test["test_receipt"]["dirty"])
        self.assertTrue(c_after_test["test_receipt"]["verified"])
        p_after_cand_test = orchestrator_guard._load_state(parent_payload)
        self.assertTrue(p_after_cand_test["test_receipt"]["dirty"])
        self.assertFalse(p_after_cand_test["test_receipt"]["verified"])

        # Synchronous best_of_n completion in parent PostToolUse does NOT clear parent's dirty receipt
        orchestrator_guard.handle_post_tool(
            dict(
                parent_payload,
                toolCall={
                    "id": "sync_bon_1",
                    "name": "invoke_subagent",
                    "args": {
                        "TypeName": "implementer",
                        "Workspace": "branch",
                        "Prompt": "MODE: BEST_OF_N\nCOHORT: iso-cohort\n" + VALID_PROMPT,
                    },
                },
                output="STATUS: COMPLETED\nFILES_CHANGED: src/main.py\nVERIFICATION: pytest ok\nNOTES: SCORE: 1.0",
            )
        )
        p_after_sync = orchestrator_guard._load_state(parent_payload)
        self.assertTrue(p_after_sync["test_receipt"]["dirty"])
        self.assertFalse(p_after_sync["test_receipt"]["verified"])

    def test_phase2_eval_cmd_verification_and_anti_tampering(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)

        # 1. Trivial EVAL_CMD commands are rejected in invoke_subagent and exit_plan_mode
        for bad_cmd in ("true", ":", "echo pass", "printf ok", "exit 0", "pytest || true", "VAR=1 :"):
            res_bad_spawn = orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={
                        "id": f"bad_{bad_cmd}",
                        "name": "invoke_subagent",
                        "args": {
                            "TypeName": "implementer",
                            "Workspace": "share",
                            "Prompt": f"MODE: BEST_OF_N\nEVAL_CMD: {bad_cmd}\n{VALID_PROMPT}",
                        },
                    },
                )
            )
            self.assertEqual(res_bad_spawn["decision"], "deny", f"Expected deny for {bad_cmd}")
            self.assertIn("Trivial or no-op EVAL_CMD rejected", res_bad_spawn["reason"])

            res_bad_plan = orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={
                        "name": "exit_plan_mode",
                        "args": {"plan": f"EVAL_CMD: {bad_cmd}\n1. Run eval"},
                    },
                )
            )
            self.assertEqual(res_bad_plan["decision"], "deny", f"Expected plan deny for {bad_cmd}")

        # 2. Valid EVAL_CMD in exit_plan_mode sets active_eval_cmd, cleared on enter_plan_mode / reset
        res_good_plan = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                lastUserInput="Optimize solver",
                toolCall={
                    "name": "exit_plan_mode",
                    "args": {"plan": "EVAL_CMD: python3 scripts/bench.py\n1. Optimize"},
                },
            )
        )
        self.assertEqual(res_good_plan["decision"], "allow")
        st_plan = orchestrator_guard._load_state(self.base_payload)
        self.assertEqual(st_plan["active_eval_cmd"], "python3 scripts/bench.py")

        # Root edit sets dirty=True; declared EVAL_CMD clears dirty=False & verified=True
        st_plan["test_receipt"]["dirty"] = True
        st_plan["test_receipt"]["verified"] = False
        orchestrator_guard._save_state(self.base_payload, st_plan)

        # Anti-tampering: bypasses must NOT count as declared EVAL_CMD
        for bypass_cmd in (
            "python3 scripts/bench.py || true",
            "$(python3 scripts/bench.py)",
            "`python3 scripts/bench.py`",
            "python3 scripts/bench.py_fake",
        ):
            self.assertFalse(
                orchestrator_guard._is_declared_eval_command(
                    bypass_cmd, st_plan, self.base_payload
                ),
                f"Bypass should not match: {bypass_cmd}",
            )

        # Valid env-prefixed or argument-extended declared EVAL_CMD passes PostToolUse
        orchestrator_guard.handle_post_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "eval_run_1",
                    "name": "run_command",
                    "args": {"CommandLine": "PYTHONPATH=. python3 scripts/bench.py --fast"},
                },
                exitCode=0,
            )
        )
        st_verified = orchestrator_guard._load_state(self.base_payload)
        self.assertFalse(st_verified["test_receipt"]["dirty"])
        self.assertTrue(st_verified["test_receipt"]["verified"])
        self.assertEqual(
            st_verified["test_receipt"]["last_test_cmd"],
            "PYTHONPATH=. python3 scripts/bench.py --fast",
        )

        # enter_plan_mode and PreInvocation reset clear active_eval_cmd
        orchestrator_guard.handle_pre_tool(
            dict(self.base_payload, toolCall={"name": "enter_plan_mode", "args": {}})
        )
        self.assertIsNone(orchestrator_guard._load_state(self.base_payload)["active_eval_cmd"])

        orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                lastUserInput="Optimize solver 2",
                toolCall={
                    "name": "exit_plan_mode",
                    "args": {"plan": "EVAL_CMD: python3 scripts/bench.py\n1. Optimize"},
                },
            )
        )
        orchestrator_guard.handle_pre_invocation(
            dict(self.base_payload, invocationNum=2, lastUserInput="Please run [plan review]")
        )
        self.assertIsNone(orchestrator_guard._load_state(self.base_payload)["active_eval_cmd"])

        # Root direct budget counts declared EVAL_CMD as a test run after edit
        st_direct = orchestrator_guard._load_state(self.base_payload)
        st_direct["active_eval_cmd"] = "python3 scripts/bench.py"
        st_direct["direct_budget"]["3"] = {
            "mutated_files": ["src/solver.py"],
            "read_files": [],
            "read_ranges": [],
            "read_lines": 0,
            "read_bytes": 0,
            "commands": [],
            "seen_calls": [],
            "test_runs_after_edit": 0,
        }
        orchestrator_guard._save_state(self.base_payload, st_direct)
        res_eval_1 = orchestrator_guard._check_and_record_direct_budget(
            dict(self.base_payload, invocationNum=3),
            "run_command",
            {"CommandLine": "python3 scripts/bench.py"},
            orchestrator_guard._load_state(self.base_payload),
        )
        self.assertIsNone(res_eval_1)
        res_eval_2 = orchestrator_guard._check_and_record_direct_budget(
            dict(self.base_payload, invocationNum=3),
            "run_command",
            {"CommandLine": "python3 scripts/bench.py"},
            orchestrator_guard._load_state(self.base_payload),
        )
        self.assertIsNotNone(res_eval_2)
        self.assertEqual(res_eval_2["decision"], "deny")

    def test_phase2_cli_status_surfaces_cohorts_and_active_eval_cmd(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)
        res_spawn = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                invocationNum=4,
                toolCall={
                    "id": "status_bon_1",
                    "name": "invoke_subagent",
                    "args": {
                        "TypeName": "implementer",
                        "Role": "Status Cand",
                        "Workspace": "share",
                        "Prompt": (
                            "MODE: BEST_OF_N\nCOHORT: status-cohort\n"
                            "EVAL_CMD: python3 scripts/eval_score.py\n" + VALID_PROMPT
                        ),
                    },
                },
            )
        )
        self.assertEqual(res_spawn["decision"], "allow")
        status_out = orchestrator_guard._cli_status(self.base_payload)
        self.assertIn("cohorts", status_out)
        self.assertIn("activeEvalCmd", status_out)
        self.assertEqual(status_out["activeEvalCmd"], "python3 scripts/eval_score.py")
        self.assertIn("status-cohort", status_out["cohorts"])
        self.assertEqual(status_out["cohorts"]["status-cohort"]["total_spawned"], 1)
        self.assertIn("activeSubagents", status_out)
        self.assertIn("testReceipt", status_out)
        self.assertIn("circuitBreaker", status_out)

    def test_phase2_empty_call_id_cohort_single_failure_does_not_overmatch(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)

        # Spawn 3 candidates with empty call_id in cohort "no-id-cohort"
        for i in range(1, 4):
            res = orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={
                        "name": "invoke_subagent",
                        "args": {
                            "TypeName": "implementer",
                            "Role": f"Cand {i}",
                            "Workspace": "share",
                            "Prompt": f"MODE: BEST_OF_N\nCOHORT: no-id-cohort\n{VALID_PROMPT}",
                        },
                    },
                )
            )
            self.assertEqual(res["decision"], "allow")

        st_spawned = orchestrator_guard._load_state(self.base_payload)
        self.assertEqual(st_spawned["cohorts"]["no-id-cohort"]["total_spawned"], 3)
        self.assertEqual(
            [p.get("candidate_id") for p in st_spawned["pending_spawns"]],
            ["no-id-cohort#1", "no-id-cohort#2", "no-id-cohort#3"],
        )

        # 1 failing candidate with empty call_id must only fail 1 candidate, not all 3
        orchestrator_guard.handle_post_tool(
            dict(
                self.base_payload,
                toolCall={
                    "name": "invoke_subagent",
                    "args": {
                        "TypeName": "implementer",
                        "Role": "Cand 2",
                        "Prompt": f"MODE: BEST_OF_N\nCOHORT: no-id-cohort\n{VALID_PROMPT}",
                    },
                },
                error="cand 2 crashed",
            )
        )
        st_after_fail = orchestrator_guard._load_state(self.base_payload)
        self.assertEqual(
            st_after_fail["cohorts"]["no-id-cohort"]["failed"],
            ["no-id-cohort#2"],
        )
        self.assertFalse(st_after_fail["circuit_breaker"]["tripped"])
        self.assertEqual(
            [p.get("candidate_id") for p in st_after_fail["pending_spawns"]],
            ["no-id-cohort#1", "no-id-cohort#3"],
        )

        # 1 synchronous completion with empty call_id must mark at most 1 pending entry completed
        orchestrator_guard.handle_post_tool(
            dict(
                self.base_payload,
                toolCall={
                    "name": "invoke_subagent",
                    "args": {
                        "TypeName": "implementer",
                        "Role": "Cand 1",
                        "Prompt": f"MODE: BEST_OF_N\nCOHORT: no-id-cohort\n{VALID_PROMPT}",
                    },
                },
                output="STATUS: COMPLETED\nFILES_CHANGED: src/a.py\nVERIFICATION: pytest ok\nNOTES: SCORE: 1.0",
            )
        )
        st_after_ok = orchestrator_guard._load_state(self.base_payload)
        self.assertEqual(
            st_after_ok["cohorts"]["no-id-cohort"]["succeeded"],
            ["no-id-cohort#1"],
        )
        statuses = {
            p["candidate_id"]: p["status"]
            for p in st_after_ok["pending_spawns"]
        }
        self.assertEqual(statuses, {"no-id-cohort#1": "completed", "no-id-cohort#3": "active"})

        # Async handle_stop deduplicates against candidate_id in parent cohorts
        brain_base = Path(self.artifact_dir) / "brain_dedup"
        parent_dir = brain_base / "parent-dedup"
        cand_dir = brain_base / "cand-dedup-1"
        parent_dir.mkdir(parents=True, exist_ok=True)
        cand_dir.mkdir(parents=True, exist_ok=True)
        parent_payload = {
            "conversationId": "parent-dedup",
            "workspacePaths": [str(orchestrator_guard.PROJECT_ROOT)],
            "artifactDirectoryPath": str(parent_dir),
            "invocationNum": 1,
            "executionNum": 1,
        }
        orchestrator_guard._cli_activate(parent_payload)
        orchestrator_guard.handle_pre_invocation(parent_payload)
        orchestrator_guard.register_subagent(
            parent_payload,
            conversation_id="cand-dedup-1",
            type_name="implementer",
            model="flash",
            workspace="share",
            role="Cand 1",
            status="active",
            mode="best_of_n",
            cohort="dedup-cohort",
            candidate_id="dedup-cohort#1",
        )
        p_st = orchestrator_guard._load_state(parent_payload)
        p_st["cohorts"]["dedup-cohort"] = {
            "total_spawned": 2,
            "succeeded": ["dedup-cohort#1"],
            "failed": [],
            "created_at": 100.0,
        }
        orchestrator_guard._save_state(parent_payload, p_st)

        cand_payload = {
            "conversationId": "cand-dedup-1",
            "workspacePaths": [str(orchestrator_guard.PROJECT_ROOT)],
            "artifactDirectoryPath": str(cand_dir),
            "isSubagent": True,
            "invocationNum": 1,
            "executionNum": 1,
        }
        res_stop = orchestrator_guard.handle_stop(
            dict(
                cand_payload,
                fullyIdle=True,
                lastAssistantMessage=(
                    "STATUS: COMPLETED\nFILES_CHANGED: src/a.py\nVERIFICATION: pytest ok\nNOTES: SCORE: 1.0"
                ),
            )
        )
        self.assertEqual(res_stop["decision"], "allow")
        p_after = orchestrator_guard._load_state(parent_payload)
        self.assertEqual(p_after["cohorts"]["dedup-cohort"]["succeeded"], ["dedup-cohort#1"])

    def test_phase2_run_workflow_guard_integration(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)

        # Cwd / ScriptPath outside allowed roots -> ask
        res_bad_cwd = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "wf_bad_cwd",
                    "name": "run_workflow",
                    "args": {"Cwd": "/etc", "Description": "Outside cwd"},
                },
            )
        )
        self.assertEqual(res_bad_cwd["decision"], "ask")

        res_bad_script = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "wf_bad_script",
                    "name": "run_workflow",
                    "args": {
                        "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                        "ScriptPath": "/etc/passwd",
                        "Description": "Outside script",
                    },
                },
            )
        )
        self.assertEqual(res_bad_script["decision"], "ask")

        # Trivial EVAL_CMD in Description -> deny
        for bad_eval in ("true", "pytest || true"):
            res_bad_eval = orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={
                        "id": f"wf_bad_eval_{bad_eval}",
                        "name": "run_workflow",
                        "args": {
                            "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                            "Description": f"EVAL_CMD: {bad_eval}\nRun workflow",
                        },
                    },
                )
            )
            self.assertEqual(res_bad_eval["decision"], "deny")
            self.assertIn("Trivial or no-op EVAL_CMD rejected", res_bad_eval["reason"])

        # Valid run_workflow with EVAL_CMD -> allow and persist active_eval_cmd
        res_good_wf = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "wf_good_1",
                    "name": "run_workflow",
                    "args": {
                        "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                        "ScriptPath": str(orchestrator_guard.PROJECT_ROOT / "scripts/wf.sh"),
                        "Description": "EVAL_CMD: python3 scripts/eval_wf.py\nRun eval workflow",
                    },
                },
            )
        )
        self.assertEqual(res_good_wf["decision"], "allow")
        st_wf = orchestrator_guard._load_state(self.base_payload)
        self.assertEqual(st_wf["active_eval_cmd"], "python3 scripts/eval_wf.py")

        # Dirty test_receipt must NOT be cleared by run_workflow success
        st_wf["test_receipt"]["dirty"] = True
        st_wf["test_receipt"]["verified"] = False
        orchestrator_guard._save_state(self.base_payload, st_wf)

        orchestrator_guard.handle_post_tool(
            dict(
                self.base_payload,
                toolCall={"id": "wf_good_1", "name": "run_workflow", "args": {}},
                exitCode=0,
            )
        )
        st_after_ok = orchestrator_guard._load_state(self.base_payload)
        self.assertTrue(st_after_ok["test_receipt"]["dirty"])
        self.assertFalse(st_after_ok["test_receipt"]["verified"])

        # Sandbox boundary error does not trip breaker; 2 real failures trip breaker
        orchestrator_guard.handle_post_tool(
            dict(
                self.base_payload,
                toolCall={"id": "wf_sandbox", "name": "run_workflow", "args": {}},
                error="There were sandbox errors that may or may not be related to the failure",
            )
        )
        self.assertEqual(
            orchestrator_guard._load_state(self.base_payload)["circuit_breaker"]["failure_streak"],
            0,
        )

        orchestrator_guard.handle_post_tool(
            dict(
                self.base_payload,
                toolCall={"id": "wf_fail_1", "name": "run_workflow", "args": {}},
                error="workflow step 1 failed",
            )
        )
        orchestrator_guard.handle_post_tool(
            dict(
                self.base_payload,
                toolCall={"id": "wf_fail_2", "name": "run_workflow", "args": {}},
                exitCode=1,
            )
        )
        st_tripped = orchestrator_guard._load_state(self.base_payload)
        self.assertEqual(st_tripped["circuit_breaker"]["failure_streak"], 2)
        self.assertTrue(st_tripped["circuit_breaker"]["tripped"])

        # Tripped breaker blocks run_workflow without REPLAN:, allows 1-shot REPLAN:
        res_blocked = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "wf_blocked",
                    "name": "run_workflow",
                    "args": {
                        "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                        "Description": "Retry workflow",
                    },
                },
            )
        )
        self.assertEqual(res_blocked["decision"], "deny")
        self.assertIn("REPLAN:", res_blocked["reason"])

        res_replan = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "wf_replan",
                    "name": "run_workflow",
                    "args": {
                        "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                        "Description": "REPLAN: switch to fallback workflow\nRun v2",
                    },
                },
            )
        )
        self.assertEqual(res_replan["decision"], "allow")
        st_replanned = orchestrator_guard._load_state(self.base_payload)
        self.assertFalse(st_replanned["circuit_breaker"]["tripped"])
        self.assertEqual(st_replanned["circuit_breaker"]["replan_count"], 1)

        # Success resets failure_streak after a single failure while keeping dirty receipt
        orchestrator_guard.handle_post_tool(
            dict(
                self.base_payload,
                toolCall={"id": "wf_fail_3", "name": "run_workflow", "args": {}},
                error="transient wf failure",
            )
        )
        self.assertEqual(
            orchestrator_guard._load_state(self.base_payload)["circuit_breaker"]["failure_streak"],
            1,
        )
        orchestrator_guard.handle_post_tool(
            dict(
                self.base_payload,
                toolCall={"id": "wf_ok_reset", "name": "run_workflow", "args": {}},
                exitCode=0,
            )
        )
        st_reset = orchestrator_guard._load_state(self.base_payload)
        self.assertEqual(st_reset["circuit_breaker"]["failure_streak"], 0)
        self.assertFalse(st_reset["circuit_breaker"]["tripped"])
        self.assertTrue(st_reset["test_receipt"]["dirty"])

    def test_phase2_empty_call_id_out_of_order_prompt_hash_attribution(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)

        prompts = [
            f"MODE: BEST_OF_N\nCOHORT: hash-cohort\n{VALID_PROMPT}\n# approach A",
            f"MODE: BEST_OF_N\nCOHORT: hash-cohort\n{VALID_PROMPT}\n# approach B",
            f"MODE: BEST_OF_N\nCOHORT: hash-cohort\n{VALID_PROMPT}\n# approach C",
        ]
        for prompt in prompts:
            res = orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={
                        "name": "invoke_subagent",
                        "args": {
                            "TypeName": "implementer",
                            "Role": "Candidate",
                            "Workspace": "share",
                            "Prompt": prompt,
                        },
                    },
                )
            )
            self.assertEqual(res["decision"], "allow")

        st_spawned = orchestrator_guard._load_state(self.base_payload)
        self.assertEqual(
            [p.get("candidate_id") for p in st_spawned["pending_spawns"]],
            ["hash-cohort#1", "hash-cohort#2", "hash-cohort#3"],
        )
        self.assertEqual(
            len({p.get("prompt_hash") for p in st_spawned["pending_spawns"] if p.get("prompt_hash")}),
            3,
        )

        # Fail candidate #2 (approach B) first with empty call_id and no UUID
        orchestrator_guard.handle_post_tool(
            dict(
                self.base_payload,
                toolCall={
                    "name": "invoke_subagent",
                    "args": {
                        "TypeName": "implementer",
                        "Role": "Candidate",
                        "Prompt": prompts[1],
                    },
                },
                error="candidate B failed",
            )
        )
        st_after_fail = orchestrator_guard._load_state(self.base_payload)
        self.assertEqual(
            st_after_fail["cohorts"]["hash-cohort"]["failed"],
            ["hash-cohort#2"],
        )
        self.assertEqual(
            [p.get("candidate_id") for p in st_after_fail["pending_spawns"]],
            ["hash-cohort#1", "hash-cohort#3"],
        )

        # Succeed candidate #3 (approach C) out of order with empty call_id
        orchestrator_guard.handle_post_tool(
            dict(
                self.base_payload,
                toolCall={
                    "name": "invoke_subagent",
                    "args": {
                        "TypeName": "implementer",
                        "Role": "Candidate",
                        "Prompt": prompts[2],
                    },
                },
                output="STATUS: COMPLETED\nFILES_CHANGED: src/auth/parser.py\nVERIFICATION: pytest ok\nNOTES: SCORE: 0.9",
            )
        )
        st_after_c = orchestrator_guard._load_state(self.base_payload)
        self.assertEqual(
            st_after_c["cohorts"]["hash-cohort"]["succeeded"],
            ["hash-cohort#3"],
        )
        statuses = {
            p["candidate_id"]: p["status"]
            for p in st_after_c["pending_spawns"]
        }
        self.assertEqual(
            statuses,
            {"hash-cohort#1": "active", "hash-cohort#3": "completed"},
        )

        # Explicitly complete candidate #1 (approach A)
        orchestrator_guard.handle_post_tool(
            dict(
                self.base_payload,
                toolCall={
                    "name": "invoke_subagent",
                    "args": {
                        "TypeName": "implementer",
                        "Role": "Candidate",
                        "Prompt": prompts[0],
                    },
                },
                output="STATUS: COMPLETED\nFILES_CHANGED: src/auth/parser.py\nVERIFICATION: pytest ok\nNOTES: SCORE: 1.0",
            )
        )
        st_after_a = orchestrator_guard._load_state(self.base_payload)
        self.assertEqual(
            st_after_a["cohorts"]["hash-cohort"]["succeeded"],
            ["hash-cohort#3", "hash-cohort#1"],
        )
        self.assertEqual(
            {p["candidate_id"]: p["status"] for p in st_after_a["pending_spawns"]},
            {"hash-cohort#1": "completed", "hash-cohort#3": "completed"},
        )

    def test_phase2_run_workflow_script_eval_cmd_and_traceback_detection(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)

        # Trivial # EVAL_CMD: true inside Script -> deny
        res_bad = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "wf_script_bad",
                    "name": "run_workflow",
                    "args": {
                        "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                        "Description": "Run eval script",
                        "Script": "#!/usr/bin/env bash\n# EVAL_CMD: true\necho hi\n",
                    },
                },
            )
        )
        self.assertEqual(res_bad["decision"], "deny")
        self.assertIn("Trivial or no-op EVAL_CMD rejected", res_bad["reason"])

        # Valid # EVAL_CMD: python3 evals/bench.py inside Script -> allow and persist active_eval_cmd
        res_good = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "wf_script_good",
                    "name": "run_workflow",
                    "args": {
                        "Cwd": str(orchestrator_guard.PROJECT_ROOT),
                        "Description": "Run benchmark script",
                        "Script": "#!/usr/bin/env bash\n# EVAL_CMD: python3 evals/bench.py\npython3 evals/bench.py\n",
                    },
                },
            )
        )
        self.assertEqual(res_good["decision"], "allow")
        st_good = orchestrator_guard._load_state(self.base_payload)
        self.assertEqual(st_good["active_eval_cmd"], "python3 evals/bench.py")

        # Two PostToolUse run_workflow calls with Python traceback in output increment & trip breaker
        orchestrator_guard.handle_post_tool(
            dict(
                self.base_payload,
                toolCall={"id": "wf_tb_1", "name": "run_workflow", "args": {}},
                output="Traceback (most recent call last):\nRuntimeError: boom",
            )
        )
        st_tb_1 = orchestrator_guard._load_state(self.base_payload)
        self.assertEqual(st_tb_1["circuit_breaker"]["failure_streak"], 1)
        self.assertFalse(st_tb_1["circuit_breaker"]["tripped"])

        orchestrator_guard.handle_post_tool(
            dict(
                self.base_payload,
                toolCall={"id": "wf_tb_2", "name": "run_workflow", "args": {}},
                output="Traceback (most recent call last):\nRuntimeError: boom",
            )
        )
        st_tb_2 = orchestrator_guard._load_state(self.base_payload)
        self.assertEqual(st_tb_2["circuit_breaker"]["failure_streak"], 2)
        self.assertTrue(st_tb_2["circuit_breaker"]["tripped"])

    def test_phase3_judge_define_subagent_and_model_policy(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)

        # define_subagent strips Write, Subagent, and MCP tools for judge (PascalCase + snake_case)
        res_pascal = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "name": "define_subagent",
                    "args": {
                        "Name": "judge",
                        "EnableWriteTools": True,
                        "EnableSubagentTools": True,
                        "EnableMCPTools": True,
                    },
                },
            )
        )
        self.assertEqual(res_pascal["decision"], "allow")
        self.assertEqual(
            res_pascal["overwrite"],
            {
                "EnableSubagentTools": False,
                "EnableWriteTools": False,
                "EnableMCPTools": False,
            },
        )

        res_snake = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "name": "define_subagent",
                    "args": {
                        "name": "judge",
                        "enable_write_tools": True,
                        "enable_subagent_tools": True,
                        "enable_mcp_tools": True,
                    },
                },
            )
        )
        self.assertEqual(res_snake["decision"], "allow")
        self.assertEqual(
            res_snake["overwrite"],
            {
                "enable_subagent_tools": False,
                "enable_write_tools": False,
                "enable_mcp_tools": False,
            },
        )

        # define_subagent strips Subagent and MCP tools (but not WriteTools) for council
        res_council_def = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "name": "define_subagent",
                    "args": {
                        "Name": "council",
                        "EnableWriteTools": True,
                        "EnableSubagentTools": True,
                        "EnableMCPTools": True,
                    },
                },
            )
        )
        self.assertEqual(res_council_def["decision"], "allow")
        self.assertEqual(
            res_council_def["overwrite"],
            {
                "EnableSubagentTools": False,
                "EnableMCPTools": False,
            },
        )

        # judge allows pro and inherit untouched, rewrites flash_lite and omitted to flash
        judge_prompt = f"MODE: COUNCIL\nCOUNCIL_ID: c-policy\nCOUNCIL_ROLE: judge\n{VALID_PROMPT}"
        for allowed_model in ("pro", "inherit", "flash"):
            res_ok = orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={
                        "id": f"judge_{allowed_model}",
                        "name": "invoke_subagent",
                        "args": {
                            "TypeName": "judge",
                            "Role": f"Judge-{allowed_model}",
                            "Model": allowed_model,
                            "Prompt": judge_prompt,
                        },
                    },
                )
            )
            self.assertEqual(res_ok["decision"], "allow")
            self.assertNotIn("overwrite", res_ok)

        res_lite = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "judge_lite",
                    "name": "invoke_subagent",
                    "args": {
                        "TypeName": "judge",
                        "Role": "Judge-lite",
                        "Model": "flash_lite",
                        "Prompt": judge_prompt,
                    },
                },
            )
        )
        self.assertEqual(res_lite["decision"], "allow")
        self.assertEqual(res_lite["overwrite"]["Model"], "flash")

        res_omitted = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "judge_omitted",
                    "name": "invoke_subagent",
                    "args": {
                        "TypeName": "judge",
                        "Role": "Judge-default",
                        "Prompt": judge_prompt,
                    },
                },
            )
        )
        self.assertEqual(res_omitted["decision"], "allow")
        self.assertEqual(res_omitted["overwrite"]["Model"], "flash")

    def test_phase3_judge_strict_readonly_report_receipt_and_freshness(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)

        # Mark dirty receipt on root before spawning judge
        st = orchestrator_guard._load_state(self.base_payload)
        receipt = orchestrator_guard._get_test_receipt(st, self.base_payload)
        receipt["dirty"] = True
        receipt["verified"] = False
        receipt["mutation_epoch"] = 1
        orchestrator_guard._save_state(self.base_payload, st)

        judge_cid = "sub-judge-readonly-1"
        orchestrator_guard.register_subagent(
            self.base_payload,
            conversation_id=judge_cid,
            type_name="judge",
            model="pro",
            spawned_mutation_epoch=1,
            mode="council",
            council_id="c-1",
            council_role="judge",
        )
        judge_payload = dict(self.base_payload, conversationId=judge_cid)

        # Strict read-only: blocks write_file and shell write redirection
        res_write = orchestrator_guard.handle_pre_tool(
            dict(
                judge_payload,
                toolCall={
                    "name": "write_file",
                    "args": {"Path": str(Path(self.artifact_dir) / "scratch" / "vote.md"), "Content": "x"},
                },
            )
        )
        self.assertEqual(res_write["decision"], "deny")
        self.assertIn("strictly read-only", res_write["reason"])

        res_shell = orchestrator_guard.handle_pre_tool(
            dict(
                judge_payload,
                toolCall={
                    "name": "run_command",
                    "args": {"CommandLine": "echo vote > vote.txt"},
                },
            )
        )
        self.assertEqual(res_shell["decision"], "deny")
        self.assertIn("strictly read-only", res_shell["reason"])

        # judge is NOT a scout and requires full 4-header report
        self.assertFalse(
            orchestrator_guard._is_scout_subagent(
                judge_payload, orchestrator_guard._load_state(judge_payload)
            )
        )
        res_stop_bad = orchestrator_guard.handle_stop(
            dict(
                judge_payload,
                fullyIdle=True,
                lastAssistantMessage="STATUS: COMPLETED\nNOTES: Voted for Proposal A.",
            )
        )
        self.assertEqual(res_stop_bad["decision"], "continue")
        self.assertIn("4-header REPORT FORMAT", res_stop_bad["reason"])

        # Completing judge via PostToolUse does NOT clear dirty test_receipt
        orchestrator_guard.handle_post_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "inv_judge_ok",
                    "name": "invoke_subagent",
                    "args": {
                        "TypeName": "judge",
                        "Role": "Judge",
                        "Model": "pro",
                        "Prompt": VALID_PROMPT,
                    },
                },
                output="STATUS: COMPLETED\nFILES_CHANGED: NONE\nVERIFICATION: n/a\nNOTES: Winner is Proposal A.",
            )
        )
        st_after_judge = orchestrator_guard._load_state(self.base_payload)
        self.assertTrue(st_after_judge["test_receipt"]["dirty"])
        self.assertFalse(st_after_judge["test_receipt"]["verified"])

        # Mutation epoch bump invalidates stale judge on send_message
        st_after_judge["test_receipt"]["mutation_epoch"] = 2
        orchestrator_guard._save_state(self.base_payload, st_after_judge)
        res_stale = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "name": "send_message",
                    "args": {
                        "Recipient": judge_cid,
                        "Message": "Re-evaluate Proposal A vs Proposal B.",
                    },
                },
            )
        )
        self.assertEqual(res_stale["decision"], "deny")
        self.assertIn("Stale judge", res_stale["reason"])

    def test_phase3_blind_judging_anti_leakage(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)

        leaking_prompts = [
            "Evaluate Proposal A (Claude Opus 5.5) vs Proposal B.",
            "Compare Draft B by Gemini 4 Carbon against Draft A.",
            "Claude Opus authored Proposal C; please score it.",
            "Review Gemini 4 Argon's proposal for the cache layer.",
        ]
        for idx, leak_text in enumerate(leaking_prompts):
            res_leak = orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={
                        "id": f"leak_{idx}",
                        "name": "invoke_subagent",
                        "args": {
                            "TypeName": "judge",
                            "Role": "Blind Judge",
                            "Model": "pro",
                            "Prompt": f"MODE: COUNCIL\nCOUNCIL_ROLE: judge\n{leak_text}",
                        },
                    },
                )
            )
            self.assertEqual(res_leak["decision"], "deny", msg=f"Should deny: {leak_text}")
            self.assertIn("Blind judging violation", res_leak["reason"])

        # Reviewer invoked with COUNCIL_ROLE: judge is also gated
        res_role_judge = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "leak_role_judge",
                    "name": "invoke_subagent",
                    "args": {
                        "TypeName": "reviewer",
                        "Role": "Judge",
                        "Model": "pro",
                        "Prompt": "REVIEW_LENS: architecture\nCOUNCIL_ROLE: judge\nOption 1 - GPT-5.2",
                    },
                },
            )
        )
        self.assertEqual(res_role_judge["decision"], "deny")
        self.assertIn("Blind judging violation", res_role_judge["reason"])

        # Anonymized proposals and benign technical prose are allowed
        benign_prompts = [
            "Proposal A: use carbon theme for the dashboard.\nProposal B: use Gemini API client with retry.",
            "Compare Proposal A, Proposal B, and Proposal C on correctness and maintainability.",
        ]
        for idx, ok_text in enumerate(benign_prompts):
            res_ok = orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={
                        "id": f"benign_{idx}",
                        "name": "invoke_subagent",
                        "args": {
                            "TypeName": "judge",
                            "Role": f"Blind Judge {idx}",
                            "Model": "pro",
                            "Prompt": f"MODE: COUNCIL\nCOUNCIL_ID: blind-ok\nCOUNCIL_ROLE: judge\n{ok_text}",
                        },
                    },
                )
            )
            self.assertEqual(res_ok["decision"], "allow", msg=f"Should allow: {ok_text}")

    def test_phase3_swarm_council_hooks_expansion_kill_and_stop(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)

        # Status / models / title allowed
        for act in ("models", "status", "title"):
            res_info = orchestrator_guard.handle_pre_tool(
                dict(self.base_payload, toolCall={"name": "swarm", "args": {"action": act}})
            )
            self.assertEqual(res_info["decision"], "allow")

        # Non-anonymous council roster name denied
        res_bad_name = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "sw_bad_name",
                    "name": "swarm",
                    "args": {
                        "action": "add",
                        "name": "proposer-claude",
                        "model": "claude-opus-4",
                        "brief": "MODE: COUNCIL\nCOUNCIL_ID: arch-1\nCOUNCIL_ROLE: proposer\nDraft design.",
                    },
                },
            )
        )
        self.assertEqual(res_bad_name["decision"], "deny")
        self.assertIn("Blind council violation", res_bad_name["reason"])

        # Blind judge leak inside swarm brief denied
        res_bad_judge = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "sw_bad_judge",
                    "name": "swarm",
                    "args": {
                        "action": "add",
                        "name": "judge-1",
                        "model": "gemini-4-pro",
                        "brief": "MODE: COUNCIL\nCOUNCIL_ID: arch-1\nCOUNCIL_ROLE: judge\nVote on Draft A by Claude Opus 5.5.",
                    },
                },
            )
        )
        self.assertEqual(res_bad_judge["decision"], "deny")
        self.assertIn("Blind judging violation", res_bad_judge["reason"])

        # Expansion via n=2 (branch) and tasks=[...] (share) + inherit worker
        res_n = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "sw_n",
                    "name": "swarm",
                    "args": {
                        "action": "add",
                        "name": "proposer",
                        "n": 2,
                        "model": "claude-sonnet",
                        "workspace": "branch",
                        "brief": "MODE: COUNCIL\nCOUNCIL_ID: arch-1\nCOUNCIL_ROLE: proposer\nEVAL_CMD: python3 -m unittest",
                    },
                },
            )
        )
        self.assertEqual(res_n["decision"], "allow")

        res_tasks = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "sw_tasks",
                    "name": "swarm",
                    "args": {
                        "action": "add",
                        "name": "reviewer",
                        "tasks": ["task1", "task2"],
                        "model": "gpt-5",
                        "workspace": "share",
                        "brief": "Review candidate patches.",
                    },
                },
            )
        )
        self.assertEqual(res_tasks["decision"], "allow")

        res_inherit = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "sw_inherit",
                    "name": "swarm",
                    "args": {
                        "action": "add",
                        "name": "judge-1",
                        "model": "gemini-4-pro",
                        "workspace": "inherit",
                        "brief": "MODE: COUNCIL\nCOUNCIL_ID: arch-1\nCOUNCIL_ROLE: judge\nScore Proposal A vs Proposal B.",
                    },
                },
            )
        )
        self.assertEqual(res_inherit["decision"], "allow")

        st = orchestrator_guard._load_state(self.base_payload)
        self.assertEqual(
            sorted(st["swarm_workers"].keys()),
            ["judge-1", "proposer-1", "proposer-2", "reviewer-1", "reviewer-2"],
        )
        self.assertTrue(st["swarm_workers"]["proposer-1"]["active_worktree"])
        self.assertTrue(st["swarm_workers"]["reviewer-2"]["active_worktree"])
        self.assertFalse(st["swarm_workers"]["judge-1"]["active_worktree"])
        self.assertEqual(st["active_eval_cmd"], "python3 -m unittest")
        self.assertEqual(st["councils"]["arch-1"]["proposers"], ["proposer-1", "proposer-2"])
        self.assertEqual(st["councils"]["arch-1"]["judges"], ["judge-1"])
        self.assertEqual(st["councils"]["arch-1"]["models"], ["claude-sonnet", "gemini-4-pro"])

        # action="model" updates worker model
        orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "name": "swarm",
                    "args": {"action": "model", "name": "judge-1", "model": "gemini-4-carbon"},
                },
            )
        )
        self.assertEqual(
            orchestrator_guard._load_state(self.base_payload)["swarm_workers"]["judge-1"]["model"],
            "gemini-4-carbon",
        )

        # Stop blocks on uncleaned isolated swarm worktrees
        res_stop_blocked = orchestrator_guard.handle_stop(
            dict(self.base_payload, executionNum=1, fullyIdle=True)
        )
        self.assertEqual(res_stop_blocked["decision"], "continue")
        self.assertIn("Uncleaned isolated swarm worktrees remain", res_stop_blocked["reason"])
        self.assertIn("swarm(action='kill')", res_stop_blocked["reason"])

        # Kill all isolated workers; leave workspace='inherit' judge-1 active
        for wname in ("proposer-1", "proposer-2", "reviewer-1", "reviewer-2"):
            orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={"name": "swarm", "args": {"action": "kill", "name": wname}},
                )
            )
        st_killed = orchestrator_guard._load_state(self.base_payload)
        self.assertEqual(st_killed["swarm_workers"]["proposer-1"]["status"], "killed")
        self.assertFalse(st_killed["swarm_workers"]["proposer-1"]["active_worktree"])
        self.assertEqual(st_killed["swarm_workers"]["judge-1"]["status"], "active")

        # Stop allows when only workspace='inherit' worker remains active
        res_stop_ok = orchestrator_guard.handle_stop(
            dict(self.base_payload, executionNum=2, fullyIdle=True)
        )
        self.assertEqual(res_stop_ok["decision"], "allow")

    def test_phase3_swarm_post_tool_rollback_breaker_and_replan(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)

        # Mark dirty receipt to verify swarm success does not clear it
        st = orchestrator_guard._load_state(self.base_payload)
        receipt = orchestrator_guard._get_test_receipt(st, self.base_payload)
        receipt["dirty"] = True
        receipt["verified"] = False
        orchestrator_guard._save_state(self.base_payload, st)

        # Trivial EVAL_CMD in swarm brief is denied
        res_trivial = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "sw_triv",
                    "name": "swarm",
                    "args": {
                        "action": "add",
                        "name": "proposer-a",
                        "brief": "MODE: COUNCIL\nEVAL_CMD: true",
                    },
                },
            )
        )
        self.assertEqual(res_trivial["decision"], "deny")
        self.assertIn("Trivial or no-op EVAL_CMD rejected", res_trivial["reason"])

        # First swarm add + PostToolUse failure rolls back by call_id and increments failure_streak
        add_args_1 = {
            "action": "add",
            "name": "proposer-a",
            "model": "claude-opus",
            "brief": "MODE: COUNCIL\nCOUNCIL_ID: c-err\nCOUNCIL_ROLE: proposer\nDraft A",
        }
        orchestrator_guard.handle_pre_tool(
            dict(self.base_payload, toolCall={"id": "sw_err_1", "name": "swarm", "args": add_args_1})
        )
        self.assertIn("proposer-a", orchestrator_guard._load_state(self.base_payload)["swarm_workers"])

        orchestrator_guard.handle_post_tool(
            dict(
                self.base_payload,
                toolCall={"id": "sw_err_1", "name": "swarm", "args": add_args_1},
                error="swarm add failed",
            )
        )
        st_err_1 = orchestrator_guard._load_state(self.base_payload)
        self.assertNotIn("proposer-a", st_err_1["swarm_workers"])
        self.assertEqual(st_err_1["councils"]["c-err"]["proposers"], [])
        self.assertEqual(st_err_1["circuit_breaker"]["failure_streak"], 1)
        self.assertFalse(st_err_1["circuit_breaker"]["tripped"])

        # Second swarm add + PostToolUse failure (empty call_id fallback) trips circuit breaker
        add_args_2 = {
            "action": "add",
            "name": "proposer-b",
            "n": 2,
            "model": "gpt-5",
            "brief": "MODE: COUNCIL\nCOUNCIL_ID: c-err\nCOUNCIL_ROLE: proposer\nDraft B",
        }
        orchestrator_guard.handle_pre_tool(
            dict(self.base_payload, toolCall={"name": "swarm", "args": add_args_2})
        )
        orchestrator_guard.handle_post_tool(
            dict(
                self.base_payload,
                toolCall={"name": "swarm", "args": add_args_2},
                exitCode=1,
            )
        )
        st_err_2 = orchestrator_guard._load_state(self.base_payload)
        self.assertNotIn("proposer-b-1", st_err_2["swarm_workers"])
        self.assertNotIn("proposer-b-2", st_err_2["swarm_workers"])
        self.assertEqual(st_err_2["circuit_breaker"]["failure_streak"], 2)
        self.assertTrue(st_err_2["circuit_breaker"]["tripped"])

        # Tripped breaker allows status/kill, blocks add without REPLAN:, allows 1-shot REPLAN:
        self.assertEqual(
            orchestrator_guard.handle_pre_tool(
                dict(self.base_payload, toolCall={"name": "swarm", "args": {"action": "status"}})
            )["decision"],
            "allow",
        )
        self.assertEqual(
            orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={"name": "swarm", "args": {"action": "kill", "name": "any"}},
                )
            )["decision"],
            "allow",
        )
        res_blocked = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "sw_blocked",
                    "name": "swarm",
                    "args": {
                        "action": "add",
                        "name": "proposer-c",
                        "brief": "Retry without replan",
                    },
                },
            )
        )
        self.assertEqual(res_blocked["decision"], "deny")
        self.assertIn("REPLAN:", res_blocked["reason"])

        res_replan = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "sw_replan",
                    "name": "swarm",
                    "args": {
                        "action": "add",
                        "name": "proposer-c",
                        "model": "gemini-4-pro",
                        "brief": "REPLAN: switch provider\nMODE: COUNCIL\nCOUNCIL_ID: c-err\nCOUNCIL_ROLE: proposer",
                    },
                },
            )
        )
        self.assertEqual(res_replan["decision"], "allow")
        st_replanned = orchestrator_guard._load_state(self.base_payload)
        self.assertFalse(st_replanned["circuit_breaker"]["tripped"])
        self.assertEqual(st_replanned["circuit_breaker"]["replan_count"], 1)

        # Success resets failure_streak while preserving dirty test_receipt
        orchestrator_guard.handle_post_tool(
            dict(
                self.base_payload,
                toolCall={"id": "sw_replan", "name": "swarm", "args": {"action": "add", "name": "proposer-c"}},
                output="Worker proposer-c ready",
            )
        )
        st_ok = orchestrator_guard._load_state(self.base_payload)
        self.assertEqual(st_ok["circuit_breaker"]["failure_streak"], 0)
        self.assertFalse(st_ok["circuit_breaker"]["tripped"])
        self.assertTrue(st_ok["test_receipt"]["dirty"])

    def test_phase3_non_swarm_council_proposer_and_cli_status(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)

        # MODE: COUNCIL research proposer allows pro/inherit/flash without flash_lite downgrade
        res_research_pro = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "inv_res_pro",
                    "name": "invoke_subagent",
                    "args": {
                        "TypeName": "research",
                        "Role": "Proposer A",
                        "Model": "pro",
                        "Prompt": "MODE: COUNCIL\nCOUNCIL_ID: fallback-1\nCOUNCIL_ROLE: proposer\n1. Compare options.",
                    },
                },
            )
        )
        self.assertEqual(res_research_pro["decision"], "allow")
        self.assertNotIn("overwrite", res_research_pro)

        res_research_flash = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "inv_res_flash",
                    "name": "invoke_subagent",
                    "args": {
                        "TypeName": "research",
                        "Role": "Proposer B",
                        "Model": "flash",
                        "Prompt": "MODE: COUNCIL\nCOUNCIL_ID: fallback-1\nCOUNCIL_ROLE: proposer\n1. Summarize design.",
                    },
                },
            )
        )
        self.assertEqual(res_research_flash["decision"], "allow")
        self.assertNotIn("overwrite", res_research_flash)

        # council type allows inherit untouched even without explicit MODE header
        res_council_inherit = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "inv_c_inh",
                    "name": "invoke_subagent",
                    "args": {
                        "TypeName": "council",
                        "Role": "Proposer C",
                        "Model": "inherit",
                        "Prompt": "1. Propose architecture.",
                    },
                },
            )
        )
        self.assertEqual(res_council_inherit["decision"], "allow")
        self.assertNotIn("overwrite", res_council_inherit)

        # Standard non-council research still rejects Model='pro'
        res_std_research = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "inv_std_res",
                    "name": "invoke_subagent",
                    "args": {
                        "TypeName": "research",
                        "Role": "Standard Research",
                        "Model": "pro",
                        "Prompt": "1. Investigate module.",
                    },
                },
            )
        )
        self.assertEqual(res_std_research["decision"], "deny")
        self.assertIn("Model='pro'", res_std_research["reason"])

        # Council proposer is scratch-only: scratch .md allowed, workspace code denied, receipt kept dirty
        st = orchestrator_guard._load_state(self.base_payload)
        receipt = orchestrator_guard._get_test_receipt(st, self.base_payload)
        receipt["dirty"] = True
        receipt["verified"] = False
        orchestrator_guard._save_state(self.base_payload, st)

        council_cid = "sub-council-proposer-1"
        orchestrator_guard.register_subagent(
            self.base_payload,
            conversation_id=council_cid,
            type_name="council",
            model="pro",
            mode="council",
            council_id="fallback-1",
            council_role="proposer",
        )
        council_payload = dict(self.base_payload, conversationId=council_cid)

        res_scratch = orchestrator_guard.handle_pre_tool(
            dict(
                council_payload,
                toolCall={
                    "name": "write_file",
                    "args": {
                        "Path": str(Path(self.artifact_dir) / "scratch" / "proposal_a.md"),
                        "Content": "# Proposal A\n",
                    },
                },
            )
        )
        self.assertEqual(res_scratch["decision"], "allow")

        res_code_mut = orchestrator_guard.handle_pre_tool(
            dict(
                council_payload,
                toolCall={
                    "name": "write_file",
                    "args": {
                        "Path": str(orchestrator_guard.PROJECT_ROOT / "src" / "hack.py"),
                        "Content": "x = 1\n",
                    },
                },
            )
        )
        self.assertEqual(res_code_mut["decision"], "deny")
        self.assertIn("may only write markdown reports", res_code_mut["reason"])

        orchestrator_guard.handle_post_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "inv_c_done",
                    "name": "invoke_subagent",
                    "args": {
                        "TypeName": "council",
                        "Role": "Proposer C",
                        "Model": "inherit",
                        "Prompt": "1. Propose architecture.",
                    },
                },
                output="STATUS: COMPLETED\nFILES_CHANGED: scratch/proposal_a.md\nVERIFICATION: n/a\nNOTES: Drafted Proposal A.",
            )
        )
        st_after = orchestrator_guard._load_state(self.base_payload)
        self.assertTrue(st_after["test_receipt"]["dirty"])
        self.assertEqual(st_after["councils"]["fallback-1"]["proposers"], ["Proposer A", "Proposer B"])

        # _cli_status surfaces councils and swarmWorkers
        status_out = orchestrator_guard._cli_status(self.base_payload)
        self.assertIn("councils", status_out)
        self.assertIn("swarmWorkers", status_out)
        self.assertIn("fallback-1", status_out["councils"])

        # _prune_state bounds councils and prunes oldest killed swarm_workers
        prune_state = orchestrator_guard._default_state()
        for i in range(orchestrator_guard.MAX_STATE_INVOCATION_HISTORY + 5):
            prune_state["councils"][f"c-{i}"] = {
                "proposers": [f"p-{j}" for j in range(orchestrator_guard.MAX_STATE_LIST_ENTRIES + 5)],
                "judges": [],
                "models": [],
                "created_at": float(i),
            }
        for i in range(orchestrator_guard.MAX_STATE_LIST_ENTRIES + 5):
            prune_state["swarm_workers"][f"w-{i}"] = {
                "name": f"w-{i}",
                "status": "killed" if i < 10 else "active",
                "spawned_at": float(i),
            }
        orchestrator_guard._prune_state(prune_state)
        self.assertEqual(len(prune_state["councils"]), orchestrator_guard.MAX_STATE_INVOCATION_HISTORY)
        self.assertNotIn("c-0", prune_state["councils"])
        first_c = next(iter(prune_state["councils"].values()))
        self.assertEqual(len(first_c["proposers"]), orchestrator_guard.MAX_STATE_LIST_ENTRIES)
        self.assertEqual(len(prune_state["swarm_workers"]), orchestrator_guard.MAX_STATE_LIST_ENTRIES)
        self.assertNotIn("w-0", prune_state["swarm_workers"])
        self.assertIn("w-5", prune_state["swarm_workers"])

    def test_phase3_blind_judge_metadata_filename_and_scratch_file_leaks(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)

        def _invoke_judge(prompt: str, call_id: str) -> dict:
            return orchestrator_guard.handle_pre_tool(
                dict(
                    self.base_payload,
                    toolCall={
                        "id": call_id,
                        "name": "invoke_subagent",
                        "args": {
                            "TypeName": "judge",
                            "Role": "Blind Judge",
                            "Model": "inherit",
                            "Prompt": prompt,
                        },
                    },
                )
            )

        # Standalone metadata lines
        res_meta1 = _invoke_judge(
            "MODE: COUNCIL\nCOUNCIL_ID: c-meta-1\nCOUNCIL_ROLE: judge\nModel: Claude Opus 5.5\nCompare proposals.",
            "j_meta1",
        )
        self.assertEqual(res_meta1["decision"], "deny")
        self.assertIn("Blind judging violation", res_meta1["reason"])

        res_meta2 = _invoke_judge(
            "MODE: COUNCIL\nCOUNCIL_ID: c-meta-2\nCOUNCIL_ROLE: judge\n- Author: Gemini 4 Carbon\nCompare proposals.",
            "j_meta2",
        )
        self.assertEqual(res_meta2["decision"], "deny")
        self.assertIn("Blind judging violation", res_meta2["reason"])

        # Wider same-line attribution (>30 chars between proposal id and model)
        res_wide = _invoke_judge(
            "MODE: COUNCIL\nCOUNCIL_ID: c-wide\nCOUNCIL_ROLE: judge\n"
            "Proposal A (written carefully over two iterations by Claude Opus 5.5)",
            "j_wide",
        )
        self.assertEqual(res_wide["decision"], "deny")
        self.assertIn("Blind judging violation", res_wide["reason"])

        # Model-named proposal filename
        res_fname = _invoke_judge(
            "MODE: COUNCIL\nCOUNCIL_ID: c-fname\nCOUNCIL_ROLE: judge\n"
            "Review scratch/proposal_claude.md and scratch/proposal_b.md.",
            "j_fname",
        )
        self.assertEqual(res_fname["decision"], "deny")
        self.assertIn("Blind judging violation", res_fname["reason"])

        # Referenced scratch file containing metadata leak
        scratch_dir = Path(self.artifact_dir) / "scratch"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        prop_a = scratch_dir / "proposal_a.md"
        prop_a.write_text("# Proposal A\nAuthor: Gemini 4 Carbon\nUse modular handlers.\n", encoding="utf-8")

        res_scratch_leak = _invoke_judge(
            f"MODE: COUNCIL\nCOUNCIL_ID: c-scratch-leak\nCOUNCIL_ROLE: judge\nReview {prop_a}.",
            "j_scratch_leak",
        )
        self.assertEqual(res_scratch_leak["decision"], "deny")
        self.assertIn("proposal_a.md", res_scratch_leak["reason"])

        # Referenced scratch file with benign technical prose ("carbon accounting with the Gemini API client")
        prop_b = scratch_dir / "proposal_b.md"
        prop_b.write_text(
            "# Proposal B\nImplement carbon accounting with the Gemini API client.\n",
            encoding="utf-8",
        )
        res_clean = _invoke_judge(
            "MODE: COUNCIL\nCOUNCIL_ID: c-clean\nCOUNCIL_ROLE: judge\nReview scratch/proposal_b.md.",
            "j_clean",
        )
        self.assertEqual(res_clean["decision"], "allow")

    def test_phase3_swarm_sharded_kill_base_name_and_kill_error_rollback(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)

        # Spawn sharded workers with isolated worktrees
        res_add = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "sw_sharded_add",
                    "name": "swarm",
                    "args": {
                        "action": "add",
                        "name": "proposer",
                        "n": 2,
                        "workspace": "share",
                        "brief": "MODE: COUNCIL\nCOUNCIL_ID: c-shard\nCOUNCIL_ROLE: proposer\nDraft design",
                    },
                },
            )
        )
        self.assertEqual(res_add["decision"], "allow")
        st_added = orchestrator_guard._load_state(self.base_payload)
        self.assertEqual(st_added["swarm_workers"]["proposer-1"]["base_name"], "proposer")
        self.assertTrue(st_added["swarm_workers"]["proposer-1"]["sharded"])

        # Killing by base name instead of individual sharded worker names is denied
        res_kill_base = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "sw_kill_base",
                    "name": "swarm",
                    "args": {"action": "kill", "name": "proposer"},
                },
            )
        )
        self.assertEqual(res_kill_base["decision"], "deny")
        self.assertIn("proposer-1", res_kill_base["reason"])
        self.assertIn("proposer-2", res_kill_base["reason"])

        # Killing proposer-1 allowed at PreToolUse -> temporarily marked killed
        res_kill_1_pre = orchestrator_guard.handle_pre_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "sw_kill_1_err",
                    "name": "swarm",
                    "args": {"action": "kill", "name": "proposer-1"},
                },
            )
        )
        self.assertEqual(res_kill_1_pre["decision"], "allow")
        self.assertEqual(
            orchestrator_guard._load_state(self.base_payload)["swarm_workers"]["proposer-1"]["status"],
            "killed",
        )

        # PostToolUse error rolls back proposer-1 to active + active_worktree=True
        orchestrator_guard.handle_post_tool(
            dict(
                self.base_payload,
                toolCall={
                    "id": "sw_kill_1_err",
                    "name": "swarm",
                    "args": {"action": "kill", "name": "proposer-1"},
                },
                error="Worker not found or kill failed",
            )
        )
        st_rolled = orchestrator_guard._load_state(self.base_payload)
        self.assertEqual(st_rolled["swarm_workers"]["proposer-1"]["status"], "active")
        self.assertTrue(st_rolled["swarm_workers"]["proposer-1"]["active_worktree"])

        # Stop still blocks because active isolated worktrees remain
        res_stop_blocked = orchestrator_guard.handle_stop(
            dict(self.base_payload, executionNum=1, fullyIdle=True)
        )
        self.assertEqual(res_stop_blocked["decision"], "continue")

        # Cleanly kill both sharded workers (PreToolUse + PostToolUse without error)
        for wname in ("proposer-1", "proposer-2"):
            cid = f"sw_kill_ok_{wname}"
            args_kill = {"action": "kill", "name": wname}
            res_pre = orchestrator_guard.handle_pre_tool(
                dict(self.base_payload, toolCall={"id": cid, "name": "swarm", "args": args_kill})
            )
            self.assertEqual(res_pre["decision"], "allow")
            orchestrator_guard.handle_post_tool(
                dict(
                    self.base_payload,
                    toolCall={"id": cid, "name": "swarm", "args": args_kill},
                    output=f"Killed {wname}",
                )
            )

        res_stop_ok = orchestrator_guard.handle_stop(
            dict(self.base_payload, executionNum=2, fullyIdle=True)
        )
        self.assertEqual(res_stop_ok["decision"], "allow")


if __name__ == "__main__":
    unittest.main()
