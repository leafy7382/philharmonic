#!/usr/bin/env python3
"""Unit tests for orchestrator_guard.py and ste_lint.py."""

from pathlib import Path
import sys
import tempfile
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

    def test_pre_invocation_pro_vs_non_pro_and_dedup(self) -> None:
        # Pro model produces no warning
        res_pro = orchestrator_guard.handle_pre_invocation(self.base_payload)
        self.assertEqual(res_pro, {})

        # Non-Pro model injects ephemeral warning once per invocationNum
        bad_payload = dict(self.base_payload, modelName="gemini-flash", invocationNum=2)
        res_warn = orchestrator_guard.handle_pre_invocation(bad_payload)
        self.assertIn("injectSteps", res_warn)
        self.assertIn("ORCHESTRATOR GUARD", res_warn["injectSteps"][0]["ephemeralMessage"])

        # Duplicate call on same invocationNum is deduplicated
        res_dup = orchestrator_guard.handle_pre_invocation(bad_payload)
        self.assertEqual(res_dup, {})

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


if __name__ == "__main__":
    unittest.main()
