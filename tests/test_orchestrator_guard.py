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

    def _payload(
        self,
        cid: Optional[str] = None,
        tool: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
        call_id: Optional[str] = None,
        base: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        p = dict(self.base_payload if base is None else base)
        if cid is not None:
            p["conversationId"] = cid
        if tool is not None:
            tc: Dict[str, Any] = {"name": tool, "args": args or {}}
            if call_id is not None:
                tc["id"] = call_id
            p["toolCall"] = tc
        p.update(extra)
        return p

    def _pre(
        self,
        tool: str,
        args: Optional[Dict[str, Any]] = None,
        cid: Optional[str] = None,
        call_id: Optional[str] = None,
        base: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        return orchestrator_guard.handle_pre_tool(
            self._payload(cid=cid, tool=tool, args=args, call_id=call_id, base=base, **extra)
        )

    def _post(
        self,
        tool: str,
        args: Optional[Dict[str, Any]] = None,
        output: str = "",
        error: str = "",
        exit_code: Optional[int] = None,
        cid: Optional[str] = None,
        call_id: Optional[str] = None,
        base: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        kw: Dict[str, Any] = {"output": output, "error": error, **extra}
        if exit_code is not None:
            kw["exitCode"] = exit_code
        return orchestrator_guard.handle_post_tool(
            self._payload(cid=cid, tool=tool, args=args, call_id=call_id, base=base, **kw)
        )

    def _stop(
        self,
        text: str = "",
        cid: Optional[str] = None,
        base: Optional[Dict[str, Any]] = None,
        fully_idle: bool = True,
        **extra: Any,
    ) -> Dict[str, Any]:
        kw: Dict[str, Any] = {"fullyIdle": fully_idle, "terminationReason": "NO_TOOL_CALL", **extra}
        if text:
            kw["lastAssistantMessage"] = text
        return orchestrator_guard.handle_stop(self._payload(cid=cid, base=base, **kw))

    def _pre_inv(self, cid: Optional[str] = None, base: Optional[Dict[str, Any]] = None, **extra: Any) -> Dict[str, Any]:
        return orchestrator_guard.handle_pre_invocation(self._payload(cid=cid, base=base, **extra))

    def _brain_payload(self, cid: str, root: str = "brain", **extra: Any) -> Dict[str, Any]:
        d = Path(self.artifact_dir) / root / cid
        d.mkdir(parents=True, exist_ok=True)
        return dict(self.base_payload, conversationId=cid, artifactDirectoryPath=str(d), **extra)

    def _load_st(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return orchestrator_guard._load_state(self.base_payload if payload is None else payload)

    def _seed_state(self, payload: Optional[Dict[str, Any]] = None, **updates: Any) -> Dict[str, Any]:
        p = self.base_payload if payload is None else payload
        st = orchestrator_guard._load_state(p)
        st.update(updates)
        orchestrator_guard._save_state(p, st)
        return st

    def _reg_sub(
        self,
        sub_type: str,
        cid: str,
        model: str = "flash",
        payload: Optional[Dict[str, Any]] = None,
        **kw: Any,
    ) -> None:
        orchestrator_guard.register_subagent(
            self.base_payload if payload is None else payload,
            conversation_id=cid,
            type_name=sub_type,
            model=model,
            **kw,
        )

    def _sub_args(
        self,
        sub_type: str = "implementer",
        role: str = "Worker",
        model: Optional[str] = None,
        prompt: str = VALID_PROMPT,
        **extra: Any,
    ) -> Dict[str, Any]:
        d: Dict[str, Any] = {"TypeName": sub_type, "Role": role, "Prompt": prompt, **extra}
        if model is not None:
            d["Model"] = model
        return d

    def _cmd_args(self, cmd: str, cwd: Optional[str] = None) -> Dict[str, Any]:
        return {"CommandLine": cmd, "Cwd": cwd or str(orchestrator_guard.PROJECT_ROOT)}

    def test_autonomous_git_diff_audit_in_handle_stop(self) -> None:
        tampered_diff = (
            "diff --git a/tests/test_example.py b/tests/test_example.py\n"
            "--- a/tests/test_example.py\n+++ b/tests/test_example.py\n"
            "@@ -10,1 +10,2 @@\n+@unittest.skip('temporarily skipping')\n def test_feature():\n"
        )
        res = self._stop(gitDiff=tampered_diff)
        self.assertEqual(res["decision"], "continue")
        self.assertIn("Diff audit detected test skip", res.get("reason", ""))

        clean_diff = (
            "diff --git a/tests/test_example.py b/tests/test_example.py\n"
            "--- a/tests/test_example.py\n+++ b/tests/test_example.py\n"
            "@@ -10,0 +10,3 @@\n+def test_new_behavior():\n+    assert 1 + 1 == 2\n"
        )
        res_clean = self._stop(executionNum=2, gitDiff=clean_diff)
        self.assertEqual(res_clean["decision"], "allow")
        self.assertNotIn("reason", res_clean)

    def test_isolation_reminder_deduplication(self) -> None:
        sub_cid = "sub-dedup-worker-101"
        self._reg_sub("implementer", sub_cid, "flash")
        res_1 = self._pre_inv(cid=sub_cid, invocationNum=1)
        self.assertIn("injectSteps", res_1)
        self.assertIn("Subagent context isolation active", res_1["injectSteps"][0]["ephemeralMessage"])
        self.assertEqual(self._pre_inv(cid=sub_cid, invocationNum=1), {})
        res_3 = self._pre_inv(cid=sub_cid, invocationNum=2)
        self.assertIn("injectSteps", res_3)
        self.assertIn("Subagent context isolation active", res_3["injectSteps"][0]["ephemeralMessage"])

    def test_pre_invocation_pro_vs_non_pro_and_dedup(self) -> None:
        # Root on a Pro model returns {}
        self.assertEqual(orchestrator_guard.handle_pre_invocation(self.base_payload), {})

        # Root on a small model returns {} and stores root_conversation_id in state
        self.assertEqual(self._pre_inv(modelName="gemini-flash", invocationNum=2), {})
        state = self._load_st()
        self.assertEqual(state.get("root_conversation_id"), self.base_payload["conversationId"].lower())

        # A registered subagent gets injectSteps with isolation reminder once, deduped on second call
        sub_cid = "sub-dedup-uuid-9999"
        self._reg_sub("scout", sub_cid, "flash_lite")
        res_sub1 = self._pre_inv(cid=sub_cid, modelName="gemini-flash-lite", invocationNum=1)
        self.assertIn("injectSteps", res_sub1)
        self.assertIn("Subagent context isolation active", res_sub1["injectSteps"][0]["ephemeralMessage"])
        self.assertEqual(self._pre_inv(cid=sub_cid, modelName="gemini-flash-lite", invocationNum=1), {})

    def test_define_subagent_forces_subagent_tools_false(self) -> None:
        res = self._pre("define_subagent", {"Name": "scout", "EnableSubagentTools": True})
        self.assertEqual(res["decision"], "allow")
        self.assertEqual(res["overwrite"], {"EnableSubagentTools": False})

        res_research = self._pre("define_subagent", {"Name": "research", "EnableWriteTools": True, "EnableSubagentTools": True})
        self.assertEqual(res_research["decision"], "allow")
        self.assertEqual(res_research["overwrite"], {"EnableSubagentTools": False})

    def test_invoke_subagent_blocks_pro_model(self) -> None:
        res = self._pre("invoke_subagent", self._sub_args("implementer", "Auth Implementer", "pro"))
        self.assertEqual(res["decision"], "deny")
        self.assertIn("Model='pro'", res["reason"])

    def test_invoke_subagent_rewrites_inherit_to_tier_defaults(self) -> None:
        res_scout = self._pre("invoke_subagent", self._sub_args("scout", "Symbol Scout", "inherit"))
        self.assertEqual(res_scout["decision"], "allow")
        self.assertEqual(res_scout["overwrite"]["Model"], "flash_lite")

        res_impl = self._pre("invoke_subagent", self._sub_args("implementer", "Code Worker"))
        self.assertEqual(res_impl["decision"], "allow")
        self.assertEqual(res_impl["overwrite"]["Model"], "flash")

    def test_invoke_subagent_and_send_message_reject_empty_prompts(self) -> None:
        res_empty_prompt = self._pre("invoke_subagent", self._sub_args("scout", "Empty Scout", "flash_lite", prompt="   "))
        self.assertEqual(res_empty_prompt["decision"], "deny")
        self.assertIn("non-empty 'Prompt'", res_empty_prompt["reason"])

        res_empty_type = self._pre("invoke_subagent", self._sub_args("", "No Type", "flash_lite"))
        self.assertEqual(res_empty_type["decision"], "deny")
        self.assertIn("non-empty 'TypeName'", res_empty_type["reason"])

        res_empty_msg = self._pre("send_message", {"Recipient": "worker-1", "Message": "   "})
        self.assertEqual(res_empty_msg["decision"], "deny")
        self.assertIn("Message cannot be empty", res_empty_msg["reason"])

    def test_send_message_tier_bypass_blocked_for_flash_lite(self) -> None:
        scout_cid = "11111111-2222-3333-4444-555555555555"
        self._reg_sub("scout", scout_cid, "flash_lite", workspace="inherit", role="File Scout")

        res_deny = self._pre("send_message", {"Recipient": scout_cid, "Message": "Edit `src/auth/parser.py` to add token validation."})
        self.assertEqual(res_deny["decision"], "deny")
        self.assertIn("flash_lite", res_deny["reason"])

        res_allow = self._pre("send_message", {"Recipient": scout_cid, "Message": "Read line 40 of `src/auth/parser.py`."})
        self.assertEqual(res_allow["decision"], "allow")

    def test_stop_hook_blocks_uncleaned_branch_until_killed(self) -> None:
        branch_cid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        self._reg_sub("implementer", branch_cid, "flash", workspace="branch", role="Branch Worker")

        # When fullyIdle is False, Stop is allowed so reactive wakeup works
        self.assertEqual(self._stop(fully_idle=False)["decision"], "allow")

        # When fullyIdle is True and branch is still active, Stop returns continue
        res_stop = self._stop()
        self.assertEqual(res_stop["decision"], "continue")
        self.assertIn(branch_cid, res_stop["reason"])

        # Killing the branch clears the gate
        self._pre("manage_subagents", {"Action": "kill", "ConversationIds": [branch_cid]})
        self.assertEqual(self._stop(executionNum=2)["decision"], "allow")

    def test_workspace_auto_allow_vs_external_paths(self) -> None:
        res_in = self._pre("run_command", self._cmd_args("python3 -m unittest discover -s tests"))
        self.assertEqual(res_in["decision"], "allow")
        self.assertIn("permissionOverrides", res_in)
        self.assertTrue(any(o.startswith("file(") for o in res_in["permissionOverrides"]))

        self.assertEqual(self._pre("run_command", self._cmd_args("cat /etc/shadow"))["decision"], "ask")
        self.assertEqual(self._pre("run_command", self._cmd_args("cat ../outside.txt"))["decision"], "ask")

    def test_post_tool_rolls_back_failed_branch_spawn(self) -> None:
        args = self._sub_args("implementer", "Failed Branch Worker", "flash", Workspace="branch")
        self.assertEqual(self._pre("invoke_subagent", args)["decision"], "allow")
        self._post("invoke_subagent", args, error="failed to create git worktree")
        self.assertEqual(self._stop()["decision"], "allow")

    def test_transcript_sync_ignores_non_subagent_uuids(self) -> None:
        transcript_file = Path(self.artifact_dir) / "transcript.jsonl"
        args = self._sub_args("scout", "Symbol Scout", "flash_lite")
        self._pre("invoke_subagent", args, transcriptPath=str(transcript_file))

        non_subagent_step_uuid = "11111111-1111-1111-1111-111111111111"
        non_subagent_task_uuid = "22222222-2222-2222-2222-222222222222"
        actual_subagent_cid = "33333333-3333-3333-3333-333333333333"
        transcript_file.write_text(
            json.dumps({"stepId": non_subagent_step_uuid, "message": "Invoked tool invoke_subagent"})
            + "\n"
            + json.dumps({
                "stepId": "44444444-4444-4444-4444-444444444444",
                "toolCallId": non_subagent_task_uuid,
                "name": "invoke_subagent",
                "result": {"conversationId": actual_subagent_cid},
            })
            + "\n",
            encoding="utf-8",
        )
        self._post("invoke_subagent", args, transcriptPath=str(transcript_file))

        state = self._load_st()
        subagents = state.get("subagents", {})
        self.assertIn(actual_subagent_cid, subagents)
        self.assertNotIn(non_subagent_step_uuid, subagents)
        self.assertNotIn(non_subagent_task_uuid, subagents)
        self.assertNotIn("44444444-4444-4444-4444-444444444444", subagents)
        self.assertEqual(len(state.get("pending_spawns", [])), 0)
        self.assertEqual(subagents[actual_subagent_cid]["model"], "flash_lite")

    def test_auto_downgrade_simple_readonly_research_to_flash_lite(self) -> None:
        simple_read_prompt = (
            "TASK: Read `src/auth/parser.py` to check token constants.\n\n"
            "RULES:\n1. Open `src/auth/parser.py`.\n2. Read the constant definitions at the top of the file.\n"
            "3. Do not change any file.\n\n"
            "REPORT FORMAT:\nReturn only these four lines:\nSTATUS: COMPLETED | BLOCKED | FAILED\n"
            "FILES_CHANGED: NONE\nVERIFICATION: Read `src/auth/parser.py`.\nNOTES: <one sentence stating the constant values>\n"
        )
        multi_step_read_prompt = (
            "TASK: Audit authentication flow across four modules.\n\n"
            "RULES:\n1. Open `src/auth/parser.py`.\n2. Open `src/auth/session.py`.\n"
            "3. Open `src/auth/middleware.py`.\n4. Compare token expiration checks across all three modules.\n\n"
            "REPORT FORMAT:\nReturn only these four lines:\nSTATUS: COMPLETED | BLOCKED | FAILED\n"
            "FILES_CHANGED: NONE\nVERIFICATION: Inspected all three authentication modules.\nNOTES: <one sentence summarizing findings>\n"
        )
        res_simple = self._pre(
            "invoke_subagent",
            self._sub_args("research", "Quick Reader", "flash", simple_read_prompt),
            call_id="call_downgrade_1",
        )
        self.assertEqual(res_simple["decision"], "allow")
        self.assertEqual(res_simple["overwrite"]["Model"], "flash_lite")

        res_deep = self._pre(
            "invoke_subagent",
            self._sub_args("research", "Deep Researcher", "flash", multi_step_read_prompt),
            call_id="call_downgrade_2",
        )
        self.assertEqual(res_deep["decision"], "allow")
        self.assertIsNone(res_deep.get("overwrite"))

    def test_pending_spawns_deduplicated_across_duplicate_pre_tool_hooks(self) -> None:
        args = self._sub_args("scout", "Dedup Scout", "flash_lite")
        for _ in range(3):
            self._pre("invoke_subagent", args, call_id="call_dedup_1")
        self.assertEqual(len(self._load_st().get("pending_spawns", [])), 1)

    def test_worktree_and_sibling_brain_paths_auto_allowed(self) -> None:
        app_data = Path(self.artifact_dir) / "appdata"
        subagent_brain = app_data / "brain" / "subagent-uuid"
        parent_scratch = app_data / "brain" / "parent-uuid" / "scratch" / "task.log"
        worktree_file = app_data / "worktrees" / "parent-uuid" / "subagent-worktree" / "README.md"
        for p in (subagent_brain, parent_scratch.parent, worktree_file.parent):
            p.mkdir(parents=True, exist_ok=True)

        self.assertEqual(
            self._pre("write_file", {"Path": str(parent_scratch)}, artifactDirectoryPath=str(subagent_brain))["decision"],
            "allow",
        )
        self.assertEqual(
            self._pre("write_file", {"Path": str(worktree_file)}, artifactDirectoryPath=str(subagent_brain))["decision"],
            "allow",
        )

    def test_direct_mutation_budget_blocks_third_workspace_file(self) -> None:
        file1, file2, file3 = (str(orchestrator_guard.PROJECT_ROOT / f"file{i}.py") for i in (1, 2, 3))
        self.assertEqual(self._pre("write_file", {"Path": file1})["decision"], "allow")
        self.assertEqual(self._pre("edit_file", {"Path": file1})["decision"], "allow")
        self.assertEqual(self._pre("write_file", {"Path": file2})["decision"], "allow")
        self.assertEqual(self._pre("edit_file", {"Path": file2})["decision"], "allow")

        res3 = self._pre("write_file", {"Path": file3})
        self.assertEqual(res3["decision"], "deny")
        self.assertIn("[ORCHESTRATOR GUARD]", res3["reason"])

        subagent_cid = "sub-mutation-cid-1111"
        self._reg_sub("implementer", subagent_cid, "flash")
        self.assertEqual(
            self._pre("write_file", {"Path": file3}, cid=subagent_cid, modelName="gemini-3-flash")["decision"],
            "allow",
        )

    def test_direct_edit_then_test_command_blocked_for_orchestrator(self) -> None:
        file1 = str(orchestrator_guard.PROJECT_ROOT / "file1.py")
        self.assertEqual(self._pre("write_file", {"Path": file1})["decision"], "allow")
        self.assertEqual(self._pre("run_command", self._cmd_args("git status"))["decision"], "allow")
        self.assertEqual(
            self._pre("run_command", self._cmd_args("python3 -m unittest discover -s tests"), call_id="test_1")["decision"],
            "allow",
        )

        res_test_2 = self._pre("run_command", self._cmd_args("python3 -m unittest discover -s tests"), call_id="test_2")
        self.assertEqual(res_test_2["decision"], "deny")
        self.assertIn("[ORCHESTRATOR GUARD]", res_test_2["reason"])

        res_subsequent_write = self._pre("write_file", {"Path": file1})
        self.assertEqual(res_subsequent_write["decision"], "deny")
        self.assertIn("[ORCHESTRATOR GUARD]", res_subsequent_write["reason"])

    def test_direct_command_budget_blocks_third_command_and_deduplicates(self) -> None:
        self.assertEqual(self._pre("run_command", self._cmd_args("ls -la"), call_id="cmd_call_1")["decision"], "allow")
        self.assertEqual(self._pre("run_command", self._cmd_args("ls -la"), call_id="cmd_call_1")["decision"], "allow")
        self.assertEqual(len(self._load_st()["direct_budget"]["1"]["commands"]), 1)

        self.assertEqual(self._pre("run_command", self._cmd_args("python3 --version"), call_id="cmd_call_2")["decision"], "allow")
        res_cmd3 = self._pre("run_command", self._cmd_args("echo hello"), call_id="cmd_call_3")
        self.assertEqual(res_cmd3["decision"], "deny")
        self.assertIn("[ORCHESTRATOR GUARD]", res_cmd3["reason"])

        self.assertEqual(
            self._pre("run_command", self._cmd_args("echo hello"), call_id="cmd_call_4", invocationNum=2)["decision"],
            "allow",
        )

    def test_empirical_test_receipt_required_before_stop(self) -> None:
        code_file = str(orchestrator_guard.PROJECT_ROOT / "src" / "app.py")
        self.assertEqual(self._pre("write_file", {"Path": code_file, "Content": "print('hello')"})["decision"], "allow")

        res_stop_1 = self._stop(executionNum=1)
        self.assertEqual(res_stop_1["decision"], "continue")
        self.assertIn("Empirical test receipt missing", res_stop_1["reason"])

        self._post("run_command", self._cmd_args("pytest tests/"), error="exit status 1", call_id="test_cmd_1")
        self.assertEqual(self._stop(executionNum=2)["decision"], "continue")

        self._post("run_command", self._cmd_args("pytest tests/"), error="", call_id="test_cmd_2")
        self.assertEqual(self._stop(executionNum=3)["decision"], "allow")

    def test_two_turn_circuit_breaker_halts_retries_and_allows_scratch_log(self) -> None:
        self._post("run_command", self._cmd_args("pytest tests/"), error="exit status 1", call_id="cmd_fail_1")
        self._post("run_command", self._cmd_args("pytest tests/"), error="exit status 1", call_id="cmd_fail_2")

        res_cmd_3 = self._pre("run_command", self._cmd_args("pytest tests/"), call_id="cmd_fail_3")
        self.assertEqual(res_cmd_3["decision"], "deny")
        self.assertIn("2-turn circuit breaker tripped", res_cmd_3["reason"])

        code_path = str(orchestrator_guard.PROJECT_ROOT / "src" / "app.py")
        res_edit = self._pre("write_file", {"Path": code_path, "Content": "print('retry')"})
        self.assertEqual(res_edit["decision"], "deny")
        self.assertIn("2-turn circuit breaker tripped", res_edit["reason"])

        scratch_log = str(Path(self.artifact_dir) / "scratch" / "fail.log")
        self.assertEqual(
            self._pre("write_file", {"Path": scratch_log, "Content": "Traceback: test failed twice"})["decision"],
            "allow",
        )
        self.assertEqual(self._stop()["decision"], "allow")

    def test_pre_tool_blocks_test_tampering_edits(self) -> None:
        test_file = str(orchestrator_guard.PROJECT_ROOT / "tests" / "test_sample.py")

        res_del_assert = self._pre("edit_file", {"Path": test_file, "Target": "        self.assertEqual(a, b)\n", "Replacement": ""})
        self.assertEqual(res_del_assert["decision"], "deny")
        self.assertIn("reduces assertion count", res_del_assert["reason"])

        res_del_test = self._pre("edit_file", {"Path": test_file, "Target": "    def test_login(self):\n        pass\n", "Replacement": ""})
        self.assertEqual(res_del_test["decision"], "deny")
        self.assertIn("reduces test count", res_del_test["reason"])

        skip_marker = "@" + "unittest.skip('skip')"
        res_add_skip = self._pre(
            "edit_file",
            {"Path": test_file, "Target": "    def test_login(self):", "Replacement": f"    {skip_marker}\n    def test_login(self):"},
        )
        self.assertEqual(res_add_skip["decision"], "deny")
        self.assertIn("adding skip/xfail decorators", res_add_skip["reason"])

        trivial_assert = "assert " + "True"
        res_trivial = self._pre(
            "edit_file",
            {"Path": test_file, "Target": "        self.assertEqual(a, b)", "Replacement": f"        {trivial_assert}"},
        )
        self.assertEqual(res_trivial["decision"], "deny")
        self.assertIn("trivial 'assert True'", res_trivial["reason"])

        res_add_test = self._pre(
            "edit_file",
            {
                "Path": test_file,
                "Target": "    def test_existing(self):\n        self.assertEqual(1, 1)",
                "Replacement": "    def test_existing(self):\n        self.assertEqual(1, 1)\n\n    def test_new(self):\n        self.assertEqual(2, 2)",
            },
        )
        self.assertEqual(res_add_test["decision"], "allow")

    def test_audit_diff_text_detects_test_tampering(self) -> None:
        res_rem = orchestrator_guard.audit_diff_text(
            "--- a/tests/test_auth.py\n+++ b/tests/test_auth.py\n@@ -10,3 +10,2 @@\n"
            "-    self.assertEqual(token, 'abc')\n     return True\n"
        )
        self.assertIsNotNone(res_rem)
        self.assertIn("Diff audit detected net removal of 1 assertion(s)", res_rem)

        skip_line = "+@" + "pytest.mark.skip"
        res_skip = orchestrator_guard.audit_diff_text(
            f"--- a/tests/test_auth.py\n+++ b/tests/test_auth.py\n@@ -5,2 +5,3 @@\n{skip_line}\n def test_login():\n     pass\n"
        )
        self.assertIsNotNone(res_skip)
        self.assertIn("Diff audit detected test skip/xfail marker added", res_skip)

        res_good = orchestrator_guard.audit_diff_text(
            "--- a/tests/test_auth.py\n+++ b/tests/test_auth.py\n@@ -20,2 +20,6 @@\n"
            "+def test_new_feature():\n+    self.assertEqual(1, 1)\n def test_existing():\n"
        )
        self.assertIsNone(res_good)

    def test_subagent_pre_invocation_injects_isolation_reminder_instead_of_tier_warning(self) -> None:
        self._reg_sub("scout", "sub-1111", "flash_lite")
        res = self._pre_inv(cid="sub-1111", modelName="gemini-3-flash-lite", invocationNum=1)
        self.assertIn("injectSteps", res)
        msg = res["injectSteps"][0]["ephemeralMessage"]
        self.assertIn("Subagent context isolation active", msg)
        self.assertIn("REPORT FORMAT", msg)

    def test_subagent_stop_enforces_four_line_report_format(self) -> None:
        sub_cid = "sub-stop-report-uuid-3333"
        self._reg_sub("implementer", sub_cid, "flash")
        res_bad = self._stop(
            "Here is my long verbose summary:\nLine 2\nLine 3\nLine 4\nLine 5",
            cid=sub_cid, modelName="gemini-3-flash", executionNum=1,
        )
        self.assertEqual(res_bad["decision"], "continue")
        self.assertIn("Subagent context-separation violation", res_bad["reason"])

        res_good = self._stop(
            "STATUS: COMPLETED\nFILES_CHANGED: NONE\nVERIFICATION: Ran unittest with exit code 0.\nNOTES: All checks passed.",
            cid=sub_cid, modelName="gemini-3-flash", executionNum=2,
        )
        self.assertEqual(res_good["decision"], "allow")

        orchestrator_guard.handle_pre_invocation(self.base_payload)
        res_root = self._stop(
            "Summary of work completed:\n\nFirst, we investigated the root cause of the bug.\n"
            "Second, we applied the fix across the affected components.\n\nFinally, all tests were verified successfully.",
            modelName="", executionNum=3,
        )
        self.assertEqual(res_root["decision"], "allow")

    def test_validate_subagent_report_allows_bounded_bullets_and_enforces_blocked_diagnosis(self) -> None:
        report_bullets = (
            "STATUS: COMPLETED\nFILES_CHANGED: src/app.py\n"
            "VERIFICATION: python3 -m unittest discover -s tests -v exited with 0.\n"
            "NOTES: All unit tests passed.\n- Added input validation check.\n- Updated docstrings.\n- Added unit tests in test_app.py."
        )
        self.assertIsNone(orchestrator_guard.validate_subagent_report(report_bullets))
        sub_cid = "sub-bullet-report-uuid-4444"
        self._reg_sub("implementer", sub_cid, "flash")
        self.assertEqual(
            self._stop(report_bullets, cid=sub_cid, modelName="gemini-3-flash", executionNum=10)["decision"],
            "allow",
        )

        err_non_bullet = orchestrator_guard.validate_subagent_report(
            "STATUS: COMPLETED\nFILES_CHANGED: NONE\nVERIFICATION: Ran unittest with exit 0.\n"
            "NOTES: Everything finished.\nThis is a non-bullet line."
        )
        self.assertIsNotNone(err_non_bullet)
        self.assertIn("must be a compact bullet", err_non_bullet)

        eleven_bullets = "\n".join(f"- bullet line {i}" for i in range(1, 12))
        err_too_many = orchestrator_guard.validate_subagent_report(
            f"STATUS: COMPLETED\nFILES_CHANGED: NONE\nVERIFICATION: Ran unittest with exit 0.\n"
            f"NOTES: Summary with 11 bullets.\n{eleven_bullets}"
        )
        self.assertIsNotNone(err_too_many)
        self.assertIn("at most 10 bullet lines", err_too_many)

        err_no_summary = orchestrator_guard.validate_subagent_report(
            "STATUS: BLOCKED\nFILES_CHANGED: NONE\nVERIFICATION: pytest tests/ failed with exit 1.\nNOTES: file:///tmp/scratch/fail.log"
        )
        self.assertIsNotNone(err_no_summary)
        self.assertIn("inline error summary", err_no_summary)

        err_no_uri = orchestrator_guard.validate_subagent_report(
            "STATUS: BLOCKED\nFILES_CHANGED: NONE\nVERIFICATION: pytest tests/ failed with exit 1.\nNOTES: AssertionError in test_login on line 42"
        )
        self.assertIsNotNone(err_no_uri)
        self.assertIn("scratch log URI", err_no_uri)

        self.assertIsNone(orchestrator_guard.validate_subagent_report(
            "STATUS: BLOCKED\nFILES_CHANGED: NONE\nVERIFICATION: pytest tests/ failed with exit 1.\n"
            "NOTES: AssertionError in test_login on line 42 at file:///tmp/scratch/fail.log"
        ))

    def test_orchestrator_direct_read_budget_and_log_peeking_blocked(self) -> None:
        file_a, file_b, file_c = (str(orchestrator_guard.PROJECT_ROOT / f"file_{x}.py") for x in ("a", "b", "c"))
        self.assertEqual(self._pre("read_file", {"Path": file_a})["decision"], "allow")
        self.assertEqual(self._pre("read_file", {"Path": file_b})["decision"], "allow")
        self.assertEqual(self._pre("read_file", {"Path": file_a})["decision"], "allow")

        res_c = self._pre("read_file", {"Path": file_c})
        self.assertEqual(res_c["decision"], "deny")
        self.assertIn("Direct execution fast-path read budget", res_c["reason"])

        fail_log = str(Path(self.artifact_dir) / "scratch" / "fail.log")
        transcript = str(Path(self.artifact_dir) / "transcript.jsonl")
        res_log = self._pre("read_file", {"Path": fail_log})
        self.assertEqual(res_log["decision"], "deny")
        self.assertIn("cannot directly read raw log or transcript", res_log["reason"])
        res_tr = self._pre("read_file", {"Path": transcript})
        self.assertEqual(res_tr["decision"], "deny")
        self.assertIn("cannot directly read raw log or transcript", res_tr["reason"])

        scout_cid = "sub-scout-read-cid-4444"
        self._reg_sub("scout", scout_cid, "flash_lite")
        self.assertEqual(self._pre("read_file", {"Path": fail_log}, cid=scout_cid, modelName="gemini-3-flash-lite")["decision"], "allow")
        self.assertEqual(self._pre("read_file", {"Path": file_c}, cid=scout_cid, modelName="gemini-3-flash-lite")["decision"], "allow")

    def test_small_model_root_orchestrator_is_treated_as_root(self) -> None:
        self.assertEqual(self._pre_inv(modelName="gemini-3-flash-lite"), {})
        f1, f2, f3 = (str(orchestrator_guard.PROJECT_ROOT / f"root_f{i}.txt") for i in (1, 2, 3))
        self.assertEqual(self._pre("write_file", {"Path": f1}, modelName="gemini-3-flash-lite")["decision"], "allow")
        self.assertEqual(self._pre("write_file", {"Path": f2}, modelName="gemini-3-flash-lite")["decision"], "allow")

        res3 = self._pre("write_file", {"Path": f3}, modelName="gemini-3-flash-lite")
        self.assertEqual(res3["decision"], "deny")
        self.assertIn("[ORCHESTRATOR GUARD]", res3["reason"])

        res_stop = self._stop(
            "This is a long verbose summary without four-line report format.\nMore details here.\nAnd here.",
            modelName="gemini-3-flash-lite",
        )
        self.assertEqual(res_stop["decision"], "allow")

    def test_research_subagent_can_write_scratch_markdown_but_not_workspace_files(self) -> None:
        sub_cid = "research-worker-001"
        self._reg_sub("research", sub_cid, "flash")
        scratch_file = str(Path(self.artifact_dir) / "scratch" / "findings.md")
        self.assertEqual(
            self._pre("write_file", {"Path": scratch_file, "Content": "# Research Findings\nAnalysis summary."}, cid=sub_cid)["decision"],
            "allow",
        )

        workspace_file = str(orchestrator_guard.PROJECT_ROOT / "src" / "app.py")
        res_workspace = self._pre("write_file", {"Path": workspace_file, "Content": "# Workspace mutation"}, cid=sub_cid)
        self.assertEqual(res_workspace["decision"], "deny")
        self.assertIn("Research subagents may only write markdown reports", res_workspace.get("reason", ""))

    def test_live_runtime_pitfalls_invocation_sync_and_cross_brain_research(self) -> None:
        payload_pre = self._payload(invocationNum=5)
        orchestrator_guard.handle_pre_invocation(payload_pre)
        self.assertEqual(self._load_st(payload_pre).get("current_invocation_num"), 5)

        tool_payload = self._payload(tool="run_command", args=self._cmd_args("echo 'pitfall 1 test'"), call_id="cmd_inv_5")
        tool_payload.pop("invocationNum", None)
        self.assertEqual(orchestrator_guard.handle_pre_tool(tool_payload).get("decision"), "allow")
        state_after_tool = self._load_st(tool_payload)
        self.assertIn("5", state_after_tool.get("direct_budget", {}))
        self.assertNotIn("0", state_after_tool.get("direct_budget", {}))

        parent_payload = self._brain_payload("parent-root", root="appdata/brain")
        sub_brain = Path(self.artifact_dir) / "appdata" / "brain" / "sub-research"
        sub_brain.mkdir(parents=True, exist_ok=True)
        (Path(parent_payload["artifactDirectoryPath"]) / "scratch").mkdir(parents=True, exist_ok=True)

        research_cid = "sub-research-cross-brain-001"
        self._reg_sub("research", research_cid, "flash", payload=parent_payload, role="Cross-Brain Researcher")
        sub_payload = self._payload(cid=research_cid, artifactDirectoryPath=str(sub_brain), modelName="gemini-3-flash")
        self.assertTrue(orchestrator_guard._is_scratch_only_subagent(sub_payload, self._load_st(sub_payload)))

        brief_file = str(Path(parent_payload["artifactDirectoryPath"]) / "scratch" / "brief.md")
        self.assertEqual(
            self._pre("write_file", {"Path": brief_file, "Content": "# Sibling Research Brief\nContent here."}, base=sub_payload).get("decision"),
            "allow",
        )
        workspace_file = str(orchestrator_guard.PROJECT_ROOT / "src.py")
        res_workspace = self._pre("write_file", {"Path": workspace_file, "Content": "x = 42\n"}, base=sub_payload)
        self.assertEqual(res_workspace.get("decision"), "deny")
        self.assertIn("Research subagents may only write markdown reports", res_workspace.get("reason", ""))

        root_state = self._load_st()
        receipt = orchestrator_guard._get_test_receipt(root_state, self.base_payload)
        receipt["dirty"], receipt["verified"] = True, False
        orchestrator_guard._save_state(self.base_payload, root_state)

        self._post("invoke_subagent", {}, call_id="call_invoke_sub_success")
        updated_receipt = orchestrator_guard._get_test_receipt(self._load_st(), self.base_payload)
        self.assertFalse(updated_receipt.get("dirty"))
        self.assertTrue(updated_receipt.get("verified"))
        self.assertEqual(updated_receipt.get("last_test_cmd"), "invoke_subagent")

    def test_direct_budget_uses_current_invocation_num_fallback(self) -> None:
        self._pre_inv(invocationNum=3)
        p1 = self._payload(tool="run_command", args=self._cmd_args("echo 'cmd 1'"), call_id="cmd_1")
        p1.pop("invocationNum", None)
        self.assertEqual(orchestrator_guard.handle_pre_tool(p1).get("decision"), "allow")
        p2 = self._payload(tool="run_command", args=self._cmd_args("echo 'cmd 2'"), call_id="cmd_2")
        p2.pop("invocationNum", None)
        self.assertEqual(orchestrator_guard.handle_pre_tool(p2).get("decision"), "allow")

        state = self._load_st()
        self.assertIn("3", state.get("direct_budget", {}))
        self.assertNotIn("0", state.get("direct_budget", {}))
        self.assertEqual(len(state["direct_budget"]["3"]["commands"]), 2)

        p3 = self._payload(tool="run_command", args=self._cmd_args("echo 'cmd 3'"), call_id="cmd_3")
        p3.pop("invocationNum", None)
        res_cmd3 = orchestrator_guard.handle_pre_tool(p3)
        self.assertEqual(res_cmd3.get("decision"), "deny")
        self.assertIn("command limit", res_cmd3.get("reason", ""))

        self._pre_inv(invocationNum=4)
        p4 = self._payload(tool="run_command", args=self._cmd_args("echo 'fresh budget cmd'"), call_id="cmd_4")
        p4.pop("invocationNum", None)
        self.assertEqual(orchestrator_guard.handle_pre_tool(p4).get("decision"), "allow")

        state_after = self._load_st()
        self.assertIn("4", state_after.get("direct_budget", {}))
        self.assertEqual(len(state_after["direct_budget"]["4"]["commands"]), 1)

        self.assertEqual(orchestrator_guard._current_invocation_key({"invocationNum": 0}, state_after), "4")
        self.assertEqual(orchestrator_guard._current_invocation_key({"invocationNum": "0"}, state_after), "4")
        self.assertEqual(orchestrator_guard._current_invocation_key({"invocationNum": 0}, {"current_invocation_num": 5}), "5")
        self.assertEqual(orchestrator_guard._current_invocation_key({"invocationNum": None}, state_after), "4")
        self.assertEqual(orchestrator_guard._current_invocation_key({}, state_after), "4")
        self.assertEqual(orchestrator_guard._current_invocation_key({"invocationNum": 7}, state_after), "7")
        self.assertEqual(orchestrator_guard._current_invocation_key({"invocationNum": "7"}, state_after), "7")
        self.assertEqual(orchestrator_guard._current_invocation_key({"invocationNum": 0}, {"current_invocation_num": 0}), "0")
        self.assertEqual(orchestrator_guard._current_invocation_key({"invocationNum": "0"}, {"current_invocation_num": 0}), "0")

        self.assertEqual(
            self._pre("run_command", self._cmd_args("echo 'proto0 cmd'"), call_id="cmd_proto0", invocationNum=0).get("decision"),
            "allow",
        )
        state_after_proto0 = self._load_st()
        self.assertEqual(len(state_after_proto0["direct_budget"]["4"]["commands"]), 2)
        self.assertNotIn("0", state_after_proto0.get("direct_budget", {}))

        st_pre = self._load_st()
        st_pre["direct_budget"]["5"] = {"commands": ["old_cmd_5"]}
        st_pre["direct_budget"]["0"] = {"commands": ["old_cmd_0"]}
        orchestrator_guard._save_state(self.base_payload, st_pre)
        self._pre_inv(invocationNum=5)
        state_after_inv5 = self._load_st()
        self.assertEqual(state_after_inv5.get("current_invocation_num"), 5)
        self.assertNotIn("5", state_after_inv5.get("direct_budget", {}))
        self.assertNotIn("0", state_after_inv5.get("direct_budget", {}))

        state_after_inv5["direct_budget"]["0"] = {"commands": ["old_cmd_0"]}
        state_after_inv5["direct_budget"]["6"] = {"commands": ["preserved_cmd_6"]}
        orchestrator_guard._save_state(self.base_payload, state_after_inv5)
        self._pre_inv(invocationNum=0)
        state_proto_cleared = self._load_st()
        self.assertEqual(state_proto_cleared.get("current_invocation_num"), 5)
        self.assertNotIn("0", state_proto_cleared.get("direct_budget", {}))
        self.assertIn("6", state_proto_cleared.get("direct_budget", {}))

        state_proto_cleared["direct_budget"]["0"] = {"commands": ["old_cmd_0"]}
        orchestrator_guard._save_state(self.base_payload, state_proto_cleared)
        self._pre_inv(invocationNum="0")
        state_str_cleared = self._load_st()
        self.assertEqual(state_str_cleared.get("current_invocation_num"), 5)
        self.assertNotIn("0", state_str_cleared.get("direct_budget", {}))

        state_after["direct_budget"]["0"] = {"commands": ["old_cmd"]}
        orchestrator_guard._save_state(self.base_payload, state_after)
        p_omit = dict(self.base_payload)
        p_omit.pop("invocationNum", None)
        orchestrator_guard.handle_pre_invocation(p_omit)
        self.assertNotIn("0", self._load_st().get("direct_budget", {}))

    def test_compound_git_and_pytest_command_not_exempt(self) -> None:
        orchestrator_guard.handle_pre_invocation(self.base_payload)

        dummy_file = orchestrator_guard.PROJECT_ROOT / "test_dummy_source.py"
        try:
            self.assertEqual(
                self._pre("write_file", {"Path": str(dummy_file), "Content": "x = 1\n"}).get("decision"),
                "allow",
            )
            self.assertEqual(
                self._pre("run_command", self._cmd_args("git status && pytest"), call_id="compound_1").get("decision"),
                "allow",
            )
            res_cmd_2 = self._pre("run_command", self._cmd_args("git status && pytest"), call_id="compound_2")
            self.assertEqual(res_cmd_2.get("decision"), "deny")
            self.assertIn("Direct workspace edit followed by iterative test execution detected", res_cmd_2.get("reason", ""))
        finally:
            if dummy_file.exists():
                dummy_file.unlink()

        clean_inv = self._payload(invocationNum=99)
        orchestrator_guard.handle_pre_invocation(clean_inv)
        res_clean = self._pre("run_command", self._cmd_args("git status && pytest"), call_id="clean_cmd_1", base=clean_inv)
        self.assertEqual(res_clean.get("decision"), "allow")
        self.assertIn("git status && pytest", self._load_st(clean_inv)["direct_budget"]["99"]["commands"])

    def test_grep_search_on_log_or_transcript_blocked_for_root(self) -> None:
        orchestrator_guard.handle_pre_invocation(self.base_payload)

        res_log = self._pre("grep_search", {"SearchPath": "/tmp/debug.log", "Query": "ERROR"})
        self.assertEqual(res_log.get("decision"), "deny")
        self.assertIn("Context separation violation", res_log.get("reason", ""))
        self.assertIn("debug.log", res_log.get("reason", ""))

        res_transcript = self._pre("grep_search", {"SearchPath": "transcript.jsonl", "Query": "conversationId"})
        self.assertEqual(res_transcript.get("decision"), "deny")
        self.assertIn("Context separation violation", res_transcript.get("reason", ""))

        res_normal = self._pre("grep_search", {"SearchPath": str(orchestrator_guard.PROJECT_ROOT / "tests"), "Query": "test"})
        self.assertEqual(res_normal.get("decision"), "allow")

    def test_transcript_offset_cursor_and_doc_uuid_ignored(self) -> None:
        transcript_file = Path(self.artifact_dir) / "transcript_cursor.jsonl"
        payload = self._payload(transcriptPath=str(transcript_file))
        state = self._seed_state(payload, pending_spawns=[{"role": "Worker 1", "type": "implementer", "model": "flash"}])

        doc_cid = "aaaa1111-aaaa-1111-aaaa-1111aaaa1111"
        transcript_file.write_text(
            json.dumps({
                "stepId": "00000000-0000-0000-0000-000000000001",
                "name": "view_file",
                "args": {"AbsolutePath": "/workspace/docs/hooks.md"},
                "content": f"Created the following subagents: conversationId: {doc_cid}",
            }) + "\n",
            encoding="utf-8",
        )

        self.assertTrue(orchestrator_guard._sync_subagents_from_transcript(payload, state))
        offset_after_first = state.get("transcript_offset", 0)
        self.assertGreater(offset_after_first, 0)
        self.assertNotIn(doc_cid, state.get("subagents", {}))
        self.assertEqual(len(state.get("pending_spawns", [])), 1)

        real_cid = "bbbb2222-bbbb-2222-bbbb-2222bbbb2222"
        with open(transcript_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "stepId": "00000000-0000-0000-0000-000000000002",
                "name": "invoke_subagent",
                "result": {"conversationId": real_cid},
            }) + "\n")

        self.assertTrue(orchestrator_guard._sync_subagents_from_transcript(payload, state))
        offset_after_second = state.get("transcript_offset", 0)
        self.assertGreater(offset_after_second, offset_after_first)
        self.assertIn(real_cid, state.get("subagents", {}))
        self.assertEqual(len(state.get("pending_spawns", [])), 0)

        self.assertFalse(orchestrator_guard._sync_subagents_from_transcript(payload, state))
        self.assertEqual(state.get("transcript_offset"), offset_after_second)

        singular_cid = "cccc3333-cccc-3333-cccc-3333cccc3333"
        state["pending_spawns"] = [{"role": "Worker 2", "type": "scout", "model": "flash_lite"}]
        with open(transcript_file, "a", encoding="utf-8") as fh:
            fh.write(f"Created the following subagent: conversationId: {singular_cid}\n")

        self.assertTrue(orchestrator_guard._sync_subagents_from_transcript(payload, state))
        self.assertIn(singular_cid, state.get("subagents", {}))
        self.assertEqual(state.get("subagents", {})[singular_cid]["model"], "flash_lite")
        self.assertEqual(len(state.get("pending_spawns", [])), 0)

    def test_self_subagent_meta_cached_from_sibling_brain(self) -> None:
        parent_payload = self._brain_payload("parent-orch-cid", root="appdata/brain")
        sub_payload = self._brain_payload("sub-worker-cid", root="appdata/brain", modelName="gemini-3-flash")
        worker_cid = "sub-worker-cid-5555"
        self._reg_sub("research", worker_cid, "flash", payload=parent_payload, role="Worker Researcher")

        sub_payload = self._payload(cid=worker_cid, base=sub_payload)
        sub_state = self._load_st(sub_payload)
        self.assertNotIn("self_subagent_meta", sub_state)

        self.assertTrue(orchestrator_guard._is_known_subagent(sub_payload, sub_state))
        self.assertIn("self_subagent_meta", sub_state)
        self.assertEqual(sub_state["self_subagent_meta"]["type_name"], "research")

        persisted_sub_state = self._load_st(sub_payload)
        self.assertIn("self_subagent_meta", persisted_sub_state)
        self.assertEqual(persisted_sub_state["self_subagent_meta"]["type_name"], "research")
        self.assertTrue(orchestrator_guard._is_scratch_only_subagent(sub_payload, persisted_sub_state))

    def test_commented_out_assertion_tampering_blocked_in_edit_and_diff(self) -> None:
        test_file = str(orchestrator_guard.PROJECT_ROOT / "tests" / "test_sample.py")
        res_edit = self._pre(
            "edit_file",
            {"Path": test_file, "Target": "        self.assertEqual(a, b)\n", "Replacement": "        # self.assertEqual(a, b)\n"},
        )
        self.assertEqual(res_edit["decision"], "deny")
        self.assertIn("reduces assertion count", res_edit["reason"])

        res_diff = orchestrator_guard.audit_diff_text(
            "--- a/tests/test_auth.py\n+++ b/tests/test_auth.py\n@@ -10,1 +10,1 @@\n"
            "-    self.assertEqual(token, 'abc')\n+    # self.assertEqual(token, 'abc')\n"
        )
        self.assertIsNotNone(res_diff)
        self.assertIn("Diff audit detected net removal of 1 assertion(s)", res_diff)

    def test_step_idx_deduplication_and_scout_does_not_clear_dirty_receipt(self) -> None:
        orchestrator_guard.handle_pre_invocation(self.base_payload)

        self.assertEqual(self._pre("run_command", self._cmd_args("echo 'step test'"), stepIdx=42)["decision"], "allow")
        self.assertEqual(self._pre("run_command", self._cmd_args("echo 'step test'"), stepIdx=42)["decision"], "allow")
        state = self._load_st()
        self.assertEqual(len(state["direct_budget"]["1"]["commands"]), 1)

        receipt = orchestrator_guard._get_test_receipt(state, self.base_payload)
        receipt["dirty"], receipt["verified"] = True, False
        orchestrator_guard._save_state(self.base_payload, state)

        self._post("invoke_subagent", {"TypeName": "scout", "Role": "Read Scout"}, call_id="scout_post_call")
        receipt_after_scout = orchestrator_guard._get_test_receipt(self._load_st(), self.base_payload)
        self.assertTrue(receipt_after_scout["dirty"])
        self.assertFalse(receipt_after_scout["verified"])

        self._post("run_command", self._cmd_args("echo pytest"), call_id="echo_post_call")
        receipt_after_echo = orchestrator_guard._get_test_receipt(self._load_st(), self.base_payload)
        self.assertTrue(receipt_after_echo["dirty"])
        self.assertFalse(receipt_after_echo["verified"])

    def test_cli_doctor_and_status_modes(self) -> None:
        status_res = orchestrator_guard._cli_status(self.base_payload)
        for k in ("projectRoot", "stateFile", "active", "currentInvocation", "directBudget", "testReceipt", "circuitBreaker"):
            self.assertIn(k, status_res)

        doctor_res = orchestrator_guard._cli_doctor()
        self.assertIn("healthy", doctor_res)
        self.assertIn("pluginHooksPresent", doctor_res)
        self.assertIn("issues", doctor_res)
        self.assertTrue(doctor_res["pluginHooksPresent"])
        self.assertIsInstance(doctor_res["issues"], list)

    def test_inactive_session_on_normal_chat(self) -> None:
        normal_payload = self._payload(
            tool="read_file",
            args={"Path": str(orchestrator_guard.PROJECT_ROOT / "README.md")},
            lastUserInput="normal chat message",
        )
        self.assertEqual(orchestrator_guard.handle_pre_invocation(normal_payload), {})
        state = self._load_st(normal_payload)
        self.assertFalse(state.get("active"))
        self.assertTrue(state.get("session_evaluated"))
        self.assertEqual(orchestrator_guard.handle_pre_tool(normal_payload), {"decision": "allow"})
        self.assertEqual(orchestrator_guard.handle_post_tool(normal_payload), {})
        self.assertEqual(self._stop(base=normal_payload), {})

    def test_slash_command_activation_and_persistence_across_turns(self) -> None:
        self.assertEqual(self._pre_inv(lastUserInput="/philharmonic implement feature X", invocationNum=1), {})
        state = self._load_st()
        self.assertTrue(state.get("active"))
        self.assertTrue(state.get("session_evaluated"))

        self.assertEqual(self._pre_inv(lastUserInput="now run the tests", invocationNum=2), {})
        self.assertTrue(self._load_st().get("active"))
        self.assertEqual(
            self._pre("read_file", {"Path": str(orchestrator_guard.PROJECT_ROOT / "README.md")}, lastUserInput="now run the tests", invocationNum=2),
            {"decision": "allow"},
        )

    def test_reading_skill_file_activates_inactive_session(self) -> None:
        self._pre_inv(lastUserInput="please look up some instructions")
        self.assertFalse(self._load_st().get("active"))

        skill_path = str(
            orchestrator_guard.PROJECT_ROOT / ".agents" / "plugins" / "philharmonic_orchestra" / "skills" / "philharmonic" / "SKILL.md"
        )
        self.assertEqual(self._pre("read_file", {"Path": skill_path}), {"decision": "allow"})
        state_after = self._load_st()
        self.assertTrue(state_after.get("active"))
        self.assertTrue(state_after.get("session_evaluated"))

    def test_cli_activate_and_deactivate_modes(self) -> None:
        act_res = orchestrator_guard._cli_activate(self.base_payload)
        self.assertEqual(act_res["status"], "activated")
        self.assertTrue(act_res["active"])
        self.assertIn("state_path", act_res)
        self.assertTrue(orchestrator_guard._cli_status(self.base_payload)["active"])

        deact_res = orchestrator_guard._cli_deactivate(self.base_payload)
        self.assertEqual(deact_res["status"], "deactivated")
        self.assertFalse(deact_res["active"])
        self.assertFalse(orchestrator_guard._cli_status(self.base_payload)["active"])

    def test_philharmonic_skill_file_exists_and_valid_frontmatter(self) -> None:
        skill_path = (
            orchestrator_guard.PROJECT_ROOT / ".agents" / "plugins" / "philharmonic_orchestra" / "skills" / "philharmonic" / "SKILL.md"
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
        self._pre_inv(lastUserInput="hello remote vm")
        self.assertFalse(self._load_st().get("active"))

        self.assertEqual(self._pre("lookup_skill", {"skill_name": "philharmonic"})["decision"], "allow")
        self.assertTrue(self._load_st().get("active"))

        orchestrator_guard._cli_deactivate(self.base_payload)
        self.assertFalse(self._load_st().get("active"))
        self.assertEqual(self._pre("lookup_skills", {"skill_names": ["other_skill", "philharmonic"]})["decision"], "allow")
        self.assertTrue(self._load_st().get("active"))

        res_define = self._pre("define_subagent", {"name": "scout", "enable_write_tools": False, "enable_subagent_tools": True})
        self.assertEqual(res_define["decision"], "allow")
        self.assertEqual(res_define["overwrite"], {"enable_subagent_tools": False})

        res_pro = self._pre(
            "invoke_subagent",
            {"Subagents": [self._sub_args("implementer", "Pro Worker", "pro")]},
            call_id="remote_sub_pro",
        )
        self.assertEqual(res_pro["decision"], "deny")
        self.assertIn("Model='pro'", res_pro["reason"])

        inherit_args = {
            "Subagents": [
                self._sub_args("scout", "Remote Scout", "inherit"),
                self._sub_args("implementer", "Remote Implementer", "inherit"),
            ]
        }
        res_inherit = self._pre("invoke_subagent", inherit_args, call_id="remote_sub_inherit")
        self.assertEqual(res_inherit["decision"], "allow")
        self.assertEqual(res_inherit["overwrite"]["Subagents"][0]["Model"], "flash_lite")
        self.assertEqual(res_inherit["overwrite"]["Subagents"][1]["Model"], "flash")
        self.assertEqual(len(self._load_st().get("pending_spawns", [])), 2)

        self._post("invoke_subagent", inherit_args, error="remote subagent spawn failed", call_id="remote_sub_inherit")
        self.assertEqual(len(self._load_st().get("pending_spawns", [])), 0)

        test_file = str(orchestrator_guard.PROJECT_ROOT / "tests" / "test_remote.py")
        res_tamper = self._pre(
            "replace_file_content",
            {"TargetFile": test_file, "TargetContent": "    assert x == 1\n", "ReplacementContent": "    # assert x == 1\n"},
        )
        self.assertEqual(res_tamper["decision"], "deny")
        self.assertIn("reduces assertion count", res_tamper["reason"])

        res_tamper_wtf = self._pre("write_to_file", {"TargetFile": test_file, "CodeContent": "def test_x():\n    assert True\n"})
        self.assertEqual(res_tamper_wtf["decision"], "deny")
        self.assertIn("assert True", res_tamper_wtf["reason"])

        code_file = str(orchestrator_guard.PROJECT_ROOT / "remote_code.py")
        self.assertEqual(self._pre("write_to_file", {"TargetFile": code_file, "CodeContent": "val = 123\n"})["decision"], "allow")
        self.assertTrue(orchestrator_guard._get_test_receipt(self._load_st(), self.base_payload).get("dirty"))

        self._post(
            "invoke_subagent",
            {"Subagents": [{"TypeName": "scout", "Role": "Read Scout"}, {"TypeName": "research", "Role": "Doc Researcher"}]},
            call_id="remote_post_readonly",
        )
        self.assertTrue(orchestrator_guard._get_test_receipt(self._load_st(), self.base_payload).get("dirty"))

        self._post(
            "invoke_subagent",
            {"Subagents": [{"TypeName": "scout", "Role": "Read Scout"}, {"TypeName": "implementer", "Role": "Code Implementer"}]},
            call_id="remote_post_impl",
        )
        receipt_verified = orchestrator_guard._get_test_receipt(self._load_st(), self.base_payload)
        self.assertFalse(receipt_verified.get("dirty"))
        self.assertTrue(receipt_verified.get("verified"))

        for fname in ("remote_a.py", "remote_b.py"):
            self.assertEqual(self._pre("view_file", {"AbsolutePath": str(orchestrator_guard.PROJECT_ROOT / fname)})["decision"], "allow")
        r3 = self._pre("view_file", {"AbsolutePath": str(orchestrator_guard.PROJECT_ROOT / "remote_c.py")})
        self.assertEqual(r3["decision"], "deny")
        self.assertIn("Direct execution fast-path read budget", r3["reason"])

    def test_redirection_splitting_and_regex_literal_paths_in_workspace(self) -> None:
        self.assertEqual(orchestrator_guard._split_shell_commands("git diff 2>&1"), ["git diff 2>&1"])
        self.assertTrue(orchestrator_guard._is_git_only_command("git diff 2>&1"))
        self.assertFalse(orchestrator_guard._is_git_only_command("git diff 2>&1 && pytest"))

        roots = orchestrator_guard._allowed_roots(self.base_payload)
        cwd = str(orchestrator_guard.PROJECT_ROOT)
        self.assertTrue(orchestrator_guard._command_stays_in_workspace("awk '/pattern/' README.md", cwd, roots))
        self.assertTrue(orchestrator_guard._command_stays_in_workspace("sed '/foo/d' README.md", cwd, roots))
        self.assertFalse(orchestrator_guard._command_stays_in_workspace("cat /etc/passwd", cwd, roots))

    def test_tripped_circuit_breaker_blocks_compound_git_command(self) -> None:
        state = self._load_st()
        cb = orchestrator_guard._get_circuit_breaker(state)
        cb["failure_streak"], cb["tripped"], cb["last_error"] = 2, True, "exit 1"
        orchestrator_guard._save_state(self.base_payload, state)

        res_compound = self._pre("run_command", self._cmd_args("git status && pytest"))
        self.assertEqual(res_compound["decision"], "deny")
        self.assertIn("2-turn circuit breaker tripped", res_compound["reason"])
        self.assertEqual(self._pre("run_command", self._cmd_args("git status"))["decision"], "allow")
        self.assertEqual(self._pre("run_command", self._cmd_args("git diff 2>&1"))["decision"], "allow")

    def test_parallel_sibling_brain_subagents_claim_distinct_pending_spawns(self) -> None:
        parent_payload = self._brain_payload("parent-cid", root="appdata_parallel/brain")
        sub1_payload = self._brain_payload("sub-1-cid", root="appdata_parallel/brain")
        sub2_payload = self._brain_payload("sub-2-cid", root="appdata_parallel/brain")

        res = self._pre(
            "invoke_subagent",
            {"Subagents": [self._sub_args("scout", "First Scout", "flash_lite"), self._sub_args("implementer", "Second Implementer", "flash")]},
            call_id="parallel_spawn_1",
            base=parent_payload,
        )
        self.assertEqual(res["decision"], "allow")

        sub1_state = self._load_st(sub1_payload)
        self.assertTrue(orchestrator_guard._is_known_subagent(sub1_payload, sub1_state))
        self.assertEqual(sub1_state["self_subagent_meta"]["type_name"], "scout")

        sub2_state = self._load_st(sub2_payload)
        self.assertTrue(orchestrator_guard._is_known_subagent(sub2_payload, sub2_state))
        self.assertEqual(sub2_state["self_subagent_meta"]["type_name"], "implementer")

        parent_state_after = self._load_st(parent_payload)
        self.assertEqual(len(parent_state_after.get("pending_spawns", [])), 0)
        self.assertEqual(parent_state_after["subagents"]["sub-1-cid"]["type_name"], "scout")
        self.assertEqual(parent_state_after["subagents"]["sub-2-cid"]["type_name"], "implementer")

    def test_scout_readonly_enforced_in_define_and_pre_tool(self) -> None:
        res_def1 = self._pre("define_subagent", {"Name": "scout", "EnableWriteTools": True, "EnableSubagentTools": True})
        self.assertEqual(res_def1["decision"], "allow")
        self.assertEqual(res_def1["overwrite"], {"EnableSubagentTools": False, "EnableWriteTools": False})

        res_def2 = self._pre("define_subagent", {"name": "scout", "enable_write_tools": True, "enable_subagent_tools": False})
        self.assertEqual(res_def2["decision"], "allow")
        self.assertEqual(res_def2["overwrite"], {"enable_write_tools": False})

        scout_cid = "scout-readonly-worker-01"
        self._reg_sub("scout", scout_cid, "flash_lite")
        scratch_md = str(Path(self.artifact_dir) / "scratch" / "notes.md")
        res_scout_write = self._pre("write_file", {"Path": scratch_md, "Content": "hello"}, cid=scout_cid)
        self.assertEqual(res_scout_write["decision"], "deny")
        self.assertIn("Scout subagents", res_scout_write["reason"])

    def test_blocked_or_failed_subagent_output_does_not_clear_dirty_receipt(self) -> None:
        state = self._load_st()
        receipt = orchestrator_guard._get_test_receipt(state)
        receipt["dirty"], receipt["verified"] = True, False
        orchestrator_guard._save_state(self.base_payload, state)

        blocked_report = (
            "STATUS: BLOCKED\nFILES_CHANGED: NONE\nVERIFICATION: pytest failed twice.\n"
            "NOTES: Failing test in auth module at file:///tmp/scratch/fail.log"
        )
        self._post("invoke_subagent", {"TypeName": "implementer"}, output=blocked_report, call_id="impl_blocked_call")
        receipt_after = orchestrator_guard._get_test_receipt(self._load_st())
        self.assertTrue(receipt_after["dirty"])
        self.assertFalse(receipt_after["verified"])

    def test_prune_state_bounds_deduplication_lists_and_cli_reset(self) -> None:
        state = self._load_st()
        limit = orchestrator_guard.MAX_STATE_LIST_ENTRIES
        state["isolation_reminded_invocations"] = [f"inv:{i}" for i in range(limit + 25)]
        state["recorded_spawn_calls"] = [f"spawn:{i}" for i in range(limit + 25)]
        state["stop_reminded_executions"] = [f"stop:{i}" for i in range(limit + 25)]
        orchestrator_guard._get_circuit_breaker(state)["seen_failures"] = [f"fail:{i}" for i in range(limit + 25)]
        orchestrator_guard._save_state(self.base_payload, state)

        reloaded = self._load_st()
        self.assertEqual(len(reloaded["isolation_reminded_invocations"]), limit)
        self.assertEqual(len(reloaded["recorded_spawn_calls"]), limit)
        self.assertEqual(len(reloaded["stop_reminded_executions"]), limit)
        self.assertEqual(len(orchestrator_guard._get_circuit_breaker(reloaded)["seen_failures"]), limit)
        self.assertEqual(reloaded["recorded_spawn_calls"][-1], f"spawn:{limit + 24}")

        reset_res = orchestrator_guard._cli_reset(self.base_payload)
        self.assertEqual(reset_res["status"], "reset")
        self.assertEqual(self._load_st()["recorded_spawn_calls"], [])

    def test_cross_brain_subagent_receipt_propagation_and_cb_invocation_reset(self) -> None:
        parent_payload = self._brain_payload("parent-root-cid", root="appdata_receipt_prop/brain", invocationNum=2)
        sub_payload = self._brain_payload("sub-impl-cid", root="appdata_receipt_prop/brain")

        orchestrator_guard.handle_pre_invocation(parent_payload)
        self._reg_sub("implementer", "sub-impl-cid", "flash", payload=parent_payload, role="Cross-Brain Implementer")

        code_path = str(orchestrator_guard.PROJECT_ROOT / "src_cross_brain.py")
        self.assertEqual(self._pre("write_file", {"Path": code_path, "Content": "x = 1\n"}, base=sub_payload)["decision"], "allow")

        parent_receipt = orchestrator_guard._get_test_receipt(self._load_st(parent_payload))
        self.assertTrue(parent_receipt["dirty"])
        self.assertFalse(parent_receipt["verified"])

        res_stop_1 = self._stop(base=parent_payload, executionNum=1)
        self.assertEqual(res_stop_1["decision"], "continue")
        self.assertIn("Empirical test receipt missing", res_stop_1["reason"])

        self._post("run_command", self._cmd_args("pytest tests/"), call_id="sub_pytest_pass", base=sub_payload)
        parent_state_2 = self._load_st(parent_payload)
        parent_receipt_2 = orchestrator_guard._get_test_receipt(parent_state_2)
        self.assertFalse(parent_receipt_2["dirty"])
        self.assertTrue(parent_receipt_2["verified"])
        self.assertEqual(self._stop(base=parent_payload, executionNum=2)["decision"], "allow")

        cb = orchestrator_guard._get_circuit_breaker(parent_state_2)
        cb["failure_streak"], cb["tripped"], cb["last_error"] = 2, True, "failed twice on inv 2"
        orchestrator_guard._save_state(parent_payload, parent_state_2)

        parent_inv3 = dict(parent_payload, invocationNum=3)
        orchestrator_guard.handle_pre_invocation(parent_inv3)
        cb_3 = orchestrator_guard._get_circuit_breaker(self._load_st(parent_inv3))
        self.assertFalse(cb_3["tripped"])
        self.assertEqual(cb_3["failure_streak"], 0)

    def test_allowed_files_header_enforces_subagent_file_allowlist(self) -> None:
        parent_payload = self._brain_payload("parent-allow-cid", root="appdata_allowed_files/brain")
        sub1_payload = self._brain_payload("sub-allow-1", root="appdata_allowed_files/brain")
        sub2_payload = self._brain_payload("sub-allow-2", root="appdata_allowed_files/brain")

        prompt_1 = (
            "TASK: Implement feature A.\nALLOWED_FILES: module_a.py, tests/test_a.py\n\n"
            "RULES:\n1. Open `module_a.py`.\n2. Do not change any file outside `ALLOWED_FILES`.\n\n"
            "REPORT FORMAT:\nSTATUS: COMPLETED | BLOCKED | FAILED\nFILES_CHANGED: module_a.py\nVERIFICATION: pytest\nNOTES: done\n"
        )
        prompt_2 = (
            "TASK: Implement feature B.\nALLOWED_FILES: module_b.py\n\n"
            "RULES:\n1. Open `module_b.py`.\n2. Do not change any file outside `ALLOWED_FILES`.\n\n"
            "REPORT FORMAT:\nSTATUS: COMPLETED | BLOCKED | FAILED\nFILES_CHANGED: module_b.py\nVERIFICATION: pytest\nNOTES: done\n"
        )
        self.assertEqual(
            self._pre(
                "invoke_subagent",
                {"Subagents": [self._sub_args("implementer", "Worker A", "flash", prompt_1), self._sub_args("implementer", "Worker B", "flash", prompt_2)]},
                call_id="spawn_allow_pair",
                base=parent_payload,
            )["decision"],
            "allow",
        )

        mod_a = str(orchestrator_guard.PROJECT_ROOT / "module_a.py")
        mod_b = str(orchestrator_guard.PROJECT_ROOT / "module_b.py")
        mod_c = str(orchestrator_guard.PROJECT_ROOT / "module_c.py")
        scratch_log = str(Path(sub1_payload["artifactDirectoryPath"]) / "scratch" / "fail.log")

        self.assertEqual(self._pre("write_file", {"Path": mod_a, "Content": "a = 1\n"}, base=sub1_payload)["decision"], "allow")
        self.assertEqual(self._pre("write_to_file", {"TargetFile": scratch_log, "CodeContent": "error\n"}, base=sub1_payload)["decision"], "allow")
        res_sub1_deny = self._pre("replace_file_content", {"TargetFile": mod_b}, base=sub1_payload)
        self.assertEqual(res_sub1_deny["decision"], "deny")
        self.assertIn("File allowlist violation", res_sub1_deny["reason"])

        self.assertEqual(self._pre("write_file", {"Path": mod_b, "Content": "b = 2\n"}, base=sub2_payload)["decision"], "allow")
        res_sub2_deny = self._pre("write_file", {"Path": mod_a, "Content": "a = 2\n"}, base=sub2_payload)
        self.assertEqual(res_sub2_deny["decision"], "deny")
        self.assertIn("File allowlist violation", res_sub2_deny["reason"])

        self.assertEqual(
            self._pre(
                "send_message",
                {"Recipient": "sub-allow-1", "Message": "TASK: Also update module_c.py.\nALLOWED_FILES: module_a.py, module_c.py\n"},
                base=parent_payload,
            )["decision"],
            "allow",
        )
        self.assertEqual(self._pre("write_file", {"Path": mod_c, "Content": "c = 3\n"}, base=sub1_payload)["decision"], "allow")

        self.assertEqual(
            self._pre(
                "send_message",
                {"Recipient": "sub-allow-1", "Message": "TASK: Read-only check.\nALLOWED_FILES: NONE\n"},
                base=parent_payload,
            )["decision"],
            "allow",
        )
        res_sub1_none_deny = self._pre("write_file", {"Path": mod_a, "Content": "a = 99\n"}, base=sub1_payload)
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

    def test_c1_no_wildcard_command_override_and_redirection_bare_root_symlink_guards(self) -> None:
        import os

        cmd = "git status"
        res = self._pre("run_command", {"CommandLine": cmd})
        self.assertEqual(res.get("decision"), "allow")
        overrides = res.get("permissionOverrides", [])
        self.assertNotIn("command(*)", overrides)
        self.assertIn(f"command({cmd})", overrides)
        self.assertTrue(any(str(o).startswith("file(") for o in overrides))

        roots = [Path(self.artifact_dir).resolve()]
        iso = self._payload(workspacePaths=[str(self.artifact_dir)])
        cwd = str(self.artifact_dir)

        self.assertFalse(orchestrator_guard._command_stays_in_workspace("cat >/etc/passwd", cwd=cwd, roots=roots))
        self.assertFalse(orchestrator_guard._command_stays_in_workspace("echo hi>>/etc/hosts", cwd=cwd, roots=roots))
        self.assertEqual(self._pre("run_command", self._cmd_args("cat >/etc/passwd", cwd), base=iso).get("decision"), "deny")
        self.assertEqual(self._pre("run_command", self._cmd_args("cat /etc/passwd", cwd), base=iso).get("decision"), "ask")

        self.assertFalse(orchestrator_guard._command_stays_in_workspace("ls /", cwd=cwd, roots=roots))
        self.assertEqual(self._pre("run_command", self._cmd_args("ls /", cwd), base=iso).get("decision"), "ask")

        self.assertFalse(orchestrator_guard._command_stays_in_workspace("ls $HOME", cwd=cwd, roots=roots))
        self.assertEqual(self._pre("run_command", self._cmd_args("ls $HOME", cwd), base=iso).get("decision"), "ask")

        oldpwd_orig = os.environ.get("OLDPWD")
        try:
            os.environ["OLDPWD"] = "/var/log"
            self.assertFalse(orchestrator_guard._command_stays_in_workspace("ls $OLDPWD", cwd=cwd, roots=roots))
            self.assertEqual(self._pre("run_command", self._cmd_args("ls $OLDPWD", cwd), base=iso).get("decision"), "ask")
        finally:
            if oldpwd_orig is not None:
                os.environ["OLDPWD"] = oldpwd_orig
            else:
                os.environ.pop("OLDPWD", None)

        symlink_path = Path(self.artifact_dir) / "link_to_hosts"
        if not symlink_path.exists():
            symlink_path.symlink_to("/etc/hosts")
        self.assertFalse(orchestrator_guard._command_stays_in_workspace("cat link_to_hosts", cwd=cwd, roots=roots))
        self.assertEqual(self._pre("run_command", self._cmd_args("cat link_to_hosts", cwd), base=iso).get("decision"), "ask")

    def test_c2_allowed_files_blocks_nested_basename_collision_and_shell_writes(self) -> None:
        self._reg_sub("implementer", "imp-c2", "flash", allowed_files=["module_a.py"])

        self.assertEqual(
            self._pre("write_file", {"Path": str(orchestrator_guard.PROJECT_ROOT / "module_a.py"), "Content": "x = 1\n"}, cid="imp-c2").get("decision"),
            "allow",
        )
        res_nested = self._pre(
            "write_file",
            {"Path": str(orchestrator_guard.PROJECT_ROOT / "nested_dir" / "module_a.py"), "Content": "x = 1\n"},
            cid="imp-c2",
        )
        self.assertEqual(res_nested.get("decision"), "deny")
        self.assertIn("File allowlist violation", res_nested.get("reason", ""))

        for cmd in ("echo hacked > module_b.py", "rm module_b.py"):
            res_cmd = self._pre("run_command", {"CommandLine": cmd}, cid="imp-c2")
            self.assertEqual(res_cmd.get("decision"), "deny")
            self.assertIn("File allowlist violation", res_cmd.get("reason", ""))

        self._reg_sub("scout", "scout-c2", "flash_lite")
        self.assertEqual(self._pre("run_command", {"CommandLine": "echo x > out.txt"}, cid="scout-c2").get("decision"), "deny")
        self.assertEqual(self._pre("run_command", {"CommandLine": "rm README.md"}, cid="scout-c2").get("decision"), "deny")
        self.assertEqual(self._pre("run_command", {"CommandLine": "git status"}, cid="scout-c2").get("decision"), "allow")
        self.assertEqual(self._pre("run_command", {"CommandLine": "ls"}, cid="scout-c2").get("decision"), "allow")

    def test_c3_nonzero_command_exit_output_fails_receipt_and_trips_cb_while_allowing_subagent_blocked_report(self) -> None:
        self._pre("write_file", {"Path": str(orchestrator_guard.PROJECT_ROOT / "code_c3.py"), "Content": "val = 42\n"})
        self.assertTrue(orchestrator_guard._get_test_receipt(self._load_st()).get("dirty"))

        self._post("run_command", {"CommandLine": "pytest tests/"}, output="Command exited with code 1\n1 failed", call_id="post-call-c3-1")
        st1 = self._load_st()
        rec1 = orchestrator_guard._get_test_receipt(st1)
        self.assertTrue(rec1.get("dirty"))
        self.assertFalse(rec1.get("verified"))
        self.assertEqual(orchestrator_guard._get_circuit_breaker(st1).get("failure_streak"), 1)

        self._post("run_command", {"CommandLine": "pytest tests/"}, output="Command exited with code 2\nfatal error", call_id="post-call-c3-2")
        self.assertTrue(orchestrator_guard._get_circuit_breaker(self._load_st()).get("tripped"))
        self.assertEqual(self._pre("send_message", {"Recipient": "sub-worker", "Message": "try again"}).get("decision"), "deny")

        self._reg_sub("implementer", "sub-c3", "flash")
        blocked_msg = (
            "STATUS: BLOCKED\nFILES_CHANGED: NONE\nVERIFICATION: pytest tests/ exited with 1\n"
            f"NOTES: Tests failed repeatedly file://{self.artifact_dir}/scratch/fail.log\n"
        )
        self.assertEqual(
            self._pre("send_message", {"Recipient": self.base_payload["conversationId"], "Message": blocked_msg}, cid="sub-c3").get("decision"),
            "allow",
        )

    def test_c4_cross_brain_isolation_preserves_unrelated_active_sessions(self) -> None:
        base_dir = Path(self.artifact_dir)
        ws_a, ws_b = base_dir / "ws_a", base_dir / "ws_b"
        ws_a.mkdir(parents=True, exist_ok=True)
        ws_b.mkdir(parents=True, exist_ok=True)

        p_a = self._brain_payload("session-a", workspacePaths=[str(ws_a.resolve())])
        p_b = self._brain_payload("session-b", workspacePaths=[str(ws_b.resolve())])
        sub_p = self._brain_payload("sub-a", workspacePaths=[str(ws_a.resolve())], modelName="Gemini 4 Flash", invocationNum=1)

        self._seed_state(
            p_a,
            active=True,
            session_evaluated=True,
            root_conversation_id="session-a",
            workspace_roots=[str(ws_a.resolve())],
            test_receipt={"dirty": True, "verified": False},
            pending_spawns=[{"call_id": "spawn-a", "type": "implementer", "model": "flash"}],
        )
        self._seed_state(
            p_b,
            active=True,
            session_evaluated=True,
            root_conversation_id="session-b",
            workspace_roots=[str(ws_b.resolve())],
            test_receipt={"dirty": True, "verified": False},
            pending_spawns=[{"call_id": "spawn-b", "type": "implementer", "model": "flash"}],
        )

        self.assertTrue(orchestrator_guard._is_known_subagent(sub_p, self._load_st(sub_p)))
        self._post("run_command", {"CommandLine": "pytest tests/test_feature.py"}, output="1 passed in 0.02s", base=sub_p)

        rec_a = orchestrator_guard._get_test_receipt(self._load_st(p_a))
        self.assertFalse(rec_a.get("dirty"))
        self.assertTrue(rec_a.get("verified"))

        st_b_final = self._load_st(p_b)
        rec_b = orchestrator_guard._get_test_receipt(st_b_final)
        self.assertTrue(rec_b.get("dirty"))
        self.assertFalse(rec_b.get("verified"))
        self.assertEqual(len(st_b_final.get("pending_spawns", [])), 1)
        self.assertEqual(st_b_final["pending_spawns"][0].get("call_id"), "spawn-b")

    def test_h1_multiprocess_concurrent_state_transactions_and_corrupt_recovery(self) -> None:
        import subprocess

        worker_code = (
            "import sys; sys.path.insert(0, sys.argv[1]); import orchestrator_guard; "
            "orchestrator_guard.register_subagent({'artifactDirectoryPath': sys.argv[2], 'conversationId': 'root-mp'}, "
            "conversation_id=sys.argv[3], type_name='scout', model='flash_lite')"
        )
        procs = [
            subprocess.Popen([sys.executable, "-c", worker_code, str(SCRIPT_DIR), self.artifact_dir, f"worker-{i}"])
            for i in range(6)
        ]
        for p in procs:
            self.assertEqual(p.wait(), 0)

        final_state = self._load_st({"artifactDirectoryPath": self.artifact_dir, "conversationId": "root-mp"})
        self.assertEqual(len(final_state.get("subagents", {})), 6)

        with tempfile.TemporaryDirectory() as corrupt_dir:
            corrupt_payload = {"artifactDirectoryPath": corrupt_dir, "conversationId": "corrupt-cid"}
            orchestrator_guard._state_file_path(corrupt_payload).write_text("{corrupt", encoding="utf-8")
            recovered = self._load_st(corrupt_payload)
            self.assertTrue(recovered.get("active"))
            self.assertTrue(recovered.get("session_evaluated"))

    def test_h2_async_spawn_ack_keeps_dirty_receipt_until_verified(self) -> None:
        self._pre("write_file", {"Path": str(orchestrator_guard.PROJECT_ROOT / "mut_h2.py"), "Content": "y = 99\n"})
        self.assertTrue(orchestrator_guard._get_test_receipt(self._load_st()).get("dirty"))

        ack_output = 'Created the following subagent:\n{"conversationId": "12345678-1234-1234-1234-1234567890ab"}'
        self._post("invoke_subagent", {"TypeName": "implementer"}, output=ack_output)

        rec_post = orchestrator_guard._get_test_receipt(self._load_st())
        self.assertTrue(rec_post.get("dirty"))
        self.assertFalse(rec_post.get("verified"))

    def test_h3_deleted_test_file_diff_multireplace_and_block_comment_tampering(self) -> None:
        verdict = orchestrator_guard.audit_diff_text(
            "--- a/tests/test_old.py\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-def test_old():\n-    assert 1 == 1\n"
        )
        self.assertIsNotNone(verdict)
        self.assertIn("Diff audit detected", verdict or "")

        res_multi = self._pre(
            "multi_replace_file_content",
            {"TargetFile": "tests/test_x.py", "ReplacementChunks": [{"TargetContent": "def test_a():\n    assert a == 1\n", "ReplacementContent": "def test_a():\n    pass\n"}]},
        )
        self.assertEqual(res_multi.get("decision"), "deny")
        self.assertIn("Test tampering blocked", res_multi.get("reason", ""))

        res_docstring = self._pre(
            "edit_file",
            {"Path": "tests/test_x.py", "Target": "def test_b():\n    assert x == 1\n", "Replacement": 'def test_b():\n    """\n    assert x == 1\n    """\n'},
        )
        self.assertEqual(res_docstring.get("decision"), "deny")
        self.assertIn("Test tampering blocked", res_docstring.get("reason", ""))

        res_c_comment = self._pre(
            "edit_file",
            {"Path": "tests/test_x.py", "Target": "def test_b():\n    assert x == 1\n", "Replacement": "def test_b():\n    /* assert x == 1 */\n"},
        )
        self.assertEqual(res_c_comment.get("decision"), "deny")
        self.assertIn("Test tampering blocked", res_c_comment.get("reason", ""))

    def test_h4_h5_m3_subagent_send_message_validation_unsynced_allowlist_and_manage_subagents(self) -> None:
        parent_p = self._brain_payload("parent-h4")
        sub_p = self._brain_payload("sub-h4", modelName="Gemini 4 Flash")
        self._reg_sub("implementer", "sub-h4", "flash", payload=parent_p)

        res_bad = self._pre("send_message", {"Recipient": "parent-h4", "Message": "STATUS: COMPLETED\nFILES_CHANGED: NONE\n"}, base=sub_p)
        self.assertEqual(res_bad.get("decision"), "deny")
        self.assertIn("Subagent report format error", res_bad.get("reason", ""))

        good_report = "STATUS: COMPLETED\nFILES_CHANGED: NONE\nVERIFICATION: pytest tests/ passed\nNOTES: Finished all tasks cleanly.\n"
        self.assertEqual(self._pre("send_message", {"Recipient": "parent-h4", "Message": good_report}, base=sub_p).get("decision"), "allow")
        self.assertEqual(orchestrator_guard.handle_stop(sub_p).get("decision"), "allow")

        unsynced_p = self._brain_payload("unsynced-recipient")
        orchestrator_guard._save_state(unsynced_p, self._load_st(unsynced_p))
        msg_allowed = "TASK: Refactor module\nALLOWED_FILES: updated.py\nRULES:\n1. Open updated.py\nREPORT FORMAT:\nSTATUS: COMPLETED\n"
        self._pre("send_message", {"Recipient": "unsynced-recipient", "Message": msg_allowed}, base=parent_p)
        self.assertEqual(self._load_st(unsynced_p).get("self_subagent_meta", {}).get("allowed_files"), ["updated.py"])

        def_res = orchestrator_guard._handle_define_subagent({"Name": "implementer", "EnableMCPTools": True})
        self.assertEqual(def_res.get("decision"), "allow")
        self.assertEqual(def_res.get("overwrite"), {"EnableMCPTools": False})
        self.assertTrue(orchestrator_guard._is_simple_readonly_task("research", "1. Open a.py\n2. Read x\n3. Do not edit any file"))

        self._seed_state(parent_p, subagents={"branch-1": {"active_branch": True}, "branch-2": {"active_branch": True}})
        orchestrator_guard._handle_manage_subagents(parent_p, {"action": "kill", "conversation_ids": ["branch-1"]})
        st_m1 = self._load_st(parent_p)
        self.assertFalse(st_m1["subagents"]["branch-1"]["active_branch"])
        self.assertTrue(st_m1["subagents"]["branch-2"]["active_branch"])

        orchestrator_guard._handle_manage_subagents(parent_p, {"Action": "kill_all"})
        self.assertFalse(self._load_st(parent_p)["subagents"]["branch-2"]["active_branch"])

    def test_l1_cli_main_subprocess_modes_transcript_tail_and_git_diff_fallback(self) -> None:
        import subprocess

        tpath = Path(self.artifact_dir) / "transcript.jsonl"
        tpath.write_text(
            json.dumps({"type": "USER_MESSAGE", "content": "hello worker"}) + "\n"
            + json.dumps({"type": "PLANNER_RESPONSE", "content": "STATUS: COMPLETED\nFILES_CHANGED: NONE\nVERIFICATION: pass\nNOTES: done\n", "tool_calls": []}) + "\n",
            encoding="utf-8",
        )
        extracted = orchestrator_guard._extract_last_model_response({"transcriptPath": str(tpath)})
        self.assertIsNotNone(extracted)
        self.assertIn("STATUS: COMPLETED", extracted or "")
        self.assertIsInstance(orchestrator_guard._get_workspace_git_diff({"workspacePaths": [str(orchestrator_guard.PROJECT_ROOT)]}), str)

        entrypoint = SCRIPT_DIR / "orchestrator_guard.py"
        cli_modes = [
            ("doctor", {}),
            ("activate", self.base_payload),
            ("status", self.base_payload),
            ("pre-invocation", self.base_payload),
            ("pre-tool", self._payload(tool="read_file", args={"Path": str(orchestrator_guard.PROJECT_ROOT / "README.md")})),
            ("pre-tool", self._payload(tool="write_file", args={"Path": str(Path(self.artifact_dir) / "scratch.txt"), "Content": "test\n"})),
            ("pre-tool", self._payload(tool="run_command", args={"CommandLine": "git status"})),
            ("post-tool", self._payload(tool="run_command", args={"CommandLine": "git status"})),
            ("stop", self.base_payload),
            ("deactivate", self.base_payload),
            ("reset", self.base_payload),
        ]
        for mode, payload in cli_modes:
            res = subprocess.run([sys.executable, str(entrypoint), mode], input=json.dumps(payload), capture_output=True, text=True, check=True)
            self.assertEqual(res.returncode, 0)
            self.assertIsInstance(json.loads(res.stdout), dict)

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
                    stdin=r_fd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3, check=True,
                )
                self.assertEqual(res.returncode, 0)
                parsed = json.loads(res.stdout)
                self.assertIsInstance(parsed, dict)
                if mode == "status":
                    self.assertEqual(parsed.get("stateFile"), str(default_state_file))
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

        act_res = subprocess.run([sys.executable, str(entrypoint), "activate"], input=json.dumps(piped_payload), capture_output=True, text=True, timeout=3, check=True)
        self.assertEqual(json.loads(act_res.stdout).get("active"), True)

        status_res = subprocess.run([sys.executable, str(entrypoint), "status"], input=json.dumps(piped_payload), capture_output=True, text=True, timeout=3, check=True)
        status_data = json.loads(status_res.stdout)
        self.assertEqual(status_data.get("stateFile"), str(custom_dir / "orchestrator_state.json"))
        self.assertTrue(status_data.get("active"))

        stdout_buf = io.StringIO()
        with mock.patch.object(sys, "argv", ["orchestrator_guard.py", "status"]), mock.patch.object(
            sys, "stdin", io.StringIO(json.dumps(piped_payload))
        ), mock.patch.object(sys, "stdout", stdout_buf):
            guard_cli.main()
        inproc_status = json.loads(stdout_buf.getvalue())
        self.assertEqual(inproc_status.get("stateFile"), str(custom_dir / "orchestrator_state.json"))
        self.assertTrue(inproc_status.get("active"))

        with mock.patch.object(sys, "stdin", io.StringIO("[1, 2, 3]")):
            self.assertEqual(guard_cli._read_stdin_payload(is_cli_mode=True), {})

    def test_replan_status_and_root_replan_circuit_breaker(self) -> None:
        valid_replan = "STATUS: NEEDS_REPLAN\nFILES_CHANGED: NONE\nVERIFICATION: NONE\nNOTES: Core assumption about database schema was invalidated."
        self.assertIsNone(orchestrator_guard.validate_subagent_report(valid_replan))

        err = orchestrator_guard.validate_subagent_report("STATUS: NEEDS_REPLAN\nFILES_CHANGED: NONE\nVERIFICATION: NONE\nNOTES: invalid")
        self.assertIsNotNone(err)
        self.assertIn("NEEDS_REPLAN", err)
        self.assertIn("at least 3 words", err)

        sub_cid = "sub-cb-replan-111"
        sub_brain = Path(self.artifact_dir).parent / sub_cid
        sub_brain.mkdir(parents=True, exist_ok=True)
        self._reg_sub("implementer", sub_cid, "flash")
        state = self._load_st()
        cb = orchestrator_guard._get_circuit_breaker(state, self.base_payload)
        cb["tripped"], cb["failure_streak"], cb["last_error"] = True, 2, "Command exited with code 1"
        orchestrator_guard._save_state(self.base_payload, state)

        self.assertEqual(
            self._pre("send_message", {"Recipient": self.base_payload["conversationId"], "Message": valid_replan}, cid=sub_cid, artifactDirectoryPath=str(sub_brain))["decision"],
            "allow",
        )

        res_root_deny = self._pre(
            "invoke_subagent",
            {"TypeName": "implementer", "Prompt": "TASK: Fix bug\nRULES:\n1. Open foo.py\n2. Run pytest\nREPORT FORMAT:\nSTATUS: COMPLETED\nFILES_CHANGED: foo.py\nVERIFICATION: pytest ok\nNOTES: done"},
        )
        self.assertEqual(res_root_deny["decision"], "deny")
        self.assertIn("circuit breaker tripped", res_root_deny["reason"])
        self.assertIn("REPLAN:", res_root_deny["reason"])

        replan_args = {
            "TypeName": "implementer",
            "Prompt": "TASK: Fix bug with revised strategy\nREPLAN: Switched to migration approach\nRULES:\n1. Open foo.py\n2. Run pytest\nREPORT FORMAT:\nSTATUS: COMPLETED\nFILES_CHANGED: foo.py\nVERIFICATION: pytest ok\nNOTES: done",
        }
        self.assertEqual(self._pre("invoke_subagent", replan_args)["decision"], "allow")
        state_after = self._load_st()
        cb_after = orchestrator_guard._get_circuit_breaker(state_after, self.base_payload)
        self.assertFalse(cb_after["tripped"])
        self.assertEqual(cb_after["failure_streak"], 0)
        self.assertEqual(cb_after["replan_count"], 1)

        cb_after["tripped"], cb_after["failure_streak"] = True, 2
        orchestrator_guard._save_state(self.base_payload, state_after)

        res_root_deny2 = self._pre("invoke_subagent", replan_args)
        self.assertEqual(res_root_deny2["decision"], "deny")
        self.assertIn("circuit breaker tripped", res_root_deny2["reason"])

    def test_line_byte_read_budget_and_list_dir_exemption(self) -> None:
        file_paths = []
        for i in range(5):
            p = orchestrator_guard.PROJECT_ROOT / f"tmp_small_{i}.txt"
            p.write_text("line 1\nline 2\nline 3\nline 4\nline 5\n", encoding="utf-8")
            file_paths.append(str(p))

        try:
            for f in file_paths[:4]:
                self.assertEqual(self._pre("read_file", {"Path": f})["decision"], "allow")

            big_p = orchestrator_guard.PROJECT_ROOT / "tmp_big_sliced.txt"
            big_p.write_text("\n".join(f"line {j}" for j in range(1, 601)) + "\n", encoding="utf-8")
            big_path = str(big_p)
            file_paths.append(big_path)

            self.assertEqual(self._pre("read_file", {"Path": big_path, "StartLine": 1, "EndLine": 200})["decision"], "allow")
            res_slice2 = self._pre("read_file", {"Path": big_path, "StartLine": 201, "EndLine": 400})
            self.assertEqual(res_slice2["decision"], "deny")
            self.assertIn("Direct execution fast-path read budget", res_slice2["reason"])
            self.assertIn("400 lines", res_slice2["reason"])

            huge_p = orchestrator_guard.PROJECT_ROOT / "tmp_huge_bytes.txt"
            huge_p.write_text("A" * 20000 + "\n", encoding="utf-8")
            huge_path = str(huge_p)
            file_paths.append(huge_path)

            state = self._load_st()
            state["direct_budget"].clear()
            orchestrator_guard._save_state(self.base_payload, state)

            res_huge = self._pre("read_file", {"Path": huge_path})
            self.assertEqual(res_huge["decision"], "deny")
            self.assertIn("16384 bytes", res_huge["reason"])

            for _ in range(15):
                self.assertEqual(self._pre("list_dir", {"Path": str(orchestrator_guard.PROJECT_ROOT)})["decision"], "allow")

            res_ld_log = self._pre("list_dir", {"Path": str(Path(self.artifact_dir) / "scratch" / "run.log")})
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
        spec_path.write_text(
            "# Grounded Specification\n\n## SIGNATURES\ndef solve(a: int) -> int: ...\n\n"
            "## INVARIANTS_AND_PITFALLS\nKeep a > 0.\n\n## VERIFICATION\nRun pytest tests/test_solve.py\n",
            encoding="utf-8",
        )
        prompt_valid = (
            f"TASK: Implement solve function\nSPEC_FILE: {spec_path}\nRULES:\n1. Open spec\n2. Run pytest\n"
            "REPORT FORMAT:\nSTATUS: COMPLETED\nFILES_CHANGED: foo.py\nVERIFICATION: pytest ok\nNOTES: done"
        )
        self.assertEqual(self._pre("invoke_subagent", {"TypeName": "implementer", "Prompt": prompt_valid})["decision"], "allow")
        self.assertTrue(any(p.get("spec_file") == str(spec_path.resolve()) for p in self._load_st().get("pending_spawns", [])))

        missing_spec_prompt = (
            "TASK: Implement feature\nSPEC_FILE: /tmp/scratch/non_existent_spec_xyz.md\nRULES:\n1. Open spec\n2. Run pytest\n"
            "REPORT FORMAT:\nSTATUS: COMPLETED\nFILES_CHANGED: foo.py\nVERIFICATION: pytest ok\nNOTES: done"
        )
        res_missing = self._pre("invoke_subagent", {"TypeName": "implementer", "Prompt": missing_spec_prompt})
        self.assertEqual(res_missing["decision"], "deny")
        self.assertIn("does not exist", res_missing["reason"])

        stub_spec_path = Path(self.artifact_dir) / "scratch" / "spec_stub.md"
        stub_spec_path.write_text("# Incomplete Spec\nJust some notes without sections.\n", encoding="utf-8")
        stub_prompt = (
            f"TASK: Implement feature\nSPEC_FILE: {stub_spec_path}\nRULES:\n1. Open spec\n2. Run pytest\n"
            "REPORT FORMAT:\nSTATUS: COMPLETED\nFILES_CHANGED: foo.py\nVERIFICATION: pytest ok\nNOTES: done"
        )
        res_stub = self._pre("invoke_subagent", {"TypeName": "implementer", "Prompt": stub_prompt})
        self.assertEqual(res_stub["decision"], "deny")
        self.assertIn("missing required grounded sections", res_stub["reason"])

    def test_staged_checkpoints_and_checkpoint_status(self) -> None:
        state = self._load_st()
        receipt = orchestrator_guard._get_test_receipt(state, self.base_payload)
        receipt["dirty"], receipt["verified"] = True, False
        orchestrator_guard._save_state(self.base_payload, state)

        stage_prompt = (
            "TASK: Multi-stage refactor\nALLOWED_FILES: foo.py\nSTAGE: 1/2\nRULES:\n1. Edit foo.py\n2. Run tests\n"
            "REPORT FORMAT:\nSTATUS: COMPLETED\nFILES_CHANGED: foo.py\nVERIFICATION: pytest ok\nNOTES: done"
        )
        self.assertEqual(self._pre("invoke_subagent", {"TypeName": "implementer", "Prompt": stage_prompt})["decision"], "allow")
        self.assertTrue(any(p.get("stage") == "1/2" for p in self._load_st().get("pending_spawns", [])))

        checkpoint_report = "STATUS: CHECKPOINT\nFILES_CHANGED: foo.py\nVERIFICATION: pytest tests/test_foo.py passed\nNOTES: Completed stage 1; ready for stage 2."
        self.assertIsNone(orchestrator_guard.validate_subagent_report(checkpoint_report))

        self.assertEqual(self._post("invoke_subagent", {"TypeName": "implementer", "Prompt": stage_prompt}, output=checkpoint_report), {})
        receipt_post = orchestrator_guard._get_test_receipt(self._load_st(), self.base_payload)
        self.assertTrue(receipt_post.get("dirty"))
        self.assertFalse(receipt_post.get("verified"))

        sub_cid = "sub-staged-stage2-cid"
        (Path(self.artifact_dir).parent / sub_cid).mkdir(parents=True, exist_ok=True)
        self._reg_sub("implementer", sub_cid, "flash", allowed_files=["foo.py"])
        res_msg = self._pre(
            "send_message",
            {"Recipient": sub_cid, "Message": "STAGE: 2/2\nALLOWED_FILES: foo.py, bar.py\nPlease proceed with stage 2."},
        )
        self.assertEqual(res_msg["decision"], "allow")
        sub_info = self._load_st().get("subagents", {}).get(sub_cid, {})
        self.assertEqual(sub_info.get("stage"), "2/2")
        self.assertIn("2/2", sub_info.get("stage_history", []))
        self.assertEqual(sub_info.get("allowed_files"), ["foo.py", "bar.py"])

    def test_compact_scout_report_shorthand(self) -> None:
        compact_report = "STATUS: COMPLETED\nNOTES: Found symbol in foo.py"
        self.assertIsNone(orchestrator_guard.validate_subagent_report(compact_report, is_scout=True))
        err_impl = orchestrator_guard.validate_subagent_report(compact_report, is_scout=False)
        self.assertIsNotNone(err_impl)
        self.assertIn("Expected 4 header lines", err_impl)

        parent_p = self._brain_payload("parent-scout-test")
        scout_cid, impl_cid = "sub-scout-compact-99", "sub-impl-compact-fail"
        scout_p = self._brain_payload(scout_cid)
        impl_p = self._brain_payload(impl_cid)

        orchestrator_guard._cli_activate(parent_p)
        self._reg_sub("scout", scout_cid, "flash_lite", payload=parent_p)
        self._reg_sub("implementer", impl_cid, "flash", payload=parent_p)

        self.assertEqual(
            self._pre("send_message", {"Recipient": "parent-scout-test", "Message": compact_report}, base=scout_p)["decision"],
            "allow",
        )
        res_impl_msg = self._pre("send_message", {"Recipient": "parent-scout-test", "Message": compact_report}, base=impl_p)
        self.assertEqual(res_impl_msg["decision"], "deny")
        self.assertIn("Subagent report format error", res_impl_msg["reason"])
        self.assertEqual(self._stop(compact_report, base=scout_p).get("decision"), "allow")

    def test_system_orchestrator_exit_plan_mode_coexistence(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)
        plan_content = "TASK: Multi-file refactor\nALLOWED_FILES: file_x.py, file_y.py, file_z.py\n1. Edit all three files\n2. Run tests"
        self.assertEqual(self._pre("exit_plan_mode", {"plan": plan_content})["decision"], "allow")

        state = self._load_st()
        self.assertTrue(state.get("plan_execute_phase"))
        self.assertFalse(state.get("plan_execute_consumed"))
        self.assertEqual(state.get("plan_allowed_files"), ["file_x.py", "file_y.py", "file_z.py"])

        for fname in ("file_x.py", "file_y.py", "file_z.py"):
            self.assertEqual(
                self._pre("edit_file", {"Path": str(orchestrator_guard.PROJECT_ROOT / fname), "Target": "", "Replacement": ""})["decision"],
                "allow",
            )

        res_out = self._pre("edit_file", {"Path": str(orchestrator_guard.PROJECT_ROOT / "out_of_plan.py"), "Target": "", "Replacement": ""})
        self.assertEqual(res_out["decision"], "deny")
        self.assertIn("File allowlist violation", res_out["reason"])

        for cmd in ("echo 1", "echo 2", "echo 3", "echo 4"):
            self.assertEqual(self._pre("run_command", {"CommandLine": cmd})["decision"], "allow")

        res_log = self._pre("read_file", {"Path": str(Path(self.artifact_dir) / "scratch" / "test.log")})
        self.assertEqual(res_log["decision"], "deny")
        self.assertIn("Context separation violation", res_log["reason"])

        self._pre_inv(lastUserInput="Please do [plan review] now")
        state_after_review = self._load_st()
        self.assertFalse(state_after_review.get("plan_execute_phase"))
        self.assertIsNone(state_after_review.get("plan_allowed_files"))

    def test_safe_git_env_prefixes_and_unsafe_env_rejection(self) -> None:
        safe_cmds = (
            "GIT_EDITOR=true git status",
            "GIT_SEQUENCE_EDITOR=: git diff",
            "GIT_PAGER=cat git log",
            "LC_ALL=C GIT_TERMINAL_PROMPT=0 git status",
            "PAGER='' git diff",
            'LANG="en_US.UTF-8" GIT_EDITOR="/bin/true" git status',
            'GIT_PAGER="" git status',
        )
        for cmd in safe_cmds:
            self.assertTrue(orchestrator_guard._is_git_only_command(cmd), f"Expected safe command to be accepted: {cmd}")

        unsafe_cmds = (
            "GIT_SSH_COMMAND=evil git status",
            "LD_PRELOAD=/tmp/lib.so git status",
            "GIT_EDITOR=/tmp/evil.sh git status",
            "GIT_EDITOR=vim git status",
            "PATH=/tmp git status",
            "GIT_EXEC_PATH=/tmp git status",
        )
        for cmd in unsafe_cmds:
            self.assertFalse(orchestrator_guard._is_git_only_command(cmd), f"Expected unsafe command to be rejected: {cmd}")

        state = self._load_st()
        cb = orchestrator_guard._get_circuit_breaker(state)
        cb["failure_streak"], cb["tripped"], cb["last_error"] = 2, True, "exit 1"
        orchestrator_guard._save_state(self.base_payload, state)

        self.assertEqual(self._pre("run_command", self._cmd_args("GIT_EDITOR=true git status"))["decision"], "allow")
        res_unsafe = self._pre("run_command", self._cmd_args("GIT_SSH_COMMAND=evil git status"))
        self.assertEqual(res_unsafe["decision"], "deny")
        self.assertIn("2-turn circuit breaker tripped", res_unsafe["reason"])

        cb["tripped"], cb["failure_streak"] = False, 0
        orchestrator_guard._save_state(self.base_payload, state)

        spec_file = Path(self.artifact_dir) / "scratch" / "spec_underscores.md"
        spec_file.parent.mkdir(parents=True, exist_ok=True)
        spec_file.write_text(
            "# Grounded Spec with Underscores\n\n## SIGNATURES_AND_SNIPPETS\ndef foo() -> None: pass\n\n"
            "## INVARIANTS_AND_PITFALLS\n- invariant 1\n\n## VERIFICATION_EXPECTATIONS\n- run pytest\n"
        )
        prompt = (
            f"TASK: Implement underscores test.\nSPEC_FILE: {spec_file}\nALLOWED_FILES: tests/test_sample.py\n\n"
            "RULES:\n1. Open file.\n2. Edit.\n3. Run pytest.\n\n"
            "REPORT FORMAT:\nSTATUS: COMPLETED\nFILES_CHANGED: tests/test_sample.py\nVERIFICATION: pytest ok\nNOTES: done\n"
        )
        self.assertEqual(self._pre("invoke_subagent", {"TypeName": "implementer", "Prompt": prompt})["decision"], "allow")

    def test_sandbox_boundary_errors_exempt_from_circuit_breaker(self) -> None:
        state = self._load_st()
        cb = orchestrator_guard._get_circuit_breaker(state)
        cb["failure_streak"], cb["tripped"] = 0, False
        orchestrator_guard._save_state(self.base_payload, state)

        exempt_cases = (
            ("cmd_ro_lock", "git checkout -b branch1", "", "fatal: Unable to create '/path/.git/index.lock': Read-only file system", False),
            ("cmd_banner", "git push origin branch1", "There were sandbox errors that may or may not be related to the failure\nexit status 1", "Command exited with code 1", False),
            ("cmd_sberr", "npm test", "", "Process failed in sandbox", True),
            ("cmd_fetch_ro", "git fetch origin", "fatal: cannot open .git/FETCH_HEAD: Read-only file system\n", "Command exited with code 128", False),
        )
        for cid_call, cmd, out, err, sb_flag in exempt_cases:
            extra = {"sandboxError": True} if sb_flag else {}
            self._post("run_command", self._cmd_args(cmd), output=out, error=err, call_id=cid_call, **extra)
            cb_now = orchestrator_guard._get_circuit_breaker(self._load_st())
            self.assertEqual(cb_now.get("failure_streak", 0), 0)
            self.assertFalse(cb_now.get("tripped", False))

        self._post("run_command", self._cmd_args("pytest tests/"), error="exit status 1", call_id="cmd_real_1")
        cb_real_1 = orchestrator_guard._get_circuit_breaker(self._load_st())
        self.assertEqual(cb_real_1.get("failure_streak", 0), 1)
        self.assertFalse(cb_real_1.get("tripped", False))

        self._post("run_command", self._cmd_args("pytest tests/"), error="exit status 1", call_id="cmd_real_2")
        cb_real_2 = orchestrator_guard._get_circuit_breaker(self._load_st())
        self.assertEqual(cb_real_2.get("failure_streak", 0), 2)
        self.assertTrue(cb_real_2.get("tripped", False))

    def test_phase1_define_subagent_verifier_and_reviewer_permissions(self) -> None:
        for role_name in ("verifier", "reviewer"):
            res = self._pre("define_subagent", {"Name": role_name, "EnableWriteTools": True, "EnableMCPTools": True, "EnableSubagentTools": True})
            self.assertEqual(res["decision"], "allow")
            self.assertEqual(res["overwrite"], {"EnableSubagentTools": False, "EnableWriteTools": False, "EnableMCPTools": False})

            res_snake = self._pre("define_subagent", {"name": role_name, "enable_write_tools": True, "enable_mcp_tools": True, "enable_subagent_tools": True})
            self.assertEqual(res_snake["decision"], "allow")
            self.assertEqual(res_snake["overwrite"], {"enable_subagent_tools": False, "enable_write_tools": False, "enable_mcp_tools": False})

    def test_phase1_verifier_reviewer_model_policy_and_regressions(self) -> None:
        short_prompt = (
            "TASK: Inspect module changes.\nREVIEW_LENS: security\nRULES:\n1. Read changed files.\n2. Run git diff.\n"
            "REPORT FORMAT:\nSTATUS: COMPLETED\nFILES_CHANGED: NONE\nVERIFICATION: ok\nNOTES: done"
        )
        for tname in ("verifier", "reviewer"):
            res_omit = self._pre("invoke_subagent", {"TypeName": tname, "Prompt": short_prompt}, call_id=f"m_omit_{tname}")
            self.assertEqual(res_omit["decision"], "allow")
            self.assertEqual(res_omit["overwrite"]["Model"], "flash")

            self.assertEqual(self._pre("invoke_subagent", {"TypeName": tname, "Model": "flash", "Prompt": short_prompt.replace("security", f"flash-{tname}")}, call_id=f"m_flash_{tname}"), {"decision": "allow"})
            self.assertEqual(self._pre("invoke_subagent", {"TypeName": tname, "Model": "pro", "Prompt": short_prompt.replace("security", f"pro-{tname}")}, call_id=f"m_pro_{tname}"), {"decision": "allow"})
            self.assertEqual(self._pre("invoke_subagent", {"TypeName": tname, "Model": "inherit", "Prompt": short_prompt.replace("security", f"inh-{tname}")}, call_id=f"m_inh_{tname}"), {"decision": "allow"})

            res_lite = self._pre("invoke_subagent", {"TypeName": tname, "Model": "flash_lite", "Prompt": short_prompt.replace("security", f"lite-{tname}")}, call_id=f"m_lite_{tname}")
            self.assertEqual(res_lite["decision"], "allow")
            self.assertEqual(res_lite["overwrite"]["Model"], "flash")

        for std_type in ("scout", "research", "implementer"):
            res_std_pro = self._pre("invoke_subagent", {"TypeName": std_type, "Model": "pro", "Prompt": VALID_PROMPT}, call_id=f"std_pro_{std_type}")
            self.assertEqual(res_std_pro["decision"], "deny")
            self.assertIn("Model='pro'", res_std_pro["reason"])

            res_std_inh = self._pre("invoke_subagent", {"TypeName": std_type, "Model": "inherit", "Prompt": VALID_PROMPT}, call_id=f"std_inh_{std_type}")
            self.assertEqual(res_std_inh["decision"], "allow")
            self.assertEqual(res_std_inh["overwrite"]["Model"], "flash_lite" if std_type == "scout" else "flash")

    def test_phase1_verifier_reviewer_strict_readonly_and_dirty_receipt(self) -> None:
        parent_p = self._brain_payload("parent-ro-test")
        v_cid, r_cid = "sub-verifier-ro-01", "sub-reviewer-ro-02"
        v_p, r_p = self._brain_payload(v_cid), self._brain_payload(r_cid)
        orchestrator_guard._cli_activate(parent_p)
        self._reg_sub("verifier", v_cid, "flash", payload=parent_p, status="active")
        self._reg_sub("reviewer", r_cid, "pro", payload=parent_p, status="active", review_lens="security")

        target_code = str(orchestrator_guard.PROJECT_ROOT / "src" / "app.py")
        for sub_p in (v_p, r_p):
            for tool_name, tool_args in (
                ("write_file", {"Path": target_code, "Content": "x = 1"}),
                ("edit_file", {"Path": target_code, "Target": "a", "Replacement": "b"}),
                ("replace_file_content", {"TargetFile": target_code, "TargetContent": "a", "ReplacementContent": "b"}),
            ):
                res_mut = self._pre(tool_name, tool_args, base=sub_p)
                self.assertEqual(res_mut["decision"], "deny")
                self.assertIn("strictly read-only", res_mut["reason"])

            for bad_cmd in ("echo hi > out.txt", "rm -f foo.py", "sed -i 's/a/b/' foo.py", "touch new.py"):
                res_cmd = self._pre("run_command", self._cmd_args(bad_cmd), base=sub_p)
                self.assertEqual(res_cmd["decision"], "deny")
                self.assertIn("strictly read-only", res_cmd["reason"])

            self.assertEqual(self._pre("read_file", {"Path": str(orchestrator_guard.PROJECT_ROOT / "README.md"), "StartLine": 1, "EndLine": 10}, base=sub_p)["decision"], "allow")
            self.assertEqual(self._pre("run_command", self._cmd_args("git diff HEAD"), base=sub_p)["decision"], "allow")

        parent_state = self._load_st(parent_p)
        rec = orchestrator_guard._get_test_receipt(parent_state, parent_p)
        rec["dirty"], rec["verified"] = True, False
        orchestrator_guard._save_state(parent_p, parent_state)

        for ro_type in ("verifier", "reviewer"):
            self._post("invoke_subagent", {"TypeName": ro_type, "Role": ro_type.title()}, output="STATUS: COMPLETED\nFILES_CHANGED: NONE\nVERIFICATION: inspected\nNOTES: VERDICT: PASS", call_id=f"sync_{ro_type}", base=parent_p)
            rec_after = orchestrator_guard._get_test_receipt(self._load_st(parent_p), parent_p)
            self.assertTrue(rec_after["dirty"])
            self.assertFalse(rec_after["verified"])

    def test_phase1_verifier_reviewer_require_4_header_report(self) -> None:
        compact_report = "STATUS: COMPLETED\nNOTES: Looks good to me"
        valid_4_header = "STATUS: COMPLETED\nFILES_CHANGED: NONE\nVERIFICATION: Ran git diff and inspected tests\nNOTES: VERDICT: APPROVE"
        parent_p = self._brain_payload("parent-report-test")
        orchestrator_guard._cli_activate(parent_p)

        for idx, role_type in enumerate(("verifier", "reviewer"), start=1):
            sub_cid = f"sub-{role_type}-report-{idx}"
            sub_p = self._brain_payload(sub_cid)
            self._reg_sub(role_type, sub_cid, "flash", payload=parent_p, status="active", review_lens="security" if role_type == "reviewer" else None)

            res_msg = self._pre("send_message", {"Recipient": "parent-report-test", "Message": compact_report}, base=sub_p)
            self.assertEqual(res_msg["decision"], "deny")
            self.assertIn("Expected 4 header lines", res_msg["reason"])

            res_stop_bad = self._stop(compact_report, base=sub_p, executionNum=1)
            self.assertEqual(res_stop_bad["decision"], "continue")
            self.assertIn("Expected 4 header lines", res_stop_bad["reason"])

            self.assertEqual(self._stop(valid_4_header, base=sub_p, executionNum=2)["decision"], "allow")
            self.assertEqual(self._load_st(parent_p)["subagents"][sub_cid]["status"], "completed")

    def test_phase1_freshness_gate_mutation_epoch_root_and_implementer(self) -> None:
        parent_p = self._brain_payload("parent-epoch-test")
        impl_cid, verifier_cid, reviewer_cid = "sub-impl-epoch-01", "sub-verifier-epoch-02", "sub-reviewer-epoch-03"
        impl_p = self._brain_payload(impl_cid)
        self._brain_payload(verifier_cid)
        self._brain_payload(reviewer_cid)

        orchestrator_guard._cli_activate(parent_p)
        orchestrator_guard.handle_pre_invocation(parent_p)
        self._reg_sub("implementer", impl_cid, "flash", payload=parent_p, status="active")
        self._reg_sub("verifier", verifier_cid, "flash", payload=parent_p, status="active", spawned_mutation_epoch=0)
        self._reg_sub("reviewer", reviewer_cid, "pro", payload=parent_p, status="active", spawned_mutation_epoch=0, review_lens="security")

        for target_cid in (verifier_cid, reviewer_cid):
            self.assertEqual(self._pre("send_message", {"Recipient": target_cid, "Message": "Please also check input bounds."}, base=parent_p)["decision"], "allow")

        self.assertEqual(
            self._pre("edit_file", {"Path": str(orchestrator_guard.PROJECT_ROOT / "src_mod.py"), "Target": "a", "Replacement": "b"}, base=impl_p)["decision"],
            "allow",
        )
        self.assertEqual(orchestrator_guard._get_test_receipt(self._load_st(parent_p), parent_p).get("mutation_epoch"), 1)

        for target_cid, r_type in ((verifier_cid, "verifier"), (reviewer_cid, "reviewer")):
            res_stale = self._pre("send_message", {"Recipient": target_cid, "Message": "Re-verify after fix."}, base=parent_p)
            self.assertEqual(res_stale["decision"], "deny")
            self.assertIn(f"Stale {r_type}", res_stale["reason"])

        fresh_v = "sub-verifier-fresh-04"
        self._reg_sub("verifier", fresh_v, "flash", payload=parent_p, status="active", spawned_mutation_epoch=1)
        self.assertEqual(self._pre("send_message", {"Recipient": fresh_v, "Message": "Clarification at epoch 1."}, base=parent_p)["decision"], "allow")

        self.assertEqual(self._pre("write_file", {"Path": str(orchestrator_guard.PROJECT_ROOT / "root_mod.py"), "Content": "x = 2\n"}, base=parent_p)["decision"], "allow")
        self.assertEqual(orchestrator_guard._cli_status(parent_p)["testReceipt"]["mutationEpoch"], 2)

        res_stale_root = self._pre("send_message", {"Recipient": fresh_v, "Message": "Verify root edit."}, base=parent_p)
        self.assertEqual(res_stale_root["decision"], "deny")
        self.assertIn("Stale verifier", res_stale_root["reason"])

    def test_phase1_disjoint_implementer_lanes_and_lifecycle_release(self) -> None:
        def _impl_entry(allowed_line: str, workspace: str = "inherit", role: str = "Lane") -> dict:
            header = f"ALLOWED_FILES: {allowed_line}\n" if allowed_line else ""
            return self._sub_args(
                "implementer", role, "flash",
                f"TASK: Implement {role}.\n{header}RULES:\n1. Edit assigned files.\n2. Run tests.\n"
                "REPORT FORMAT:\nSTATUS: COMPLETED\nFILES_CHANGED: a.py\nVERIFICATION: ok\nNOTES: done",
                Workspace=workspace,
            )

        res_batch_overlap = self._pre("invoke_subagent", {"Subagents": [_impl_entry("web/", role="Lane Web"), _impl_entry("web/App.tsx", role="Lane App")]}, call_id="batch_overlap_1")
        self.assertEqual(res_batch_overlap["decision"], "deny")
        self.assertIn("Overlapping ALLOWED_FILES", res_batch_overlap["reason"])
        st_atomic = self._load_st()
        self.assertEqual(st_atomic.get("pending_spawns"), [])
        self.assertEqual(st_atomic.get("recorded_spawn_calls"), [])

        disjoint_subagents = {"Subagents": [_impl_entry("web/", role="Lane Web"), _impl_entry("api/", role="Lane API"), _impl_entry("docs/", role="Lane Docs")]}
        self.assertEqual(self._pre("invoke_subagent", disjoint_subagents, call_id="batch_disjoint_1")["decision"], "allow")

        res_pending_conflict = self._pre("invoke_subagent", _impl_entry("api/routes.py", role="Lane API Routes"), call_id="cli_conflict_1")
        self.assertEqual(res_pending_conflict["decision"], "deny")
        self.assertIn("Overlapping ALLOWED_FILES", res_pending_conflict["reason"])

        self._post("invoke_subagent", disjoint_subagents, output="STATUS: COMPLETED\nFILES_CHANGED: web/a.ts\nVERIFICATION: ok\nNOTES: done", call_id="batch_disjoint_1")
        self.assertEqual(self._pre("invoke_subagent", _impl_entry("api/routes.py", role="Lane API Routes"), call_id="cli_after_post_1")["decision"], "allow")

        parent_p = self._brain_payload("parent-lane-test")
        lane_a, lane_b, lane_c = "sub-lane-a-01", "sub-lane-b-02", "sub-lane-c-03"
        lane_a_p = self._brain_payload(lane_a)
        self._brain_payload(lane_b)
        orchestrator_guard._cli_activate(parent_p)
        self._reg_sub("implementer", lane_a, "flash", payload=parent_p, allowed_files=["src/a.py"], status="active")

        res_sub_overlap = self._pre("invoke_subagent", _impl_entry("src/a.py", role="Conflicting Worker"), call_id="sub_overlap_1", base=parent_p)
        self.assertEqual(res_sub_overlap["decision"], "deny")
        self.assertIn("Overlapping ALLOWED_FILES", res_sub_overlap["reason"])

        self._reg_sub("implementer", lane_b, "flash", payload=parent_p, workspace="branch", allowed_files=["src/shared.py"], status="active")
        self.assertEqual(self._pre("invoke_subagent", _impl_entry("src/shared.py", workspace="share", role="Share Worker"), call_id="sub_isolated_share", base=parent_p)["decision"], "allow")
        self.assertEqual(self._pre("invoke_subagent", _impl_entry("", role="Unrestricted Legacy Worker"), call_id="sub_omitted_allowed", base=parent_p)["decision"], "allow")

        self.assertEqual(self._stop("STATUS: COMPLETED\nFILES_CHANGED: src/a.py\nVERIFICATION: pytest ok\nNOTES: done", base=lane_a_p)["decision"], "allow")
        self.assertEqual(self._pre("invoke_subagent", _impl_entry("src/a.py", role="Follow-up Worker"), call_id="sub_reuse_after_stop", base=parent_p)["decision"], "allow")

        self._reg_sub("implementer", lane_c, "flash", payload=parent_p, allowed_files=["src/c.py"], status="active")
        res_reactivate_conflict = self._pre("send_message", {"Recipient": lane_a, "Message": "ALLOWED_FILES: src/c.py\nContinue work on c.py."}, base=parent_p)
        self.assertEqual(res_reactivate_conflict["decision"], "deny")
        self.assertIn("Overlapping ALLOWED_FILES", res_reactivate_conflict["reason"])

        self._pre("manage_subagents", {"Action": "kill", "ConversationIds": [lane_c]}, base=parent_p)
        self.assertEqual(self._pre("send_message", {"Recipient": lane_a, "Message": "ALLOWED_FILES: src/c.py\nContinue work on c.py."}, base=parent_p)["decision"], "allow")

    def test_phase1_reviewer_lens_enforcement_and_round2_release(self) -> None:
        def _rev_entry(lens_line: Optional[str], role: str = "Reviewer") -> dict:
            header = f"REVIEW_LENS: {lens_line}\n" if lens_line is not None else ""
            return self._sub_args(
                "reviewer", role, "flash",
                f"TASK: Review changes.\n{header}RULES:\n1. Inspect diff.\n2. Report findings.\n"
                "REPORT FORMAT:\nSTATUS: COMPLETED\nFILES_CHANGED: NONE\nVERIFICATION: git diff\nNOTES: VERDICT: APPROVE",
            )

        for bad_lens in (None, '""', "---"):
            res_missing = self._pre("invoke_subagent", _rev_entry(bad_lens), call_id=f"rev_bad_{bad_lens}")
            self.assertEqual(res_missing["decision"], "deny")
            self.assertIn("REVIEW_LENS", res_missing["reason"])

        for idx, blank_prompt in enumerate(("TASK: Review changes.\nREVIEW_LENS:\nRULES:\n1. Review.\n", "REVIEW_LENS:   \nTASK: Review changes.\nRULES:\n1. Review.\n")):
            res_blank = self._pre(
                "invoke_subagent",
                self._sub_args("reviewer", "Reviewer", "flash", f"{blank_prompt}REPORT FORMAT:\nSTATUS: COMPLETED\nFILES_CHANGED: NONE\nVERIFICATION: git diff\nNOTES: VERDICT: APPROVE"),
                call_id=f"rev_blank_{idx}",
            )
            self.assertEqual(res_blank["decision"], "deny")
            self.assertIn("REVIEW_LENS", res_blank["reason"])

        res_dup_batch = self._pre("invoke_subagent", {"Subagents": [_rev_entry("edge cases", "Reviewer 1"), _rev_entry("edge_cases", "Reviewer 2")]}, call_id="rev_dup_batch")
        self.assertEqual(res_dup_batch["decision"], "deny")
        self.assertIn("Duplicate REVIEW_LENS 'edge-cases'", res_dup_batch["reason"])
        self.assertEqual(self._load_st().get("pending_spawns"), [])

        res_distinct = self._pre(
            "invoke_subagent",
            {"Subagents": [_rev_entry("ux", "UX Reviewer"), _rev_entry("simplicity", "Simplicity Reviewer"), _rev_entry("edge-cases", "Edge Cases Reviewer"), _rev_entry("security", "Security Reviewer")]},
            call_id="rev_distinct_batch",
        )
        self.assertEqual(res_distinct["decision"], "allow")

        res_dup_pending = self._pre("invoke_subagent", _rev_entry("Edge Cases", "Dup Reviewer"), call_id="rev_dup_pending")
        self.assertEqual(res_dup_pending["decision"], "deny")
        self.assertIn("Duplicate REVIEW_LENS 'edge-cases'", res_dup_pending["reason"])

        parent_p = self._brain_payload("parent-rev-round2")
        rev1_cid = "sub-rev-sec-r1"
        rev1_p = self._brain_payload(rev1_cid)
        orchestrator_guard._cli_activate(parent_p)
        self._reg_sub("reviewer", rev1_cid, "pro", payload=parent_p, status="active", review_lens="security")

        status_r1 = orchestrator_guard._cli_status(parent_p)
        self.assertEqual(status_r1["activeSubagents"][rev1_cid]["status"], "active")
        self.assertEqual(status_r1["activeSubagents"][rev1_cid]["reviewLens"], "security")

        res_dup_active = self._pre("invoke_subagent", _rev_entry("security", "Round 2 Early"), call_id="rev_dup_active", base=parent_p)
        self.assertEqual(res_dup_active["decision"], "deny")
        self.assertIn("Duplicate REVIEW_LENS 'security'", res_dup_active["reason"])

        self.assertEqual(self._stop("STATUS: COMPLETED\nFILES_CHANGED: NONE\nVERIFICATION: git diff\nNOTES: VERDICT: REQUEST_CHANGES", base=rev1_p)["decision"], "allow")
        self.assertEqual(self._pre("invoke_subagent", _rev_entry("security", "Round 2 Fresh Security"), call_id="rev_r2_ok", base=parent_p)["decision"], "allow")

    def test_orchestrator_guard_failsafe_catches_import_indentation_error(self) -> None:
        import shutil
        import subprocess

        scripts_copy = Path(self.artifact_dir) / "scripts_copy"
        guard_copy = scripts_copy / "guard"
        guard_copy.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(orchestrator_guard.__file__).resolve(), scripts_copy / "orchestrator_guard.py")
        (guard_copy / "__init__.py").write_text("def broken():\n  pass\n   bad_indent = 1\n", encoding="utf-8")

        for mode in ("pre-tool", "stop", "post-tool"):
            proc = subprocess.run([sys.executable, str(scripts_copy / "orchestrator_guard.py"), mode], input="{}", text=True, capture_output=True, check=False)
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
        self.assertEqual(self._pre("exit_plan_mode", {"plan": plan_text}, lastUserInput="Implement feature")["decision"], "allow")
        st = self._load_st()
        self.assertTrue(st.get("plan_execute_phase"))
        self.assertEqual(st.get("plan_execute_user_input"), "Implement feature")
        self.assertFalse(st.get("plan_execute_turn_finished"))

        for inv_num in (2, 3):
            self._pre_inv(invocationNum=inv_num, lastUserInput="Implement feature")
            st_step = self._load_st()
            self.assertTrue(st_step.get("plan_execute_phase"))
            self.assertTrue(st_step.get("plan_execute_consumed"))
            self.assertEqual(st_step.get("plan_allowed_files"), ["file_a.py", "file_b.py", "file_c.py"])

        self._reg_sub("implementer", "sub-active-plan-1", "flash", status="active")
        self.assertEqual(self._stop()["decision"], "allow")
        self.assertFalse(self._load_st().get("plan_execute_turn_finished"))
        self._pre_inv(invocationNum=4, lastUserInput="Implement feature")
        self.assertTrue(self._load_st().get("plan_execute_phase"))

        st_pending = self._load_st()
        st_pending["subagents"]["sub-active-plan-1"]["status"] = "completed"
        st_pending["pending_spawns"] = [{"type": "implementer", "role": "worker", "status": "active"}]
        orchestrator_guard._save_state(self.base_payload, st_pending)
        self._stop()
        self.assertFalse(self._load_st().get("plan_execute_turn_finished"))
        self._pre_inv(invocationNum=5, lastUserInput="Implement feature")
        self.assertTrue(self._load_st().get("plan_execute_phase"))

        self._pre_inv(invocationNum=6, lastUserInput="Next task")
        st_new = self._load_st()
        self.assertFalse(st_new.get("plan_execute_phase"))
        self.assertIsNone(st_new.get("plan_allowed_files"))
        self.assertIsNone(st_new.get("plan_execute_user_input"))
        self.assertFalse(st_new.get("plan_execute_turn_finished"))

        self._seed_state(pending_spawns=[])
        self._pre("exit_plan_mode", {"plan": plan_text}, lastUserInput="Implement feature 2")
        self._pre_inv(invocationNum=7, lastUserInput="Implement feature 2")
        self._stop()
        self.assertTrue(self._load_st().get("plan_execute_turn_finished"))
        self._pre_inv(invocationNum=8, lastUserInput="Implement feature 2")
        st_after_done = self._load_st()
        self.assertFalse(st_after_done.get("plan_execute_phase"))
        self.assertIsNone(st_after_done.get("plan_allowed_files"))

        self._pre("exit_plan_mode", {"plan": plan_text}, lastUserInput="Implement feature 3")
        self._pre_inv(invocationNum=9, lastUserInput="Please run [plan review]")
        st_review = self._load_st()
        self.assertFalse(st_review.get("plan_execute_phase"))
        self.assertIsNone(st_review.get("plan_allowed_files"))

    def test_phase2_share_workspace_tracking_stop_gate_and_kill_cleanup(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)
        share_cid = "11111111-2222-3333-4444-555555555555"
        self._reg_sub("implementer", share_cid, "flash", workspace="share", role="Share Worker")
        st = self._load_st()
        self.assertFalse(st["subagents"][share_cid]["active_branch"])
        self.assertTrue(st["subagents"][share_cid]["active_worktree"])

        res_stop = self._stop()
        self.assertEqual(res_stop["decision"], "continue")
        self.assertIn("Uncleaned branched subagent worktrees remain", res_stop["reason"])
        self.assertIn(share_cid, res_stop["reason"])

        self._pre("manage_subagents", {"Action": "kill", "ConversationIds": [share_cid]})
        st_killed = self._load_st()
        self.assertFalse(st_killed["subagents"][share_cid]["active_worktree"])
        self.assertFalse(st_killed["subagents"][share_cid]["active_branch"])
        self.assertEqual(self._stop(executionNum=2)["decision"], "allow")

        spawn_res = self._pre("invoke_subagent", self._sub_args("implementer", "Share Pending", "flash", Workspace="share"), call_id="share_pending_1")
        self.assertEqual(spawn_res["decision"], "allow")
        st_pending = self._load_st()
        self.assertTrue(st_pending["pending_spawns"][0]["active_worktree"])
        self.assertFalse(st_pending["pending_spawns"][0]["active_branch"])

        res_stop_pending = self._stop(executionNum=3)
        self.assertEqual(res_stop_pending["decision"], "continue")
        self.assertIn("Share Pending", res_stop_pending["reason"])

        self._pre("manage_subagents", {"Action": "kill_all"})
        self.assertEqual(self._stop(executionNum=4)["decision"], "allow")

    def test_phase2_best_of_n_mode_cohort_workspace_model_and_atomic_batch(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)

        res_inherit = self._pre(
            "invoke_subagent",
            self._sub_args("implementer", "Candidate Inherit", prompt="MODE: BEST_OF_N\nALLOWED_FILES: src/a.py\n" + VALID_PROMPT, Workspace="inherit"),
            call_id="bon_inherit",
        )
        self.assertEqual(res_inherit["decision"], "deny")
        self.assertIn("MODE: BEST_OF_N requires isolated Workspace='branch' or Workspace='share'", res_inherit["reason"])

        res_non_bon_pro = self._pre("invoke_subagent", self._sub_args("implementer", "Standard Pro", "pro", Workspace="branch"), call_id="non_bon_pro")
        self.assertEqual(res_non_bon_pro["decision"], "deny")
        self.assertIn("Model='pro'", res_non_bon_pro["reason"])

        res_auto_cohort = self._pre(
            "invoke_subagent",
            {"Subagents": [
                self._sub_args("implementer", "Cand A", "pro", "MODE: BEST_OF_N\nALLOWED_FILES: src/a.py\n" + VALID_PROMPT, Workspace="branch"),
                self._sub_args("implementer", "Cand B", "inherit", "MODE: best-of-n\nALLOWED_FILES: src/a.py\n" + VALID_PROMPT, Workspace="share"),
            ]},
            call_id="bon_auto_batch",
            invocationNum=7,
        )
        self.assertEqual(res_auto_cohort["decision"], "allow")
        overwritten = res_auto_cohort.get("overwrite", {}).get("Subagents", [])
        self.assertEqual(overwritten[0]["Model"], "pro")
        self.assertEqual(overwritten[1]["Model"], "flash")

        st_auto = self._load_st()
        self.assertIn("best-of-n-7", st_auto["cohorts"])
        self.assertEqual(st_auto["cohorts"]["best-of-n-7"]["total_spawned"], 2)
        self.assertEqual(st_auto["pending_spawns"][0]["cohort"], "best-of-n-7")
        self.assertEqual(st_auto["pending_spawns"][0]["mode"], "best_of_n")
        self.assertTrue(st_auto["pending_spawns"][1]["active_worktree"])

        res_cohort_only = self._pre(
            "invoke_subagent",
            self._sub_args("implementer", "Cand C", prompt="COHORT: Cache_Fix\nALLOWED_FILES: src/a.py\n" + VALID_PROMPT, Workspace="share"),
            call_id="bon_cohort_only",
            invocationNum=8,
        )
        self.assertEqual(res_cohort_only["decision"], "allow")
        st_c = self._load_st()
        self.assertIn("cache-fix", st_c["cohorts"])
        self.assertEqual(st_c["cohorts"]["cache-fix"]["total_spawned"], 1)

        before_pending, before_calls = len(st_c["pending_spawns"]), len(st_c["recorded_spawn_calls"])
        res_atomic_fail = self._pre(
            "invoke_subagent",
            {"Subagents": [
                self._sub_args("implementer", "Good Entry", "pro", "COHORT: atomic-cohort\nEVAL_CMD: python3 scripts/eval.py\n" + VALID_PROMPT, Workspace="branch"),
                self._sub_args("implementer", "Bad Inherit Entry", prompt="COHORT: atomic-cohort\n" + VALID_PROMPT, Workspace="inherit"),
            ]},
            call_id="bon_atomic_fail",
            invocationNum=9,
        )
        self.assertEqual(res_atomic_fail["decision"], "deny")
        st_after_fail = self._load_st()
        self.assertNotIn("atomic-cohort", st_after_fail["cohorts"])
        self.assertEqual(len(st_after_fail["pending_spawns"]), before_pending)
        self.assertEqual(len(st_after_fail["recorded_spawn_calls"]), before_calls)
        self.assertIsNone(st_after_fail.get("active_eval_cmd"))

    def test_phase2_cohort_circuit_breaker_per_candidate_and_async_sync(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)

        for i in range(1, 6):
            res = self._pre("invoke_subagent", self._sub_args("implementer", f"Cand {i}", prompt=f"MODE: BEST_OF_N\nCOHORT: five-way\n{VALID_PROMPT}", Workspace="share"), call_id=f"five_{i}")
            self.assertEqual(res["decision"], "allow")

        self._post("invoke_subagent", {"TypeName": "implementer", "Prompt": "COHORT: five-way\n" + VALID_PROMPT}, error="candidate 1 crashed", call_id="five_1")
        self._post("invoke_subagent", {"TypeName": "implementer", "Prompt": "COHORT: five-way\n" + VALID_PROMPT}, output="STATUS: FAILED\nFILES_CHANGED: NONE\nVERIFICATION: pytest\nNOTES: failed tests see file:///tmp/fail.log", call_id="five_2")
        st_mid = self._load_st()
        self.assertFalse(st_mid["circuit_breaker"]["tripped"])
        self.assertEqual(st_mid["circuit_breaker"]["failure_streak"], 0)
        self.assertEqual(len(st_mid["cohorts"]["five-way"]["failed"]), 2)

        for i in range(1, 3):
            self._pre("invoke_subagent", self._sub_args("implementer", f"Two {i}", prompt=f"MODE: BEST_OF_N\nCOHORT: two-way\n{VALID_PROMPT}", Workspace="branch"), call_id=f"two_{i}")
        for i in range(1, 3):
            self._post("invoke_subagent", {"TypeName": "implementer", "Prompt": "COHORT: two-way\n" + VALID_PROMPT}, error=f"two_{i} failed", call_id=f"two_{i}")
        st_tripped = self._load_st()
        self.assertTrue(st_tripped["circuit_breaker"]["tripped"])
        self.assertGreaterEqual(st_tripped["circuit_breaker"]["failure_streak"], 2)
        self.assertEqual(len(st_tripped["cohorts"]["two-way"]["failed"]), 2)

        parent_p = self._brain_payload("parent-bon")
        cand1_p = self._brain_payload("cand-async-1", isSubagent=True)
        cand2_p = self._brain_payload("cand-async-2", isSubagent=True)
        orchestrator_guard._cli_activate(parent_p)
        orchestrator_guard.handle_pre_invocation(parent_p)
        for cid_idx, c_cid in enumerate(("cand-async-1", "cand-async-2"), start=1):
            self._reg_sub("implementer", c_cid, "flash", payload=parent_p, workspace="share", role=f"Async {cid_idx}", status="active", mode="best_of_n", cohort="async-cohort")
        p_state = self._load_st(parent_p)
        p_state["cohorts"]["async-cohort"] = {"total_spawned": 2, "succeeded": [], "failed": [], "created_at": 100.0}
        orchestrator_guard._save_state(parent_p, p_state)

        for step in (1, 2):
            self._post("run_command", {"CommandLine": "pytest tests/missing.py"}, exit_code=1, call_id=f"c1_cmd_{step}", base=cand1_p)
        self.assertTrue(self._load_st(cand1_p)["circuit_breaker"]["tripped"])

        res_c1_stop = self._stop("STATUS: BLOCKED\nFILES_CHANGED: NONE\nVERIFICATION: pytest failed\nNOTES: Candidate failed twice see file:///tmp/c1.log", base=cand1_p)
        self.assertEqual(res_c1_stop["decision"], "allow")
        p_after_c1 = self._load_st(parent_p)
        self.assertEqual(p_after_c1["cohorts"]["async-cohort"]["failed"], ["cand-async-1"])
        self.assertFalse(p_after_c1["circuit_breaker"]["tripped"])

        res_c2_stop = self._stop("STATUS: COMPLETED\nFILES_CHANGED: src/a.py\nVERIFICATION: pytest passed\nNOTES: SCORE: 1.0", base=cand2_p)
        self.assertEqual(res_c2_stop["decision"], "allow")
        p_after_c2 = self._load_st(parent_p)
        self.assertEqual(p_after_c2["cohorts"]["async-cohort"]["succeeded"], ["cand-async-2"])
        self.assertFalse(p_after_c2["circuit_breaker"]["tripped"])

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
        iso_app = Path(self.temp_dir.name) / "iso_app"
        wt_dir = iso_app / "worktrees" / "parent-iso" / "w1"
        (wt_dir / "src").mkdir(parents=True, exist_ok=True)
        parent_p = self._brain_payload("parent-iso", root="iso_app/brain")
        cand_p = self._brain_payload("cand-iso", root="iso_app/brain", workspacePaths=[str(orchestrator_guard.PROJECT_ROOT), str(wt_dir)], isSubagent=True)

        orchestrator_guard._cli_activate(parent_p)
        orchestrator_guard.handle_pre_invocation(parent_p)
        self._seed_state(parent_p, test_receipt={"dirty": True, "verified": False, "last_mutated_file": "src/main.py", "last_test_cmd": "", "mutation_epoch": 4})
        self._reg_sub("implementer", "cand-iso", "flash", payload=parent_p, workspace="branch", role="Iso Candidate", allowed_files=["src/example_iso.py"], status="active", mode="best_of_n", cohort="iso-cohort")

        self.assertEqual(self._pre("edit_file", {"Path": str(orchestrator_guard.PROJECT_ROOT / "src" / "example_iso.py"), "Target": "a", "Replacement": "b"}, call_id="cand_edit_1", base=cand_p)["decision"], "allow")
        self.assertEqual(self._pre("edit_file", {"Path": str(wt_dir / "src" / "example_iso.py"), "Target": "a", "Replacement": "b"}, call_id="cand_wt_allow", base=cand_p)["decision"], "allow")
        res_wt_deny = self._pre("edit_file", {"Path": str(wt_dir / "src" / "outside_allowlist.py"), "Target": "a", "Replacement": "b"}, call_id="cand_wt_deny", base=cand_p)
        self.assertEqual(res_wt_deny["decision"], "deny")
        self.assertIn("File allowlist violation", res_wt_deny["reason"])

        p_after_edit = self._load_st(parent_p)
        self.assertEqual(p_after_edit["test_receipt"]["mutation_epoch"], 4)
        self.assertEqual(p_after_edit["test_receipt"]["last_mutated_file"], "src/main.py")

        self._post("run_command", {"CommandLine": "python3 -m unittest discover -s tests"}, exit_code=0, call_id="cand_test_1", base=cand_p)
        c_after_test = self._load_st(cand_p)
        self.assertFalse(c_after_test["test_receipt"]["dirty"])
        self.assertTrue(c_after_test["test_receipt"]["verified"])
        p_after_cand_test = self._load_st(parent_p)
        self.assertTrue(p_after_cand_test["test_receipt"]["dirty"])
        self.assertFalse(p_after_cand_test["test_receipt"]["verified"])

        self._post(
            "invoke_subagent",
            {"TypeName": "implementer", "Workspace": "branch", "Prompt": "MODE: BEST_OF_N\nCOHORT: iso-cohort\n" + VALID_PROMPT},
            output="STATUS: COMPLETED\nFILES_CHANGED: src/main.py\nVERIFICATION: pytest ok\nNOTES: SCORE: 1.0",
            call_id="sync_bon_1",
            base=parent_p,
        )
        p_after_sync = self._load_st(parent_p)
        self.assertTrue(p_after_sync["test_receipt"]["dirty"])
        self.assertFalse(p_after_sync["test_receipt"]["verified"])

    def test_phase2_eval_cmd_verification_and_anti_tampering(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)

        for bad_cmd in ("true", ":", "echo pass", "printf ok", "exit 0", "pytest || true", "VAR=1 :"):
            res_bad_spawn = self._pre("invoke_subagent", {"TypeName": "implementer", "Workspace": "share", "Prompt": f"MODE: BEST_OF_N\nEVAL_CMD: {bad_cmd}\n{VALID_PROMPT}"}, call_id=f"bad_{bad_cmd}")
            self.assertEqual(res_bad_spawn["decision"], "deny", f"Expected deny for {bad_cmd}")
            self.assertIn("Trivial or no-op EVAL_CMD rejected", res_bad_spawn["reason"])

            res_bad_plan = self._pre("exit_plan_mode", {"plan": f"EVAL_CMD: {bad_cmd}\n1. Run eval"})
            self.assertEqual(res_bad_plan["decision"], "deny", f"Expected plan deny for {bad_cmd}")

        res_good_plan = self._pre("exit_plan_mode", {"plan": "EVAL_CMD: python3 scripts/bench.py\n1. Optimize"}, lastUserInput="Optimize solver")
        self.assertEqual(res_good_plan["decision"], "allow")
        st_plan = self._load_st()
        self.assertEqual(st_plan["active_eval_cmd"], "python3 scripts/bench.py")

        st_plan["test_receipt"]["dirty"], st_plan["test_receipt"]["verified"] = True, False
        orchestrator_guard._save_state(self.base_payload, st_plan)

        for bypass_cmd in ("python3 scripts/bench.py || true", "$(python3 scripts/bench.py)", "`python3 scripts/bench.py`", "python3 scripts/bench.py_fake"):
            self.assertFalse(orchestrator_guard._is_declared_eval_command(bypass_cmd, st_plan, self.base_payload), f"Bypass should not match: {bypass_cmd}")

        self._post("run_command", {"CommandLine": "PYTHONPATH=. python3 scripts/bench.py --fast"}, exit_code=0, call_id="eval_run_1")
        st_verified = self._load_st()
        self.assertFalse(st_verified["test_receipt"]["dirty"])
        self.assertTrue(st_verified["test_receipt"]["verified"])
        self.assertEqual(st_verified["test_receipt"]["last_test_cmd"], "PYTHONPATH=. python3 scripts/bench.py --fast")

        self._pre("enter_plan_mode", {})
        self.assertIsNone(self._load_st()["active_eval_cmd"])

        self._pre("exit_plan_mode", {"plan": "EVAL_CMD: python3 scripts/bench.py\n1. Optimize"}, lastUserInput="Optimize solver 2")
        self._pre_inv(invocationNum=2, lastUserInput="Please run [plan review]")
        self.assertIsNone(self._load_st()["active_eval_cmd"])

        st_direct = self._load_st()
        st_direct["active_eval_cmd"] = "python3 scripts/bench.py"
        st_direct["direct_budget"]["3"] = {"mutated_files": ["src/solver.py"], "read_files": [], "read_ranges": [], "read_lines": 0, "read_bytes": 0, "commands": [], "seen_calls": [], "test_runs_after_edit": 0}
        orchestrator_guard._save_state(self.base_payload, st_direct)

        self.assertIsNone(orchestrator_guard._check_and_record_direct_budget(self._payload(invocationNum=3), "run_command", {"CommandLine": "python3 scripts/bench.py"}, self._load_st()))
        res_eval_2 = orchestrator_guard._check_and_record_direct_budget(self._payload(invocationNum=3), "run_command", {"CommandLine": "python3 scripts/bench.py"}, self._load_st())
        self.assertIsNotNone(res_eval_2)
        self.assertEqual(res_eval_2["decision"], "deny")

    def test_phase2_cli_status_surfaces_cohorts_and_active_eval_cmd(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)
        res_spawn = self._pre(
            "invoke_subagent",
            self._sub_args("implementer", "Status Cand", prompt="MODE: BEST_OF_N\nCOHORT: status-cohort\nEVAL_CMD: python3 scripts/eval_score.py\n" + VALID_PROMPT, Workspace="share"),
            call_id="status_bon_1",
            invocationNum=4,
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
        prompt_no_id = f"MODE: BEST_OF_N\nCOHORT: no-id-cohort\n{VALID_PROMPT}"

        for i in range(1, 4):
            self.assertEqual(self._pre("invoke_subagent", self._sub_args("implementer", f"Cand {i}", prompt=prompt_no_id, Workspace="share"))["decision"], "allow")

        st_spawned = self._load_st()
        self.assertEqual(st_spawned["cohorts"]["no-id-cohort"]["total_spawned"], 3)
        self.assertEqual([p.get("candidate_id") for p in st_spawned["pending_spawns"]], ["no-id-cohort#1", "no-id-cohort#2", "no-id-cohort#3"])

        self._post("invoke_subagent", {"TypeName": "implementer", "Role": "Cand 2", "Prompt": prompt_no_id}, error="cand 2 crashed")
        st_after_fail = self._load_st()
        self.assertEqual(st_after_fail["cohorts"]["no-id-cohort"]["failed"], ["no-id-cohort#2"])
        self.assertFalse(st_after_fail["circuit_breaker"]["tripped"])
        self.assertEqual([p.get("candidate_id") for p in st_after_fail["pending_spawns"]], ["no-id-cohort#1", "no-id-cohort#3"])

        self._post("invoke_subagent", {"TypeName": "implementer", "Role": "Cand 1", "Prompt": prompt_no_id}, output="STATUS: COMPLETED\nFILES_CHANGED: src/a.py\nVERIFICATION: pytest ok\nNOTES: SCORE: 1.0")
        st_after_ok = self._load_st()
        self.assertEqual(st_after_ok["cohorts"]["no-id-cohort"]["succeeded"], ["no-id-cohort#1"])
        self.assertEqual({p["candidate_id"]: p["status"] for p in st_after_ok["pending_spawns"]}, {"no-id-cohort#1": "completed", "no-id-cohort#3": "active"})

        parent_p = self._brain_payload("parent-dedup", root="dedup_app/brain")
        cand_p = self._brain_payload("cand-dedup-1", root="dedup_app/brain", isSubagent=True)
        orchestrator_guard._cli_activate(parent_p)
        orchestrator_guard.handle_pre_invocation(parent_p)
        self._reg_sub("implementer", "cand-dedup-1", "flash", payload=parent_p, workspace="share", role="Cand 1", status="active", mode="best_of_n", cohort="dedup-cohort", candidate_id="dedup-cohort#1")
        p_st = self._load_st(parent_p)
        p_st["cohorts"]["dedup-cohort"] = {"total_spawned": 2, "succeeded": ["dedup-cohort#1"], "failed": [], "created_at": 100.0}
        orchestrator_guard._save_state(parent_p, p_st)

        self.assertEqual(self._stop("STATUS: COMPLETED\nFILES_CHANGED: src/a.py\nVERIFICATION: pytest ok\nNOTES: SCORE: 1.0", base=cand_p)["decision"], "allow")
        self.assertEqual(self._load_st(parent_p)["cohorts"]["dedup-cohort"]["succeeded"], ["dedup-cohort#1"])

    def test_phase2_run_workflow_guard_integration(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)

        self.assertEqual(self._pre("run_workflow", {"Cwd": "/etc", "Description": "Outside cwd"}, call_id="wf_bad_cwd")["decision"], "ask")
        self.assertEqual(self._pre("run_workflow", {"Cwd": str(orchestrator_guard.PROJECT_ROOT), "ScriptPath": "/etc/passwd", "Description": "Outside script"}, call_id="wf_bad_script")["decision"], "ask")

        for bad_eval in ("true", "pytest || true"):
            res_bad_eval = self._pre("run_workflow", {"Cwd": str(orchestrator_guard.PROJECT_ROOT), "Description": f"EVAL_CMD: {bad_eval}\nRun workflow"}, call_id=f"wf_bad_eval_{bad_eval}")
            self.assertEqual(res_bad_eval["decision"], "deny")
            self.assertIn("Trivial or no-op EVAL_CMD rejected", res_bad_eval["reason"])

        res_good_wf = self._pre(
            "run_workflow",
            {"Cwd": str(orchestrator_guard.PROJECT_ROOT), "ScriptPath": str(orchestrator_guard.PROJECT_ROOT / "scripts/wf.sh"), "Description": "EVAL_CMD: python3 scripts/eval_wf.py\nRun eval workflow"},
            call_id="wf_good_1",
        )
        self.assertEqual(res_good_wf["decision"], "allow")
        st_wf = self._load_st()
        self.assertEqual(st_wf["active_eval_cmd"], "python3 scripts/eval_wf.py")

        st_wf["test_receipt"]["dirty"], st_wf["test_receipt"]["verified"] = True, False
        orchestrator_guard._save_state(self.base_payload, st_wf)
        self._post("run_workflow", {}, exit_code=0, call_id="wf_good_1")
        st_after_ok = self._load_st()
        self.assertTrue(st_after_ok["test_receipt"]["dirty"])
        self.assertFalse(st_after_ok["test_receipt"]["verified"])

        self._post("run_workflow", {}, error="There were sandbox errors that may or may not be related to the failure", call_id="wf_sandbox")
        self.assertEqual(self._load_st()["circuit_breaker"]["failure_streak"], 0)

        self._post("run_workflow", {}, error="workflow step 1 failed", call_id="wf_fail_1")
        self._post("run_workflow", {}, exit_code=1, call_id="wf_fail_2")
        st_tripped = self._load_st()
        self.assertEqual(st_tripped["circuit_breaker"]["failure_streak"], 2)
        self.assertTrue(st_tripped["circuit_breaker"]["tripped"])

        res_blocked = self._pre("run_workflow", {"Cwd": str(orchestrator_guard.PROJECT_ROOT), "Description": "Retry workflow"}, call_id="wf_blocked")
        self.assertEqual(res_blocked["decision"], "deny")
        self.assertIn("REPLAN:", res_blocked["reason"])

        res_replan = self._pre("run_workflow", {"Cwd": str(orchestrator_guard.PROJECT_ROOT), "Description": "REPLAN: switch to fallback workflow\nRun v2"}, call_id="wf_replan")
        self.assertEqual(res_replan["decision"], "allow")
        st_replanned = self._load_st()
        self.assertFalse(st_replanned["circuit_breaker"]["tripped"])
        self.assertEqual(st_replanned["circuit_breaker"]["replan_count"], 1)

        self._post("run_workflow", {}, error="transient wf failure", call_id="wf_fail_3")
        self.assertEqual(self._load_st()["circuit_breaker"]["failure_streak"], 1)
        self._post("run_workflow", {}, exit_code=0, call_id="wf_ok_reset")
        st_reset = self._load_st()
        self.assertEqual(st_reset["circuit_breaker"]["failure_streak"], 0)
        self.assertFalse(st_reset["circuit_breaker"]["tripped"])
        self.assertTrue(st_reset["test_receipt"]["dirty"])

    def test_phase2_empty_call_id_out_of_order_prompt_hash_attribution(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)

        prompts = [f"MODE: BEST_OF_N\nCOHORT: hash-cohort\n{VALID_PROMPT}\n# approach {tag}" for tag in ("A", "B", "C")]
        for prompt in prompts:
            self.assertEqual(self._pre("invoke_subagent", self._sub_args("implementer", "Candidate", prompt=prompt, Workspace="share"))["decision"], "allow")

        st_spawned = self._load_st()
        self.assertEqual([p.get("candidate_id") for p in st_spawned["pending_spawns"]], ["hash-cohort#1", "hash-cohort#2", "hash-cohort#3"])
        self.assertEqual(len({p.get("prompt_hash") for p in st_spawned["pending_spawns"] if p.get("prompt_hash")}), 3)

        self._post("invoke_subagent", {"TypeName": "implementer", "Role": "Candidate", "Prompt": prompts[1]}, error="candidate B failed")
        st_after_fail = self._load_st()
        self.assertEqual(st_after_fail["cohorts"]["hash-cohort"]["failed"], ["hash-cohort#2"])
        self.assertEqual([p.get("candidate_id") for p in st_after_fail["pending_spawns"]], ["hash-cohort#1", "hash-cohort#3"])

        self._post("invoke_subagent", {"TypeName": "implementer", "Role": "Candidate", "Prompt": prompts[2]}, output="STATUS: COMPLETED\nFILES_CHANGED: src/auth/parser.py\nVERIFICATION: pytest ok\nNOTES: SCORE: 0.9")
        st_after_c = self._load_st()
        self.assertEqual(st_after_c["cohorts"]["hash-cohort"]["succeeded"], ["hash-cohort#3"])
        self.assertEqual({p["candidate_id"]: p["status"] for p in st_after_c["pending_spawns"]}, {"hash-cohort#1": "active", "hash-cohort#3": "completed"})

        self._post("invoke_subagent", {"TypeName": "implementer", "Role": "Candidate", "Prompt": prompts[0]}, output="STATUS: COMPLETED\nFILES_CHANGED: src/auth/parser.py\nVERIFICATION: pytest ok\nNOTES: SCORE: 1.0")
        st_after_a = self._load_st()
        self.assertEqual(st_after_a["cohorts"]["hash-cohort"]["succeeded"], ["hash-cohort#3", "hash-cohort#1"])
        self.assertEqual({p["candidate_id"]: p["status"] for p in st_after_a["pending_spawns"]}, {"hash-cohort#1": "completed", "hash-cohort#3": "completed"})

    def test_phase2_run_workflow_script_eval_cmd_and_traceback_detection(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)

        res_bad = self._pre(
            "run_workflow",
            {"Cwd": str(orchestrator_guard.PROJECT_ROOT), "Description": "Run eval script", "Script": "#!/usr/bin/env bash\n# EVAL_CMD: true\necho hi\n"},
            call_id="wf_script_bad",
        )
        self.assertEqual(res_bad["decision"], "deny")
        self.assertIn("Trivial or no-op EVAL_CMD rejected", res_bad["reason"])

        res_good = self._pre(
            "run_workflow",
            {"Cwd": str(orchestrator_guard.PROJECT_ROOT), "Description": "Run benchmark script", "Script": "#!/usr/bin/env bash\n# EVAL_CMD: python3 evals/bench.py\npython3 evals/bench.py\n"},
            call_id="wf_script_good",
        )
        self.assertEqual(res_good["decision"], "allow")
        self.assertEqual(self._load_st()["active_eval_cmd"], "python3 evals/bench.py")

        self._post("run_workflow", {}, output="Traceback (most recent call last):\nRuntimeError: boom", call_id="wf_tb_1")
        st_tb_1 = self._load_st()
        self.assertEqual(st_tb_1["circuit_breaker"]["failure_streak"], 1)
        self.assertFalse(st_tb_1["circuit_breaker"]["tripped"])

        self._post("run_workflow", {}, output="Traceback (most recent call last):\nRuntimeError: boom", call_id="wf_tb_2")
        st_tb_2 = self._load_st()
        self.assertEqual(st_tb_2["circuit_breaker"]["failure_streak"], 2)
        self.assertTrue(st_tb_2["circuit_breaker"]["tripped"])
        self.assertIsNotNone(st_tb_2["circuit_breaker"].get("last_error"))

    def test_phase3_judge_define_subagent_and_model_policy(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)

        res_pascal = self._pre("define_subagent", {"Name": "judge", "EnableWriteTools": True, "EnableSubagentTools": True, "EnableMCPTools": True})
        self.assertEqual(res_pascal["decision"], "allow")
        self.assertEqual(res_pascal["overwrite"], {"EnableSubagentTools": False, "EnableWriteTools": False, "EnableMCPTools": False})

        res_snake = self._pre("define_subagent", {"name": "judge", "enable_write_tools": True, "enable_subagent_tools": True, "enable_mcp_tools": True})
        self.assertEqual(res_snake["decision"], "allow")
        self.assertEqual(res_snake["overwrite"], {"enable_subagent_tools": False, "enable_write_tools": False, "enable_mcp_tools": False})

        res_council_def = self._pre("define_subagent", {"Name": "council", "EnableWriteTools": True, "EnableSubagentTools": True, "EnableMCPTools": True})
        self.assertEqual(res_council_def["decision"], "allow")
        self.assertEqual(res_council_def["overwrite"], {"EnableSubagentTools": False, "EnableMCPTools": False})

        judge_prompt = f"MODE: COUNCIL\nCOUNCIL_ID: c-policy\nCOUNCIL_ROLE: judge\n{VALID_PROMPT}"
        for allowed_model in ("pro", "inherit", "flash"):
            res_ok = self._pre("invoke_subagent", self._sub_args("judge", f"Judge-{allowed_model}", allowed_model, judge_prompt), call_id=f"judge_{allowed_model}")
            self.assertEqual(res_ok["decision"], "allow")
            self.assertNotIn("overwrite", res_ok)

        res_lite = self._pre("invoke_subagent", self._sub_args("judge", "Judge-lite", "flash_lite", judge_prompt), call_id="judge_lite")
        self.assertEqual(res_lite["decision"], "allow")
        self.assertEqual(res_lite["overwrite"]["Model"], "flash")

        res_omitted = self._pre("invoke_subagent", self._sub_args("judge", "Judge-default", prompt=judge_prompt), call_id="judge_omitted")
        self.assertEqual(res_omitted["decision"], "allow")
        self.assertEqual(res_omitted["overwrite"]["Model"], "flash")

    def test_phase3_judge_strict_readonly_report_receipt_and_freshness(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)
        st = self._load_st()
        receipt = orchestrator_guard._get_test_receipt(st, self.base_payload)
        receipt["dirty"], receipt["verified"], receipt["mutation_epoch"] = True, False, 1
        orchestrator_guard._save_state(self.base_payload, st)

        judge_cid = "sub-judge-readonly-1"
        self._reg_sub("judge", judge_cid, "pro", spawned_mutation_epoch=1, mode="council", council_id="c-1", council_role="judge")
        judge_payload = self._payload(cid=judge_cid)

        res_write = self._pre("write_file", {"Path": str(Path(self.artifact_dir) / "scratch" / "vote.md"), "Content": "x"}, base=judge_payload)
        self.assertEqual(res_write["decision"], "deny")
        self.assertIn("strictly read-only", res_write["reason"])

        res_shell = self._pre("run_command", {"CommandLine": "echo vote > vote.txt"}, base=judge_payload)
        self.assertEqual(res_shell["decision"], "deny")
        self.assertIn("strictly read-only", res_shell["reason"])

        self.assertFalse(orchestrator_guard._is_scout_subagent(judge_payload, self._load_st(judge_payload)))
        res_stop_bad = self._stop("STATUS: COMPLETED\nNOTES: Voted for Proposal A.", base=judge_payload)
        self.assertEqual(res_stop_bad["decision"], "continue")
        self.assertIn("4-header REPORT FORMAT", res_stop_bad["reason"])

        self._post("invoke_subagent", self._sub_args("judge", "Judge", "pro"), output="STATUS: COMPLETED\nFILES_CHANGED: NONE\nVERIFICATION: n/a\nNOTES: Winner is Proposal A.", call_id="inv_judge_ok")
        st_after_judge = self._load_st()
        self.assertTrue(st_after_judge["test_receipt"]["dirty"])
        self.assertFalse(st_after_judge["test_receipt"]["verified"])

        st_after_judge["test_receipt"]["mutation_epoch"] = 2
        orchestrator_guard._save_state(self.base_payload, st_after_judge)
        res_stale = self._pre("send_message", {"Recipient": judge_cid, "Message": "Re-evaluate Proposal A vs Proposal B."})
        self.assertEqual(res_stale["decision"], "deny")
        self.assertIn("Stale judge", res_stale["reason"])

    def test_phase3_blind_judging_anti_leakage(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)

        leaking_prompts = (
            "Evaluate Proposal A (Claude Opus 5.5) vs Proposal B.",
            "Compare Draft B by Gemini 4 Carbon against Draft A.",
            "Claude Opus authored Proposal C; please score it.",
            "Review Gemini 4 Argon's proposal for the cache layer.",
        )
        for idx, leak_text in enumerate(leaking_prompts):
            res_leak = self._pre("invoke_subagent", self._sub_args("judge", "Blind Judge", "pro", f"MODE: COUNCIL\nCOUNCIL_ROLE: judge\n{leak_text}"), call_id=f"leak_{idx}")
            self.assertEqual(res_leak["decision"], "deny", msg=f"Should deny: {leak_text}")
            self.assertIn("Blind judging violation", res_leak["reason"])

        res_role_judge = self._pre("invoke_subagent", self._sub_args("reviewer", "Judge", "pro", "REVIEW_LENS: architecture\nCOUNCIL_ROLE: judge\nOption 1 - GPT-5.2"), call_id="leak_role_judge")
        self.assertEqual(res_role_judge["decision"], "deny")
        self.assertIn("Blind judging violation", res_role_judge["reason"])

        benign_prompts = (
            "Proposal A: use carbon theme for the dashboard.\nProposal B: use Gemini API client with retry.",
            "Compare Proposal A, Proposal B, and Proposal C on correctness and maintainability.",
        )
        for idx, ok_text in enumerate(benign_prompts):
            res_ok = self._pre("invoke_subagent", self._sub_args("judge", f"Blind Judge {idx}", "pro", f"MODE: COUNCIL\nCOUNCIL_ID: blind-ok\nCOUNCIL_ROLE: judge\n{ok_text}"), call_id=f"benign_{idx}")
            self.assertEqual(res_ok["decision"], "allow", msg=f"Should allow: {ok_text}")

    def test_phase3_swarm_council_hooks_expansion_kill_and_stop(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)

        for act in ("models", "status", "title"):
            self.assertEqual(self._pre("swarm", {"action": act})["decision"], "allow")

        res_bad_name = self._pre(
            "swarm",
            {"action": "add", "name": "proposer-claude", "model": "claude-opus-4", "brief": "MODE: COUNCIL\nCOUNCIL_ID: arch-1\nCOUNCIL_ROLE: proposer\nDraft design."},
            call_id="sw_bad_name",
        )
        self.assertEqual(res_bad_name["decision"], "deny")
        self.assertIn("Blind council violation", res_bad_name["reason"])

        res_bad_judge = self._pre(
            "swarm",
            {"action": "add", "name": "judge-1", "model": "gemini-4-pro", "brief": "MODE: COUNCIL\nCOUNCIL_ID: arch-1\nCOUNCIL_ROLE: judge\nVote on Draft A by Claude Opus 5.5."},
            call_id="sw_bad_judge",
        )
        self.assertEqual(res_bad_judge["decision"], "deny")
        self.assertIn("Blind judging violation", res_bad_judge["reason"])

        self.assertEqual(
            self._pre("swarm", {"action": "add", "name": "proposer", "n": 2, "model": "claude-sonnet", "workspace": "branch", "brief": "MODE: COUNCIL\nCOUNCIL_ID: arch-1\nCOUNCIL_ROLE: proposer\nEVAL_CMD: python3 -m unittest"}, call_id="sw_n")["decision"],
            "allow",
        )
        self.assertEqual(
            self._pre("swarm", {"action": "add", "name": "reviewer", "tasks": ["task1", "task2"], "model": "gpt-5", "workspace": "share", "brief": "Review candidate patches."}, call_id="sw_tasks")["decision"],
            "allow",
        )
        self.assertEqual(
            self._pre("swarm", {"action": "add", "name": "judge-1", "model": "gemini-4-pro", "workspace": "inherit", "brief": "MODE: COUNCIL\nCOUNCIL_ID: arch-1\nCOUNCIL_ROLE: judge\nScore Proposal A vs Proposal B."}, call_id="sw_inherit")["decision"],
            "allow",
        )

        st = self._load_st()
        self.assertEqual(sorted(st["swarm_workers"].keys()), ["judge-1", "proposer-1", "proposer-2", "reviewer-1", "reviewer-2"])
        self.assertTrue(st["swarm_workers"]["proposer-1"]["active_worktree"])
        self.assertTrue(st["swarm_workers"]["reviewer-2"]["active_worktree"])
        self.assertFalse(st["swarm_workers"]["judge-1"]["active_worktree"])
        self.assertEqual(st["active_eval_cmd"], "python3 -m unittest")
        self.assertEqual(st["councils"]["arch-1"]["proposers"], ["proposer-1", "proposer-2"])
        self.assertEqual(st["councils"]["arch-1"]["judges"], ["judge-1"])
        self.assertEqual(st["councils"]["arch-1"]["models"], ["claude-sonnet", "gemini-4-pro"])

        self._pre("swarm", {"action": "model", "name": "judge-1", "model": "gemini-4-carbon"})
        self.assertEqual(self._load_st()["swarm_workers"]["judge-1"]["model"], "gemini-4-carbon")

        res_stop_blocked = self._stop(executionNum=1)
        self.assertEqual(res_stop_blocked["decision"], "continue")
        self.assertIn("Uncleaned isolated swarm worktrees remain", res_stop_blocked["reason"])
        self.assertIn("swarm(action='kill')", res_stop_blocked["reason"])

        for wname in ("proposer-1", "proposer-2", "reviewer-1", "reviewer-2"):
            self._pre("swarm", {"action": "kill", "name": wname})
        st_killed = self._load_st()
        self.assertEqual(st_killed["swarm_workers"]["proposer-1"]["status"], "killed")
        self.assertFalse(st_killed["swarm_workers"]["proposer-1"]["active_worktree"])
        self.assertEqual(st_killed["swarm_workers"]["judge-1"]["status"], "active")
        self.assertEqual(self._stop(executionNum=2)["decision"], "allow")

    def test_phase3_swarm_post_tool_rollback_breaker_and_replan(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)
        st = self._load_st()
        receipt = orchestrator_guard._get_test_receipt(st, self.base_payload)
        receipt["dirty"], receipt["verified"] = True, False
        orchestrator_guard._save_state(self.base_payload, st)

        res_trivial = self._pre("swarm", {"action": "add", "name": "proposer-a", "brief": "MODE: COUNCIL\nEVAL_CMD: true"}, call_id="sw_triv")
        self.assertEqual(res_trivial["decision"], "deny")
        self.assertIn("Trivial or no-op EVAL_CMD rejected", res_trivial["reason"])

        add_args_1 = {"action": "add", "name": "proposer-a", "model": "claude-opus", "brief": "MODE: COUNCIL\nCOUNCIL_ID: c-err\nCOUNCIL_ROLE: proposer\nDraft A"}
        self._pre("swarm", add_args_1, call_id="sw_err_1")
        self.assertIn("proposer-a", self._load_st()["swarm_workers"])

        self._post("swarm", add_args_1, error="swarm add failed", call_id="sw_err_1")
        st_err_1 = self._load_st()
        self.assertNotIn("proposer-a", st_err_1["swarm_workers"])
        self.assertEqual(st_err_1["councils"]["c-err"]["proposers"], [])
        self.assertEqual(st_err_1["circuit_breaker"]["failure_streak"], 1)
        self.assertFalse(st_err_1["circuit_breaker"]["tripped"])

        add_args_2 = {"action": "add", "name": "proposer-b", "n": 2, "model": "gpt-5", "brief": "MODE: COUNCIL\nCOUNCIL_ID: c-err\nCOUNCIL_ROLE: proposer\nDraft B"}
        self._pre("swarm", add_args_2)
        self._post("swarm", add_args_2, exit_code=1)
        st_err_2 = self._load_st()
        self.assertNotIn("proposer-b-1", st_err_2["swarm_workers"])
        self.assertNotIn("proposer-b-2", st_err_2["swarm_workers"])
        self.assertEqual(st_err_2["circuit_breaker"]["failure_streak"], 2)
        self.assertTrue(st_err_2["circuit_breaker"]["tripped"])

        self.assertEqual(self._pre("swarm", {"action": "status"})["decision"], "allow")
        self.assertEqual(self._pre("swarm", {"action": "kill", "name": "any"})["decision"], "allow")

        res_blocked = self._pre("swarm", {"action": "add", "name": "proposer-c", "brief": "Retry without replan"}, call_id="sw_blocked")
        self.assertEqual(res_blocked["decision"], "deny")
        self.assertIn("REPLAN:", res_blocked["reason"])

        res_replan = self._pre("swarm", {"action": "add", "name": "proposer-c", "model": "gemini-4-pro", "brief": "REPLAN: switch provider\nMODE: COUNCIL\nCOUNCIL_ID: c-err\nCOUNCIL_ROLE: proposer"}, call_id="sw_replan")
        self.assertEqual(res_replan["decision"], "allow")
        st_replanned = self._load_st()
        self.assertFalse(st_replanned["circuit_breaker"]["tripped"])
        self.assertEqual(st_replanned["circuit_breaker"]["replan_count"], 1)

        self._post("swarm", {"action": "add", "name": "proposer-c"}, output="Worker proposer-c ready", call_id="sw_replan")
        st_ok = self._load_st()
        self.assertEqual(st_ok["circuit_breaker"]["failure_streak"], 0)
        self.assertFalse(st_ok["circuit_breaker"]["tripped"])
        self.assertTrue(st_ok["test_receipt"]["dirty"])

    def test_phase3_non_swarm_council_proposer_and_cli_status(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)

        res_research_pro = self._pre("invoke_subagent", self._sub_args("research", "Proposer A", "pro", "MODE: COUNCIL\nCOUNCIL_ID: fallback-1\nCOUNCIL_ROLE: proposer\n1. Compare options."), call_id="inv_res_pro")
        self.assertEqual(res_research_pro["decision"], "allow")
        self.assertNotIn("overwrite", res_research_pro)

        res_research_flash = self._pre("invoke_subagent", self._sub_args("research", "Proposer B", "flash", "MODE: COUNCIL\nCOUNCIL_ID: fallback-1\nCOUNCIL_ROLE: proposer\n1. Summarize design."), call_id="inv_res_flash")
        self.assertEqual(res_research_flash["decision"], "allow")
        self.assertNotIn("overwrite", res_research_flash)

        res_council_inherit = self._pre("invoke_subagent", self._sub_args("council", "Proposer C", "inherit", "1. Propose architecture."), call_id="inv_c_inh")
        self.assertEqual(res_council_inherit["decision"], "allow")
        self.assertNotIn("overwrite", res_council_inherit)

        res_std_research = self._pre("invoke_subagent", self._sub_args("research", "Standard Research", "pro", "1. Investigate module."), call_id="inv_std_res")
        self.assertEqual(res_std_research["decision"], "deny")
        self.assertIn("Model='pro'", res_std_research["reason"])

        st = self._load_st()
        receipt = orchestrator_guard._get_test_receipt(st, self.base_payload)
        receipt["dirty"], receipt["verified"] = True, False
        orchestrator_guard._save_state(self.base_payload, st)

        council_cid = "sub-council-proposer-1"
        self._reg_sub("council", council_cid, "pro", mode="council", council_id="fallback-1", council_role="proposer")

        self.assertEqual(
            self._pre("write_file", {"Path": str(Path(self.artifact_dir) / "scratch" / "proposal_a.md"), "Content": "# Proposal A\n"}, cid=council_cid)["decision"],
            "allow",
        )
        res_code_mut = self._pre("write_file", {"Path": str(orchestrator_guard.PROJECT_ROOT / "src" / "hack.py"), "Content": "x = 1\n"}, cid=council_cid)
        self.assertEqual(res_code_mut["decision"], "deny")
        self.assertIn("may only write markdown reports", res_code_mut["reason"])

        self._post(
            "invoke_subagent",
            self._sub_args("council", "Proposer C", "inherit", "1. Propose architecture."),
            output="STATUS: COMPLETED\nFILES_CHANGED: scratch/proposal_a.md\nVERIFICATION: n/a\nNOTES: Drafted Proposal A.",
            call_id="inv_c_done",
        )
        st_after = self._load_st()
        self.assertTrue(st_after["test_receipt"]["dirty"])
        self.assertEqual(st_after["councils"]["fallback-1"]["proposers"], ["Proposer A", "Proposer B"])

        status_out = orchestrator_guard._cli_status(self.base_payload)
        self.assertIn("councils", status_out)
        self.assertIn("swarmWorkers", status_out)
        self.assertIn("fallback-1", status_out["councils"])

        prune_state = orchestrator_guard._default_state()
        for i in range(orchestrator_guard.MAX_STATE_INVOCATION_HISTORY + 5):
            prune_state["councils"][f"c-{i}"] = {"proposers": [f"p-{j}" for j in range(orchestrator_guard.MAX_STATE_LIST_ENTRIES + 5)], "judges": [], "models": [], "created_at": float(i)}
        for i in range(orchestrator_guard.MAX_STATE_LIST_ENTRIES + 5):
            prune_state["swarm_workers"][f"w-{i}"] = {"name": f"w-{i}", "status": "killed" if i < 10 else "active", "spawned_at": float(i)}
        orchestrator_guard._prune_state(prune_state)
        self.assertEqual(len(prune_state["councils"]), orchestrator_guard.MAX_STATE_INVOCATION_HISTORY)
        self.assertNotIn("c-0", prune_state["councils"])
        self.assertEqual(len(next(iter(prune_state["councils"].values()))["proposers"]), orchestrator_guard.MAX_STATE_LIST_ENTRIES)
        self.assertEqual(len(prune_state["swarm_workers"]), orchestrator_guard.MAX_STATE_LIST_ENTRIES)
        self.assertNotIn("w-0", prune_state["swarm_workers"])
        self.assertIn("w-5", prune_state["swarm_workers"])

    def test_phase3_blind_judge_metadata_filename_and_scratch_file_leaks(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)

        def _invoke_judge(prompt: str, call_id: str) -> dict:
            return self._pre("invoke_subagent", self._sub_args("judge", "Blind Judge", "inherit", prompt), call_id=call_id)

        for cid_call, text in (
            ("j_meta1", "MODE: COUNCIL\nCOUNCIL_ID: c-meta-1\nCOUNCIL_ROLE: judge\nModel: Claude Opus 5.5\nCompare proposals."),
            ("j_meta2", "MODE: COUNCIL\nCOUNCIL_ID: c-meta-2\nCOUNCIL_ROLE: judge\n- Author: Gemini 4 Carbon\nCompare proposals."),
            ("j_wide", "MODE: COUNCIL\nCOUNCIL_ID: c-wide\nCOUNCIL_ROLE: judge\nProposal A (written carefully over two iterations by Claude Opus 5.5)"),
            ("j_fname", "MODE: COUNCIL\nCOUNCIL_ID: c-fname\nCOUNCIL_ROLE: judge\nReview scratch/proposal_claude.md and scratch/proposal_b.md."),
        ):
            res = _invoke_judge(text, cid_call)
            self.assertEqual(res["decision"], "deny")
            self.assertIn("Blind judging violation", res["reason"])
        self.assertEqual(_invoke_judge("MODE: COUNCIL\nCOUNCIL_ID: c-meta-3\nCOUNCIL_ROLE: judge\nModel: GPT-5\nCompare.", "j_meta3")["decision"], "deny")
        self.assertEqual(_invoke_judge("MODE: COUNCIL\nCOUNCIL_ID: c-meta-4\nCOUNCIL_ROLE: judge\nAuthor: Claude Sonnet\nCompare.", "j_meta4")["decision"], "deny")

        scratch_dir = Path(self.artifact_dir) / "scratch"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        prop_a = scratch_dir / "proposal_a.md"
        prop_a.write_text("# Proposal A\nAuthor: Gemini 4 Carbon\nUse modular handlers.\n", encoding="utf-8")

        res_scratch_leak = _invoke_judge(f"MODE: COUNCIL\nCOUNCIL_ID: c-scratch-leak\nCOUNCIL_ROLE: judge\nReview {prop_a}.", "j_scratch_leak")
        self.assertEqual(res_scratch_leak["decision"], "deny")
        self.assertIn("proposal_a.md", res_scratch_leak["reason"])

        (scratch_dir / "proposal_b.md").write_text("# Proposal B\nImplement carbon accounting with the Gemini API client.\n", encoding="utf-8")
        self.assertEqual(_invoke_judge("MODE: COUNCIL\nCOUNCIL_ID: c-clean\nCOUNCIL_ROLE: judge\nReview scratch/proposal_b.md.", "j_clean")["decision"], "allow")

    def test_phase3_swarm_sharded_kill_base_name_and_kill_error_rollback(self) -> None:
        orchestrator_guard._cli_activate(self.base_payload)

        res_add = self._pre(
            "swarm",
            {"action": "add", "name": "proposer", "n": 2, "workspace": "share", "brief": "MODE: COUNCIL\nCOUNCIL_ID: c-shard\nCOUNCIL_ROLE: proposer\nDraft design"},
            call_id="sw_sharded_add",
        )
        self.assertEqual(res_add["decision"], "allow")
        st_added = self._load_st()
        self.assertEqual(st_added["swarm_workers"]["proposer-1"]["base_name"], "proposer")
        self.assertTrue(st_added["swarm_workers"]["proposer-1"]["sharded"])

        res_kill_base = self._pre("swarm", {"action": "kill", "name": "proposer"}, call_id="sw_kill_base")
        self.assertEqual(res_kill_base["decision"], "deny")
        self.assertIn("proposer-1", res_kill_base["reason"])
        self.assertIn("proposer-2", res_kill_base["reason"])

        self.assertEqual(self._pre("swarm", {"action": "kill", "name": "proposer-1"}, call_id="sw_kill_1_err")["decision"], "allow")
        self.assertEqual(self._load_st()["swarm_workers"]["proposer-1"]["status"], "killed")

        self._post("swarm", {"action": "kill", "name": "proposer-1"}, error="Worker not found or kill failed", call_id="sw_kill_1_err")
        st_rolled = self._load_st()
        self.assertEqual(st_rolled["swarm_workers"]["proposer-1"]["status"], "active")
        self.assertTrue(st_rolled["swarm_workers"]["proposer-1"]["active_worktree"])
        self.assertEqual(self._stop(executionNum=1)["decision"], "continue")

        for wname in ("proposer-1", "proposer-2"):
            cid_k = f"sw_kill_ok_{wname}"
            self.assertEqual(self._pre("swarm", {"action": "kill", "name": wname}, call_id=cid_k)["decision"], "allow")
            self._post("swarm", {"action": "kill", "name": wname}, output=f"Killed {wname}", call_id=cid_k)

        self.assertEqual(self._stop(executionNum=2)["decision"], "allow")

    def test_concurrency_abba_lock_order_and_atomic_persistence(self) -> None:
        import contextlib
        from unittest import mock
        import guard.identity as identity_mod
        import guard.policies as policies_mod
        import guard.state_store as state_store_mod
        import guard.subagent_policy as subagent_policy_mod
        import guard.verification as verification_mod

        root_dir = "conc_app/brain"
        parent_p = self._brain_payload("parent-abba", root=root_dir)
        child_p = self._brain_payload("child-abba", root=root_dir, isSubagent=True)
        scout_p = self._brain_payload("child-scout-1", root=root_dir, isSubagent=True)
        verifier_p = self._brain_payload("child-verifier", root=root_dir, isSubagent=True)
        sub_s_dir = Path(self.artifact_dir) / root_dir / "sub-s"
        root_cs2a_dir = Path(self.artifact_dir) / root_dir / "root-cs2a"
        sub_s_dir.mkdir(parents=True, exist_ok=True)
        root_cs2a_dir.mkdir(parents=True, exist_ok=True)

        orchestrator_guard._cli_activate(parent_p)
        orchestrator_guard.handle_pre_invocation(parent_p)
        self._pre("invoke_subagent", self._sub_args("implementer", "Worker", Workspace="inherit"), call_id="abba_spawn_1", base=parent_p)

        held_stack: list[str] = []
        orig_tx = state_store_mod.state_transaction

        @contextlib.contextmanager
        def checked_tx(payload_or_path: Any):
            resolved = str(state_store_mod._resolve_state_path(payload_or_path).resolve())
            distinct = set(held_stack)
            if distinct and resolved not in distinct:
                raise AssertionError(f"AB-BA deadlock risk: nested state_transaction {resolved} while holding {distinct}")
            held_stack.append(resolved)
            try:
                with orig_tx(payload_or_path):
                    yield
            finally:
                held_stack.pop()

        with contextlib.ExitStack() as stack:
            for mod_obj in (state_store_mod, identity_mod, policies_mod, subagent_policy_mod, verification_mod):
                stack.enter_context(mock.patch.object(mod_obj, "state_transaction", checked_tx))
            orchestrator_guard.handle_pre_invocation(child_p)
            res_edit = self._pre("edit_file", {"Path": str(orchestrator_guard.PROJECT_ROOT / "src" / "abba.py"), "Target": "a", "Replacement": "b"}, call_id="abba_edit", base=child_p)
            self.assertEqual(res_edit["decision"], "allow")
            self._post("run_command", {"CommandLine": "pytest tests/"}, exit_code=0, call_id="abba_test", base=child_p)
            self._pre("send_message", {"Recipient": "child-abba", "Message": "STAGE: 2/2\nALLOWED_FILES: src/abba.py\nProceed."}, call_id="abba_msg", base=parent_p)
            self._stop("STATUS: COMPLETED\nFILES_CHANGED: src/abba.py\nVERIFICATION: pytest passed\nNOTES: Done.", base=child_p)

        hp = self._brain_payload("parent-hijack", root=root_dir)
        orchestrator_guard._cli_activate(hp)
        for cid, ws in (("sync_branch", "branch"), ("sync_inherit", "inherit")):
            args_inv = {"TypeName": "implementer", "Workspace": ws, "Prompt": VALID_PROMPT}
            self._pre("invoke_subagent", args_inv, call_id=cid, base=hp)
            self._post("invoke_subagent", args_inv, output="STATUS: COMPLETED\nFILES_CHANGED: src/a.py\nVERIFICATION: pytest ok\nNOTES: done", call_id=cid, base=hp)
        self.assertEqual(len(self._load_st(hp)["pending_spawns"]), 1)

        self._pre("invoke_subagent", {"TypeName": "scout", "Role": "Log Scout", "Prompt": "TASK: Read logs.\nRULES:\n1. Read log.\nREPORT FORMAT:\nSTATUS: COMPLETED\nNOTES: ok"}, call_id="async_scout", base=hp)
        res_write = self._pre("write_file", {"Path": str(orchestrator_guard.PROJECT_ROOT / "src" / "hack.py"), "Content": "x = 1\n"}, base=scout_p)
        self.assertEqual(res_write["decision"], "deny")
        self.assertEqual(self._load_st(scout_p)["self_subagent_meta"]["type"], "scout")
        self.assertEqual(self._load_st(hp)["pending_spawns"][0]["status"], "completed")

        self._seed_state(hp, test_receipt={"dirty": True, "verified": False, "last_mutated_file": "src/core.py", "last_test_cmd": "", "mutation_epoch": 2})
        self._reg_sub("verifier", "child-verifier", "flash", payload=hp)
        self._post("run_command", {"CommandLine": "pytest tests/"}, exit_code=1, call_id="fail_v_test", base=verifier_p)
        p_after = self._load_st(hp)
        self.assertTrue(p_after["test_receipt"]["dirty"])
        self.assertFalse(p_after["test_receipt"]["verified"])
        self.assertEqual(p_after["test_receipt"]["mutation_epoch"], 2)

        corrupt_dir = Path(self.artifact_dir) / root_dir / "corrupt-brain"
        corrupt_dir.mkdir(parents=True, exist_ok=True)
        (corrupt_dir / "orchestrator_state.json").write_text("{bad json", encoding="utf-8")
        corrupt_loaded = self._load_st(self._payload(cid="corrupt-cid", artifactDirectoryPath=str(corrupt_dir), base=parent_p))
        self.assertTrue(corrupt_loaded["active"])
        self.assertTrue(corrupt_loaded["session_evaluated"])

        inv_res = self._pre(
            "invoke_subagent",
            self._sub_args("implementer", "Worker", "flash", "TASK: Edit src/a.py.\nALLOWED_FILES: src/a.py\nReference: .agents/plugins/philharmonic_orchestra/skills/philharmonic/SKILL.md\nRULES:\n1. Edit src/a.py.\nREPORT FORMAT:\nSTATUS: COMPLETED\nFILES_CHANGED: src/a.py\nVERIFICATION: ok\n"),
            call_id="spawn-sub-s",
            base=hp,
        )
        self.assertEqual(inv_res["decision"], "allow")
        sub_no_flag = {
            "conversationId": "sub-s",
            "workspacePaths": [str(orchestrator_guard.PROJECT_ROOT)],
            "artifactDirectoryPath": str(sub_s_dir),
            "lastUserInput": "TASK: Edit src/a.py.\nALLOWED_FILES: src/a.py\nReference: .agents/plugins/philharmonic_orchestra/skills/philharmonic/SKILL.md\n",
            "toolCall": {"name": "read_file", "args": {"Path": "src/a.py"}},
        }
        self.assertFalse(identity_mod._is_slash_command_triggered(sub_no_flag))
        sub_pay = dict(sub_no_flag, isSubagent=True)
        self.assertFalse(identity_mod._is_slash_command_triggered(sub_pay))
        orchestrator_guard.handle_pre_tool(sub_pay)
        sub_st = self._load_st(sub_pay)
        self.assertIsNotNone(sub_st.get("self_subagent_meta"))
        self.assertFalse(identity_mod._is_root_orchestrator(sub_pay, sub_st))

        cs2a_pay = {
            "conversationId": "root-cs2a",
            "workspacePaths": [str(orchestrator_guard.PROJECT_ROOT)],
            "artifactDirectoryPath": str(root_cs2a_dir),
            "toolCall": {"name": "read_file", "args": {"Path": str(orchestrator_guard.PROJECT_ROOT / ".agents/plugins/philharmonic_orchestra/skills/philharmonic/SKILL.md")}},
        }
        self.assertTrue(identity_mod._is_slash_command_triggered(cs2a_pay))
        orchestrator_guard.handle_pre_tool(cs2a_pay)
        self.assertEqual(self._load_st(cs2a_pay).get("root_conversation_id"), "root-cs2a")

    def test_guard_self_protection_and_failsafe_entrypoint(self) -> None:
        import contextlib
        import io
        from unittest import mock
        import guard.constants as guard_constants
        import guard.verification as guard_verification

        orchestrator_guard._cli_activate(self.base_payload)
        state_json = str(Path(self.artifact_dir) / "orchestrator_state.json")
        state_lock = str(Path(self.artifact_dir) / "orchestrator_state.lock")
        fb_state = str(orchestrator_guard.PROJECT_ROOT / ".agents" / ".state" / "orchestrator_state.json")
        plugin_policies = str(orchestrator_guard.PROJECT_ROOT / ".agents" / "plugins" / "philharmonic_orchestra" / "scripts" / "guard" / "policies.py")
        hooks_json = str(orchestrator_guard.PROJECT_ROOT / ".agents" / "hooks.json")

        self.assertFalse(orchestrator_guard._is_artifact_path(state_json, self.base_payload))
        self.assertFalse(orchestrator_guard._is_artifact_path(state_lock, self.base_payload))

        for target_p in (state_json, state_lock, fb_state, plugin_policies, hooks_json):
            self.assertEqual(self._pre("write_file", {"Path": target_p, "Content": "{}"})["decision"], "deny", f"Direct write must deny {target_p}")

        for bad_shell in (
            f"rm -f {state_json}",
            f"sed -i 's/a/b/' {plugin_policies}",
            f"cp README.md {hooks_json}",
            "python3 -c \"import pathlib; pathlib.Path('.agents/hooks.json').write_text('x')\"",
            "cd .agents/plugins/philharmonic_orchestra && ls",
            f"echo '{{}}' > {state_lock}",
            "chmod -x .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py",
        ):
            self.assertEqual(self._pre("run_command", {"CommandLine": bad_shell})["decision"], "deny", f"Shell guard protection must deny: {bad_shell}")

        self._reg_sub("implementer", "selfhost-dir", "flash", allowed_files=[".agents/plugins/philharmonic_orchestra/scripts/guard"])
        self.assertEqual(self._pre("edit_file", {"Path": plugin_policies, "Target": "a", "Replacement": "b"}, cid="selfhost-dir")["decision"], "deny")
        self._reg_sub("implementer", "selfhost-exact", "flash", allowed_files=[plugin_policies])
        self.assertEqual(self._pre("edit_file", {"Path": plugin_policies, "Target": "a", "Replacement": "b"}, cid="selfhost-exact")["decision"], "allow")

        for bad_cid in (123, ["x"]):
            self.assertIn(self._pre("read_file", {"Path": "README.md", "StartLine": 1, "EndLine": 5}, cid=bad_cid).get("decision"), ("allow", "deny"))
            self.assertIsInstance(guard_verification.handle_post_tool(self._payload(cid=bad_cid, tool="run_command", args={"CommandLine": "git status"}, output="ok")), dict)
            self.assertIsInstance(guard_verification.handle_stop(self._payload(cid=bad_cid, isSubagent=True, fullyIdle=True, terminationReason="NO_TOOL_CALL")), dict)
            guard_verification._propagate_receipt_to_parent_brain(self._payload(cid=bad_cid), dirty=True, verified=False)

        self.assertLess(len((Path(guard_constants.SCRIPT_DIR) / "orchestrator_guard.py").read_text(encoding="utf-8").splitlines()), 80)

        for exc_cls, expected_dec in ((ImportError("missing"), "allow"), (IndentationError("bad indent"), "allow"), (MemoryError("oom"), "deny"), (RecursionError("deep"), "deny")):
            for mode in ("pre-tool", "stop"):
                out_buf = io.StringIO()
                with contextlib.ExitStack() as stack:
                    stack.enter_context(mock.patch.object(sys, "argv", ["orchestrator_guard.py", mode]))
                    stack.enter_context(mock.patch.dict(orchestrator_guard._safe_entrypoint.__globals__, {"main": mock.MagicMock(side_effect=exc_cls)}))
                    stack.enter_context(mock.patch.object(sys, "stdout", out_buf))
                    cm = stack.enter_context(self.assertRaises(SystemExit))
                    orchestrator_guard._safe_entrypoint()
                self.assertEqual(cm.exception.code, 0)
                self.assertEqual(json.loads(out_buf.getvalue())["decision"], expected_dec)

        err_buf = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(sys, "argv", ["orchestrator_guard.py", "doctor"]))
            stack.enter_context(mock.patch.dict(orchestrator_guard._safe_entrypoint.__globals__, {"main": mock.MagicMock(side_effect=ImportError("broken"))}))
            stack.enter_context(mock.patch.object(sys, "stderr", err_buf))
            cm_doc = stack.enter_context(self.assertRaises(SystemExit))
            orchestrator_guard._safe_entrypoint()
        self.assertEqual(cm_doc.exception.code, 1)
        self.assertIn("Guard error: broken", err_buf.getvalue())

        tr_path = Path(self.temp_dir.name) / "transcript.jsonl"
        tr_path.write_text('[]\n123\n"hi"\n' + json.dumps({"type": "PLANNER_RESPONSE", "content": "final answer"}) + "\n", encoding="utf-8")
        self.assertEqual(guard_verification._extract_last_model_response(str(tr_path)), "final answer")

    def test_root_closed_world_run_command_allowlist(self) -> None:
        root_p = self._brain_payload("root-cw", lastUserInput="/philharmonic verify root allowlist")
        root_dir = root_p["artifactDirectoryPath"]
        orchestrator_guard._cli_activate(root_p)

        allowed_cmds = (
            "ls -la", "cat README.md", "rg foo src/", "git status", "git diff 2>&1", "git log -n 5",
            "git branch -a", "git tag -l", "pytest -v", "python3 -m unittest", "tox -e py311",
            "go test ./...", "cargo test", "ruff check .", "mypy src", "tsc --noEmit", "python3 --version",
            "/usr/bin/python3 --version", "node --version", "go version",
            "python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py doctor",
            "python3 .agents/plugins/philharmonic_orchestra/scripts/orchestrator_guard.py status",
            f"mkdir -p {root_dir}/scratch", f"echo hi > {root_dir}/scratch/report.md",
            "git add README.md",
            'git commit -m "Harden guard"',
            'git add README.md && git commit -m "Update docs"',
            'git commit -am "Update"', "git commit --amend --no-edit",
        )
        for idx, cmd in enumerate(allowed_cmds, start=1):
            res = self._pre("run_command", {"CommandLine": cmd}, base=root_p, invocationNum=idx)
            self.assertEqual(res["decision"], "allow", f"Root should allow {cmd!r}: {res}")

        # Fix 8: Root verified safe git stage/commit chain must NOT dirty test_receipt or bump mutation_epoch
        self._seed_state(root_p, test_receipt={"dirty": False, "verified": True, "last_test_cmd": "pytest -v", "mutation_epoch": 3})
        res_git_commit = self._pre("run_command", {"CommandLine": 'git add README.md && git commit -m "Docs"'}, base=root_p)
        self.assertEqual(res_git_commit["decision"], "allow")
        rec_after_git = self._load_st(root_p)["test_receipt"]
        self.assertFalse(rec_after_git["dirty"])
        self.assertTrue(rec_after_git["verified"])
        self.assertEqual(rec_after_git["mutation_epoch"], 3)

        denied_cmds = (
            "sh -co errexit 'rm -rf .'", "bash -c 'echo ok'", "fish -C 'echo hi' -c 'rm -rf .'", "mksh -c 'rm -rf .'",
            "rbash -c 'rm -rf .'", "python3 -c 'import x; assert x.ok()'", "python3 -c 'import math; print(math.sqrt(4))'",
            "python3 -c \"open('a','w').write('x')\"", "find . -exec ls {} +", "find . | xargs grep foo",
            "tar -tf /tmp/x.tar", "unzip -l /tmp/x.zip", "perl -e 'print 1'", "node -e 'console.log(1)'",
            "chmod +x scripts/foo.sh", "rsync -a src/ build/", "tar -xf /tmp/x.tar -C build", "patch -p1 -d build < /tmp/x.patch",
            "find . -name '*.pyc' -delete", "{rm,-rf,.agents}", "git diff {--output=/tmp/x,--stat}", "find . {-delete,-print}",
            "rg {--pre=sh,x}", "sort {-o,/tmp/x}", "tar --index-file=src/out -tf /tmp/x.tar",
            "tar --rsh-command='rm -rf .' -tf /tmp/x.tar", "tox -- rm -rf src", "jest -u", "vitest init",
            "go test -coverprofile=/tmp/cov.out ./...", "go test -o /tmp/t.bin ./...", "cargo test --target-dir=/tmp/t",
            "cargo test --manifest-path Cargo.toml", "bash -c 'echo $(rm -rf .)'", "echo `id`", "cat <<'EOF'\nhello\nEOF",
            "bash $'<<EOF\\n$(rm -rf .)\\nEOF'", "awk 'BEGIN{system(\"rm -rf .\")}'", "sed '1e rm -rf .' README.md",
            r"find . -exec awk 'BEGIN{system(\"rm -rf .\")}' {} \;", "git -c core.sshCommand='rm -rf .' status",
            "git clean -fd", "git diff --no-index a b", "git grep -O vim foo", "echo x > src/out.txt",
            "echo x > /tmp/out.txt", "pytest -o addopts=x", "pytest --basetemp=/tmp/b", "tsc --noemit false", "tsc --init",
            "git add -f .agents/.state/orchestrator_state.json", "git add .agents/.state/orchestrator_state.json",
            "git add .agents/hooks.json", "git add .agents/plugins/philharmonic_orchestra/scripts/guard/policies.py",
            "git add .", "git add -A",
            "git add .agents/plugins/philharmonic_orchestra/scripts/guard/policies.py && rm -rf .agents",
            "git -c core.editor=evil commit -m x", "GIT_EDITOR=evil git commit -m x", "git commit -e -m x",
            "git commit --interactive", "git commit -m x -- .agents/.state/orchestrator_state.json",
            "git commit -m x -- .agents/hooks.json", "git commit -m x -- .agents/plugins/philharmonic_orchestra/scripts/guard/policies.py",
            "./script --version", "/tmp/evil --help", "./python3 --version", "./bash --version",
            "export X='a[$(id)]'", "export X='a[$(id)]' && [[ $X -eq 0 ]]", "[[ $X -eq 0 ]]", "[ $X -eq 0 ]", "test $X -eq 0",
        )
        for cmd in denied_cmds:
            res = self._pre("run_command", {"CommandLine": cmd}, base=root_p)
            self.assertEqual(res["decision"], "deny", f"Root must deny {cmd!r}: {res}")

        # Fix 7: Bare pytest in /tmp or artifact/scratch Cwd must be denied on Root and never clear dirty receipt
        import guard.verification as guard_verification

        for bad_cwd in ("/tmp", f"{root_dir}/scratch"):
            self.assertEqual(self._pre("run_command", {"CommandLine": "pytest", "Cwd": bad_cwd}, base=root_p)["decision"], "deny")
        self._seed_state(root_p, test_receipt={"dirty": True, "verified": False, "last_test_cmd": "", "mutation_epoch": 4})
        for bad_cwd in ("/tmp", f"{root_dir}/scratch"):
            self._post("run_command", {"CommandLine": "pytest", "Cwd": bad_cwd}, exit_code=0, call_id=f"bad_cwd_{bad_cwd}", base=root_p)
            rec_bad = self._load_st(root_p)["test_receipt"]
            self.assertTrue(rec_bad["dirty"])
            self.assertFalse(rec_bad["verified"])
        subshell_tmp_cmd = "(cd /usr/lib/python3.12/email/mime) && cd ../../../../tmp && pytest"
        self.assertTrue(guard_verification._has_unsafe_test_cwd(subshell_tmp_cmd, str(orchestrator_guard.PROJECT_ROOT), root_p))
        self._post("run_command", {"CommandLine": subshell_tmp_cmd}, exit_code=0, call_id="subshell_tmp_cd", base=root_p)
        rec_sub = self._load_st(root_p)["test_receipt"]
        self.assertTrue(rec_sub["dirty"])
        self.assertFalse(rec_sub["verified"])

        st = self._load_st(root_p)
        st["circuit_breaker"]["tripped"], st["circuit_breaker"]["failure_streak"] = True, 2
        orchestrator_guard._save_state(root_p, st)
        self.assertEqual(self._pre("run_command", {"CommandLine": "ls -la"}, base=root_p)["decision"], "deny")
        self.assertEqual(self._pre("run_command", {"CommandLine": "git status"}, base=root_p)["decision"], "allow")

    def test_readonly_and_scratch_subagent_containment(self) -> None:
        app_dir = Path(self.temp_dir.name) / "ro_app"
        parent_dir = app_dir / "brain" / "parent-ro"
        other_dir = app_dir / "brain" / "other-ro"
        wt_root = app_dir / "worktrees" / "parent-ro" / "w1"
        for d in (parent_dir / "scratch", other_dir / "scratch", wt_root / "tests", wt_root / "scratch"):
            d.mkdir(parents=True, exist_ok=True)
        parent_p = self._brain_payload("parent-ro", root="ro_app/brain", lastUserInput="/philharmonic ro containment")
        orchestrator_guard._cli_activate(parent_p)

        ro_roles = (("ro-scout", "scout", "flash_lite"), ("ro-research", "research", "flash"), ("ro-reviewer", "reviewer", "pro"), ("ro-council", "council", "pro"), ("ro-judge", "judge", "pro"))
        for cid, tname, model in ro_roles:
            self._reg_sub(tname, cid, model, payload=parent_p)

        bad_cmds = (
            "echo x 1>a.py", "echo x &>a.py", "echo x >|a.py", "echo x >>a.py", "git checkout -- a.py",
            "python3 -c 'print(1)'", "python3 -c \"open('a.py','w').write('x')\"", "find . -exec ls {} +",
            "find . -name '*.tmp' -delete", "tar -tf /tmp/x.tar", "unzip -l /tmp/x.zip", "sh -co errexit 'rm -rf .'",
            "awk 'BEGIN{system(\"rm -rf .\")}'", "sed '1e rm -rf .' README.md", "cat <<'EOF'\nhello\nEOF",
            "git add README.md", 'git commit -m "x"',
            "export X='a[$(id)]' && [[ $X -eq 0 ]]", "[[ 1 -eq 1 ]]", "[ 1 -eq 1 ]", "test 1 -eq 1",
        )
        for cid, tname, _ in ro_roles:
            for idx, ok_cmd in enumerate(("git status", "ls -la", "pytest -v", "python3 --version"), start=1):
                r_ok = self._pre("run_command", {"CommandLine": ok_cmd}, cid=cid, base=parent_p, invocationNum=idx)
                self.assertEqual(r_ok["decision"], "allow", f"{tname} should allow {ok_cmd!r}: {r_ok}")
            for bad_cmd in bad_cmds:
                r_deny = self._pre("run_command", {"CommandLine": bad_cmd}, cid=cid, base=parent_p)
                self.assertEqual(r_deny["decision"], "deny", f"{tname} must deny {bad_cmd!r}: {r_deny}")

        own_scratch_md = str(parent_dir / "scratch" / "notes.md")
        for cid in ("ro-research", "ro-council"):
            self.assertEqual(self._pre("write_file", {"Path": own_scratch_md, "Content": "# ok\n"}, cid=cid, base=parent_p)["decision"], "allow")
            self.assertEqual(self._pre("run_command", {"CommandLine": f"echo hi > {own_scratch_md}"}, cid=cid, base=parent_p)["decision"], "allow")
            for bad_path in (str(orchestrator_guard.PROJECT_ROOT / "src" / "hack.py"), str(wt_root / "scratch" / "hack.md"), str(other_dir / "scratch" / "leak.md"), str(parent_dir / "outside.md")):
                self.assertEqual(self._pre("write_file", {"Path": bad_path, "Content": "# bad\n"}, cid=cid, base=parent_p)["decision"], "deny")
        for strict_cid in ("ro-scout", "ro-reviewer", "ro-judge"):
            self.assertEqual(self._pre("write_file", {"Path": own_scratch_md, "Content": "# no\n"}, cid=strict_cid, base=parent_p)["decision"], "deny")

        leak_json = parent_dir / "scratch" / "meta.json"
        leak_json.write_text('{"author": "Grok-3", "proposal": "A"}', encoding="utf-8")
        for leak_prompt in ("- **Author**: **Claude Opus 5.5**\nCompare A and B.", "Proposal A\n(Gemini 4 Pro)\nCompare.", "Option 1: `o3-mini`\nCompare.", f"Review {leak_json} for Proposal A."):
            r_j = self._pre("invoke_subagent", self._sub_args("judge", "Judge", "pro", f"MODE: COUNCIL\nCOUNCIL_ID: c-ro\nCOUNCIL_ROLE: judge\n{leak_prompt}"), call_id=f"jl_{hash(leak_prompt)}", base=parent_p)
            self.assertEqual(r_j["decision"], "deny")

        self._reg_sub("judge", "ro-blind-j", "pro", payload=parent_p, council_id="c-ro", council_role="judge")
        self.assertEqual(self._pre("send_message", {"Recipient": "ro-blind-j", "Message": "STATUS: update\nProposal A by Claude Opus 5.5"}, base=parent_p)["decision"], "deny")
        for bad_sw in ({"action": "add", "name": "judge-claude", "brief": "Evaluate."}, {"action": "add", "name": "judge-1", "brief": "Proposal A by ChatGPT vs Proposal B."}):
            self.assertEqual(self._pre("swarm", bad_sw, base=parent_p)["decision"], "deny")

        for non_impl in ("scout", "research", "verifier", "reviewer", "judge", "council"):
            self.assertEqual(self._pre("invoke_subagent", self._sub_args(non_impl, "Worker", prompt=f"MODE: BEST_OF_N\nREVIEW_LENS: sec\n{VALID_PROMPT}", Workspace="share"), call_id=f"bon_{non_impl}", base=parent_p)["decision"], "deny")
        for bad_args in (
            {"TypeName": "implementer", "Prompt": f"MODE: COUNCIL\n{VALID_PROMPT}"},
            {"TypeName": "implementer", "Workspace": "share", "Prompt": f"MODE: BEST_OF_N\nCOUNCIL_ID: c1\n{VALID_PROMPT}"},
            {"TypeName": "research", "Model": "pro", "Prompt": "COUNCIL_ID: c1\n1. Investigate."},
        ):
            self.assertEqual(self._pre("invoke_subagent", bad_args, call_id=f"mc_{hash(str(bad_args))}", base=parent_p)["decision"], "deny")

        for idx in (1, 2):
            a = {"TypeName": "implementer", "Workspace": "share", "Prompt": f"MODE: BEST_OF_N\nCOHORT: n1-c-{idx}\n{VALID_PROMPT}"}
            self._pre("invoke_subagent", a, call_id=f"n1_{idx}", base=parent_p)
            self._post("invoke_subagent", a, error="fail", call_id=f"n1_{idx}", base=parent_p)
        self.assertTrue(self._load_st(parent_p)["circuit_breaker"]["tripped"])

        st = self._load_st(parent_p)
        st["circuit_breaker"] = {"failure_streak": 0, "tripped": False, "seen_failures": [], "last_error": "", "replan_count": 0}
        orchestrator_guard._save_state(parent_p, st)
        args_orig = {"action": "add", "name": "prop-dup", "model": "claude-opus", "brief": "MODE: COUNCIL\nCOUNCIL_ID: c-dup\nCOUNCIL_ROLE: proposer\nDraft A"}
        self._pre("swarm", args_orig, call_id="d1", base=parent_p)
        self._post("swarm", args_orig, output="ok", call_id="d1", base=parent_p)
        args_dup = {"action": "add", "name": "prop-dup", "model": "gpt-5", "brief": "MODE: COUNCIL\nCOUNCIL_ID: c-dup\nCOUNCIL_ROLE: proposer\nDraft B"}
        self._pre("swarm", args_dup, call_id="d2", base=parent_p)
        self._post("swarm", args_dup, error="dup", call_id="d2", base=parent_p)
        st_restored = self._load_st(parent_p)
        self.assertEqual(st_restored["swarm_workers"]["prop-dup"]["model"], "claude-opus")
        self.assertNotIn("_prev_worker", st_restored["swarm_workers"]["prop-dup"])

        self._reg_sub("implementer", "wt-impl", "flash", payload=parent_p, workspace="branch", allowed_files=["tests/test_wt.py"])
        self.assertEqual(
            self._pre("write_file", {"Path": str(wt_root / "tests" / "test_wt.py"), "Content": "def test_x():\n    assert 1 == 1\n"}, cid="wt-impl", base=parent_p, workspacePaths=[str(orchestrator_guard.PROJECT_ROOT), str(wt_root)])["decision"],
            "deny",
        )

    def test_implementer_allowed_files_and_shell_boundaries(self) -> None:
        import guard.shell_targets as guard_shell_targets

        orchestrator_guard._cli_activate(self.base_payload)
        self._reg_sub("implementer", "impl-scoped", "flash", allowed_files=["src/allowed.py", "tests/test_ok.py"])
        self._reg_sub("implementer", "impl-unrestricted", "flash")

        self.assertEqual(self._pre("edit_file", {"Path": "src/allowed.py", "Target": "a", "Replacement": "b"}, cid="impl-scoped")["decision"], "allow")
        self.assertEqual(self._pre("write_file", {"Path": "src/other.py", "Content": "x = 1\n"}, cid="impl-scoped")["decision"], "deny")

        ok_cmds = (
            "touch src/allowed.py", "dd if=/dev/zero of=src/allowed.py bs=1 count=0",
            "sed -i 's/a/b/' src/allowed.py", "sed -i 's/hello/world/' src/allowed.py",
            "cp src/protected.py src/allowed.py", "rm -f src/allowed.py", "mv src/allowed.py tests/test_ok.py",
            "chmod +x src/allowed.py", "chmod -x src/allowed.py",
            "sort -o src/allowed.py README.md", "sort -uo src/allowed.py README.md", "cat <<'EOF'\nhello\nEOF",
            "pytest --version", "pytest --help",
        )
        for idx, ok_cmd in enumerate(ok_cmds, start=1):
            res = self._pre("run_command", {"CommandLine": ok_cmd}, cid="impl-scoped", invocationNum=idx)
            self.assertEqual(res["decision"], "allow", f"Scoped implementer should allow {ok_cmd!r}: {res}")

        st_root = self._load_st()
        self.assertTrue(st_root["test_receipt"]["dirty"])
        self.assertGreaterEqual(st_root["test_receipt"]["mutation_epoch"], 1)

        bad_cmds = (
            "touch src/other.py", "sed -i 's/a/b/' src/other.py", f"echo x > /tmp/..{orchestrator_guard.PROJECT_ROOT}/src/other.py",
            "ln -s src/protected.py src/allowed.py", "cp -s src/protected.py src/allowed.py", "ln src/protected.py src/allowed.py",
            "tar -xf /tmp/x.tar", "rsync -a src/allowed.py src/other.py", "patch -p1 < /tmp/x.patch",
            "sh -c 'touch src/allowed.py'", "bash -c 'echo x > src/allowed.py'", "mksh -c 'rm -rf .'", "perl -i -pe 's/a/b/' src/allowed.py",
            "(bash -h; true) < /tmp/x",
            "cp -ft src/outside src/allowed.py", "cp -ftsrc/outside src/allowed.py", "mv -ft src/outside src/allowed.py",
            "sort -uo src/other.py README.md", "sort -uosrc/other.py README.md",
            "sed -i '1e rm -rf .' src/allowed.py", "sed -i '1e id' src/allowed.py", "sed -i 's/a/b/e' src/allowed.py",
            "sed -i 's/a/b/w out.txt' src/allowed.py", "sed -i 's/a/b/w src/other.py' src/allowed.py",
            "sed -i 's/a/b/W src/other.py' src/allowed.py", "sed -i.ext 'e' src/allowed.py", "cd /tmp && sed 'e' -- -e1",
            "sed -i 's|.*[[:alpha:]|].*|id;#|e' src/allowed.py", "sed -i 's|.*[[:alpha:]|].*|x|w src/other.py' src/allowed.py",
            "sed -i '/[[:alpha:]/]/e sh /tmp/x' src/allowed.py", "cd /tmp && pytest --help",
            "chmod -x src/other.py", "./script --version",
        )
        for bad_cmd in bad_cmds:
            res = self._pre("run_command", {"CommandLine": bad_cmd}, cid="impl-scoped")
            self.assertEqual(res["decision"], "deny", f"Scoped implementer must deny {bad_cmd!r}: {res}")

        self.assertEqual(self._pre("run_command", {"CommandLine": "sed -i 's/a/b/w .agents/hooks.json' src/allowed.py"}, cid="impl-unrestricted")["decision"], "deny")
        self.assertEqual(self._pre("run_command", {"CommandLine": "chmod -w+x .agents/hooks.json"}, cid="impl-unrestricted")["decision"], "deny")
        self.assertEqual(self._pre("run_workflow", {"Description": "run", "Script": "pytest"}, cid="impl-scoped")["decision"], "deny")

        # Direct shell_targets assertions for clustered flags, negative chmod mode, dev meta allowlist, and staging pytest Cwd
        self.assertEqual(guard_shell_targets._extract_shell_write_targets("cp -ft out_dir src.py", ["cp -ft out_dir src.py"])[0], ["out_dir"])
        self.assertEqual(guard_shell_targets._extract_shell_write_targets("cp -ftout_dir src.py", ["cp -ftout_dir src.py"])[0], ["out_dir"])
        self.assertEqual(guard_shell_targets._extract_shell_write_targets("cp -t=d src", ["cp -t=d src"])[0], ["=d"])
        self.assertEqual(guard_shell_targets._extract_shell_write_targets("sort -uo out.txt in.txt", ["sort -uo out.txt in.txt"])[0], ["out.txt"])
        self.assertEqual(guard_shell_targets._extract_shell_write_targets("sort -uoout.txt in.txt", ["sort -uoout.txt in.txt"])[0], ["out.txt"])
        self.assertEqual(guard_shell_targets._extract_shell_write_targets("sort -o=a.py in.txt", ["sort -o=a.py in.txt"])[0], ["=a.py"])
        self.assertIn("orchestrator_guard.py", guard_shell_targets._extract_shell_write_targets("chmod -x orchestrator_guard.py", ["chmod -x orchestrator_guard.py"])[0])
        self.assertIn(".agents/hooks.json", guard_shell_targets._extract_shell_write_targets("chmod -w+x .agents/hooks.json", ["chmod -w+x .agents/hooks.json"])[0])
        self.assertTrue(guard_shell_targets._extract_shell_write_targets("./script --version", ["./script --version"])[1])
        self.assertTrue(guard_shell_targets._extract_shell_write_targets("./bash --version", ["./bash --version"])[1])
        self.assertTrue(guard_shell_targets._extract_shell_write_targets("bash --version", ["bash --version"])[1])
        self.assertTrue(guard_shell_targets._extract_shell_write_targets("(bash -h; true) < /tmp/x", ["(bash -h; true) < /tmp/x"])[1])
        self.assertFalse(guard_shell_targets._extract_shell_write_targets("pytest --version", ["pytest --version"])[1])
        self.assertFalse(guard_shell_targets._extract_shell_write_targets("pytest --help", ["pytest --help"])[1])
        self.assertFalse(guard_shell_targets._extract_shell_write_targets("/usr/bin/python3 --version", ["/usr/bin/python3 --version"])[1])
        self.assertEqual(guard_shell_targets._extract_shell_write_targets("sed -i 's/hello/world/' a.py", ["sed -i 's/hello/world/' a.py"]), (["a.py"], False))
        self.assertTrue(guard_shell_targets._extract_shell_write_targets("sed -i.ext 'e' a.py", ["sed -i.ext 'e' a.py"])[1])
        self.assertTrue(guard_shell_targets._extract_shell_write_targets("cd /tmp && sed 'e' -- -e1", ["cd /tmp", "sed 'e' -- -e1"])[1])
        self.assertTrue(guard_shell_targets._extract_shell_write_targets("cd /tmp && pytest --help", ["cd /tmp", "pytest --help"])[1])
        self.assertTrue(guard_shell_targets._extract_shell_write_targets("pytest --help", ["pytest --help"], cwd=f"{self.artifact_dir}/scratch", payload=self.base_payload)[1])
        self.assertTrue(guard_shell_targets._extract_shell_write_targets("sed -i 's|.*[[:alpha:]|].*|id;#|e' a.py", ["sed -i 's|.*[[:alpha:]|].*|id;#|e' a.py"])[1])
        self.assertTrue(guard_shell_targets._extract_shell_write_targets("sed -i 's|.*[[:alpha:]|].*|x|w other.py' a.py", ["sed -i 's|.*[[:alpha:]|].*|x|w other.py' a.py"])[1])
        self.assertTrue(guard_shell_targets._extract_shell_write_targets("sed -i '/[[:alpha:]/]/e sh /tmp/x' a.py", ["sed -i '/[[:alpha:]/]/e sh /tmp/x' a.py"])[1])
        self.assertTrue(guard_shell_targets._extract_shell_write_targets("pytest", ["pytest"], cwd=f"{self.artifact_dir}/scratch", payload=self.base_payload)[1])

        for ancestor_cmd in ("rm -rf {.,foo}", "chmod -R 000 {.,foo}"):
            res_anc = self._pre("run_command", {"CommandLine": ancestor_cmd}, cid="impl-unrestricted")
            self.assertEqual(res_anc["decision"], "deny", f"Unrestricted implementer must deny {ancestor_cmd!r}: {res_anc}")

    def test_workspace_containment_stays_in_workspace(self) -> None:
        import guard.identity as guard_identity

        ws_dir = Path(self.temp_dir.name) / "ws"
        (ws_dir / "src").mkdir(parents=True, exist_ok=True)
        outside_file = Path(self.temp_dir.name) / "outside_secret.txt"
        outside_file.write_text("secret", encoding="utf-8")
        (ws_dir / "src" / "escape_link").symlink_to(outside_file)
        roots = [ws_dir.resolve()]

        for ws_bad in ("cat /etc/passwd", "cat ../outside_secret.txt", "cat src/escape_link", "cd /etc && ls", "cd ~-", "cd ~+", "coproc cat /etc/passwd", "cat $HOME/secret"):
            self.assertFalse(guard_identity._command_stays_in_workspace(ws_bad, str(ws_dir), roots), f"_command_stays_in_workspace must reject {ws_bad!r}")

        for ws_ok in ("cat README.md", "ls src/", "cd src && ls", "echo ${#var} foo#bar", "git status"):
            self.assertTrue(guard_identity._command_stays_in_workspace(ws_ok, str(ws_dir), roots), f"_command_stays_in_workspace should allow {ws_ok!r}")

    def test_pass1a_bug_fixes(self) -> None:
        import guard.identity as guard_identity
        import guard.path_guards as guard_path_guards

        root_state = {"root_conversation_id": "root-conv", "self_subagent_meta": {"type": "implementer", "role": "implementer"}}
        self.assertFalse(guard_identity._is_known_subagent(self._payload(cid="root-conv"), root_state, persist_self_meta=False))

        mixed_json = str(Path(self.artifact_dir) / "Orchestrator_State.JSON")
        mixed_lock = str(Path(self.artifact_dir) / "ORCHESTRATOR_STATE.LOCK")
        self.assertFalse(guard_path_guards._is_artifact_path(mixed_json, self.base_payload))
        self.assertFalse(guard_path_guards._is_artifact_path(mixed_lock, self.base_payload))
        agents_dir = orchestrator_guard.PROJECT_ROOT / ".agents"
        self.assertTrue(guard_path_guards._is_guard_state_target(".state/sub", payload=self.base_payload, cwd=str(agents_dir)))
        self.assertTrue(guard_path_guards._is_guard_plugin_target("hooks.json", cwd=str(agents_dir)))

        self.assertFalse(guard_path_guards._safe_fnmatchcase("a.py", "[[:punct:]]*.py"))
        self.assertFalse(guard_path_guards._safe_fnmatchcase("a.py", "{a,b}.py"))
        self.assertTrue(guard_path_guards._safe_fnmatchcase("foo.py", "*.py"))

    def test_workflow_closed_world_allowlist_and_eval_receipts(self) -> None:
        import guard.ast_guard as guard_ast
        import guard.constants as guard_constants
        import guard.policies as guard_policies

        orchestrator_guard._cli_activate(self.base_payload)

        bad_workflow_args = (
            {"Script": "# EVAL_CMD: rm -rf src\npytest", "Cwd": str(guard_constants.PROJECT_ROOT)},
            {"Script": "# EVAL_CMD: true\necho hi", "Cwd": str(guard_constants.PROJECT_ROOT)},
            {"Script": "pytest", "Cwd": str(guard_constants.PROJECT_ROOT / ".agents" / "plugins" / "philharmonic_orchestra")},
            {"ScriptPath": str(guard_constants.PROJECT_ROOT / ".agents" / ".state" / "orchestrator_state.json"), "Cwd": str(guard_constants.PROJECT_ROOT)},
            {"ScriptPath": str(guard_constants.PROJECT_ROOT / ".agents" / "plugins" / "philharmonic_orchestra" / "scripts" / "orchestrator_guard.py"), "Cwd": str(guard_constants.PROJECT_ROOT)},
        )
        for wf_args in bad_workflow_args:
            res = self._pre("run_workflow", wf_args)
            self.assertEqual(res["decision"], "deny", f"run_workflow must deny {wf_args!r}: {res}")

        for ok_wf in ("import json\ndata = json.dumps({'ok': True})\nagent('scout', data)", "# EVAL_CMD: python3 scripts/bench.py\npython3 scripts/bench.py"):
            res_ok = self._pre("run_workflow", {"Script": ok_wf, "Cwd": str(guard_constants.PROJECT_ROOT)})
            self.assertEqual(res_ok["decision"], "allow", f"run_workflow should allow {ok_wf!r}: {res_ok}")

        for triv in ("sh -c 'true'", "bash -c -- 'true'", "python3 -c 'pass'", "python3 -B -c 'pass'", "python3 -c 'import sys; sys.exit(0)'", "pytest || true", "pytest; true", "pytest | cat"):
            self.assertTrue(guard_policies._is_trivial_eval_command(triv), f"Should be trivial: {triv!r}")

        for bad_test in ("pytest || true", "pytest; true", "pytest | cat", "pytest &", "tox -- rm -rf src", "go test -coverprofile=/tmp/c ./..."):
            self.assertFalse(guard_ast._is_test_command(bad_test), f"_is_test_command must reject {bad_test!r}")
        self.assertTrue(guard_ast._is_test_command("cd sub && pytest"))
        self.assertTrue(guard_ast._is_test_command("pytest && echo done"))

        eval_st = orchestrator_guard._default_state()
        for triv_decl in ("jest -v", "vitest list", "go test -list .", "cargo test --list", "pytest -c x.ini"):
            eval_st["active_eval_cmd"] = triv_decl
            self.assertFalse(guard_policies._is_declared_eval_command(triv_decl, eval_st, self.base_payload))

        eval_st["active_eval_cmd"] = "python3 bench.py"
        for bad_decl in ("python3 bench.py; true", "python3 bench.py | cat", "cd /tmp/fake && python3 bench.py"):
            self.assertFalse(guard_policies._is_declared_eval_command(bad_decl, eval_st, self.base_payload))
        self.assertTrue(guard_policies._is_declared_eval_command("cd sub && python3 bench.py", eval_st, self.base_payload))

    def test_module_architecture_and_export_parity(self) -> None:
        import ast
        import guard
        import guard.constants as guard_constants

        public_exports = (
            "CLI_MODES", "PROJECT_ROOT", "SCRIPT_DIR", "audit_diff_text",
            "handle_post_tool", "handle_pre_invocation", "handle_pre_tool",
            "handle_stop", "main", "register_subagent", "state_transaction",
            "validate_subagent_report",
        )
        self.assertEqual(tuple(guard.__all__), public_exports)
        self.assertEqual(len(guard.__all__), 12)
        self.assertEqual(list(guard.__all__), sorted(guard.__all__))
        for sym in guard.__all__:
            self.assertTrue(hasattr(guard, sym), f"Missing guard.{sym}")
            self.assertTrue(hasattr(orchestrator_guard, sym), f"Missing orchestrator_guard.{sym}")
        for compat_sym in ("_default_state", "_load_state", "_is_test_command", "_split_shell_commands"):
            self.assertTrue(hasattr(guard, compat_sym), f"Missing compat guard.{compat_sym}")
            self.assertTrue(hasattr(orchestrator_guard, compat_sym), f"Missing compat orchestrator_guard.{compat_sym}")

        scripts_dir = Path(guard_constants.SCRIPT_DIR)
        guard_dir = scripts_dir / "guard"
        deps: dict[str, list[str]] = {}
        line_counts: dict[str, int] = {}
        for py_file in sorted(guard_dir.glob("*.py")):
            mod = "guard" if py_file.name == "__init__.py" else f"guard.{py_file.stem}"
            src = py_file.read_text(encoding="utf-8")
            line_counts[py_file.name] = len(src.splitlines())
            tree = ast.parse(src)
            edges: set[str] = set()
            for node in tree.body:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("guard"):
                            edges.add(alias.name)
                elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("guard"):
                    edges.add(node.module)
            deps[mod] = sorted(edges)

        visiting: set[str] = set()
        visited: set[str] = set()

        def _check_acyclic(node_name: str) -> None:
            self.assertNotIn(node_name, visiting, f"Cycle detected at {node_name}")
            if node_name in visited:
                return
            visiting.add(node_name)
            for nxt in deps.get(node_name, ()):
                _check_acyclic(nxt)
            visiting.remove(node_name)
            visited.add(node_name)

        for mod_name in deps:
            _check_acyclic(mod_name)
        self.assertEqual(len(visited), len(deps))

        self.assertLess(len((scripts_dir / "orchestrator_guard.py").read_text(encoding="utf-8").splitlines()), 80)
        budgets = {
            "__init__.py": 90,
            "ast_guard.py": 200,
            "cli.py": 200,
            "constants.py": 350,
            "identity.py": 820,
            "model_tiers.py": 340,
            "path_guards.py": 560,
            "policies.py": 520,
            "root_allowlist.py": 420,
            "shell_targets.py": 380,
            "shell_tokens.py": 760,
            "state_store.py": 300,
            "subagent_policy.py": 1400,
            "verification.py": 1150,
        }
        self.assertEqual(set(budgets.keys()), set(line_counts.keys()))
        for fname, limit in budgets.items():
            self.assertLess(line_counts[fname], limit, f"{fname} exceeded {limit} lines: {line_counts[fname]}")
        self.assertGreater(sum(line_counts.values()), 0)
        self.assertLess(sum(line_counts.values()), 6800)

    def test_root_safe_git_push_allowlist(self) -> None:
        root_p = self._brain_payload("root-git-push", lastUserInput="/philharmonic safe git push")
        orchestrator_guard._cli_activate(root_p)

        allowed_pushes = (
            "git push",
            "git push origin",
            "git push origin master",
            "git push -u origin feature/foo",
            "git push --dry-run origin HEAD:main",
            "git push origin v1.0.0 --tags",
            'git add -u && git commit -m "msg" && git push origin master',
            "git push -nvqu origin main",
            "git push -- origin master",
        )
        for idx, cmd in enumerate(allowed_pushes, start=1):
            res = self._pre("run_command", {"CommandLine": cmd}, base=root_p, invocationNum=idx)
            self.assertEqual(res["decision"], "allow", f"Root should allow {cmd!r}: {res}")

        denied_pushes = (
            "git push -f",
            "git push --force",
            "git push --force-with-lease",
            "git push --force-if-includes",
            "git push origin +master",
            "git push origin :master",
            "git push origin main:+main",
            "git push origin main:",
            "git push --delete origin master",
            "git push -d origin master",
            "git push --prune origin",
            "git push --mirror origin",
            "git push --all origin",
            "git push --no-verify origin master",
            "git push -o foo origin master",
            "git push --receive-pack=sh origin master",
            "git push --exec=sh origin master",
            "git push --repo=https://example.com/repo.git master",
            "git push --recurse-submodules=on-demand origin master",
            "git push ext::sh origin master",
            "git push /tmp/repo master",
            "git push https://example.com/repo.git master",
            "git push user@host:repo.git master",
            "git push origin 'refs/heads/*'",
            "git push origin a..b",
            "git push origin HEAD//main",
            "git push origin main@{1}",
            "git push -- -origin master",
            "./git push",
            "/tmp/git push",
            "./git status",
            "/tmp/git status",
            "SSLKEYLOGFILE=.agents/hooks.json git push",
            "HTTPS_PROXY=http://evil git push",
            "https_proxy=http://evil git push",
            "GIT_SSL_NO_VERIFY=1 git push",
            "GIT_ALLOW_PROTOCOL=ext git push",
            "GNUPGHOME=/tmp/g git push",
            "CURL_CA_BUNDLE=/tmp/ca git push origin main",
            "git --bare -C . push origin main",
            "git --git-dir=/tmp/repo push",
            "git --git-dir=. push",
            "git --work-tree=. push",
            "git --git-dir=. add README.md",
            'git --work-tree=. commit -m "msg"',
            "git -C../out=1 push",
            "git push evil_repo main",
        )
        for cmd in denied_pushes:
            res = self._pre("run_command", {"CommandLine": cmd}, base=root_p)
            self.assertEqual(res["decision"], "deny", f"Root must deny {cmd!r}: {res}")

        res_c_status = self._pre("run_command", {"CommandLine": "git -C../out=1 status"}, base=root_p)
        self.assertIn(res_c_status["decision"], ("deny", "ask"), f"Root must block -C../out=1 status: {res_c_status}")
        self.assertFalse(orchestrator_guard._command_stays_in_workspace("git -C/etc/evil=/tmp status", str(orchestrator_guard.PROJECT_ROOT), [orchestrator_guard.PROJECT_ROOT]))
        self.assertEqual(self._pre("run_command", {"CommandLine": "git push evil_repo main", "Cwd": "tests"}, base=root_p)["decision"], "deny")
        self.assertEqual(self._pre("run_command", {"CommandLine": "git push origin main", "Cwd": "tests"}, base=root_p, invocationNum=22)["decision"], "allow")

        with tempfile.TemporaryDirectory() as repo_td:
            td_p = Path(repo_td)
            td_base = dict(root_p, workspacePaths=[str(td_p)])
            (td_p / "evil_repo").mkdir()
            self.assertEqual(
                self._pre("run_command", {"CommandLine": "git push evil_repo main", "Cwd": str(td_p)}, base=td_base)["decision"],
                "deny",
            )
            (td_p / "evil_repo").rmdir()
            (td_p / "evil_repo.git").mkdir()
            self.assertEqual(
                self._pre("run_command", {"CommandLine": "git push evil_repo main", "Cwd": str(td_p)}, base=td_base)["decision"],
                "deny",
            )
            (td_p / "evil_repo.git").rmdir()
            (td_p / ".git").mkdir()
            (td_p / ".git" / "config").write_text('[remote "origin"]\n\turl = https://example.com/repo.git\n', encoding="utf-8")
            self.assertEqual(
                self._pre("run_command", {"CommandLine": "git push evil_repo main", "Cwd": str(td_p)}, base=td_base)["decision"],
                "deny",
            )
            self.assertEqual(
                self._pre("run_command", {"CommandLine": "git push origin main", "Cwd": str(td_p)}, base=td_base, invocationNum=20)["decision"],
                "allow",
            )

        self._seed_state(root_p, test_receipt={"dirty": False, "verified": True, "last_test_cmd": "pytest -v", "mutation_epoch": 5})
        res_push = self._pre("run_command", {"CommandLine": "git push origin master"}, base=root_p, invocationNum=21)
        self.assertEqual(res_push["decision"], "allow")
        rec_after = self._load_st(root_p)["test_receipt"]
        self.assertFalse(rec_after["dirty"])
        self.assertTrue(rec_after["verified"])
        self.assertEqual(rec_after["mutation_epoch"], 5)

        subagents = (
            ("push-scout", "scout", "flash_lite"),
            ("push-verifier", "verifier", "pro"),
            ("push-reviewer", "reviewer", "pro"),
            ("push-judge", "judge", "pro"),
            ("push-research", "research", "flash"),
            ("push-council", "council", "pro"),
        )
        for cid, tname, model in subagents:
            self._reg_sub(tname, cid, model, payload=root_p)
            for push_cmd in ("git push", "git push origin master", "git push --dry-run origin HEAD:main"):
                res_sub = self._pre("run_command", {"CommandLine": push_cmd}, cid=cid, base=root_p)
                self.assertEqual(res_sub["decision"], "deny", f"{tname} must deny {push_cmd!r}: {res_sub}")

    def test_model_tier_mapping_hydrogen_collapse_and_config(self) -> None:
        import os
        from unittest import mock
        import guard.model_tiers as mt

        root_p = self._brain_payload("root-model-tiers", lastUserInput="/philharmonic test model tiers")
        orchestrator_guard._cli_activate(root_p)

        simple_read_prompt = (
            "TASK: Read `src/auth/parser.py` constants.\nRULES:\n1. Open `src/auth/parser.py`.\n"
            "2. Check MAX_LEN.\n3. Do not change any file.\n"
            "REPORT FORMAT:\nSTATUS: COMPLETED\nFILES_CHANGED: NONE\nVERIFICATION: ok\nNOTES: done\n"
        )
        complex_read_prompt = (
            "TASK: Audit auth across four modules.\nRULES:\n1. Open `a.py`.\n2. Open `b.py`.\n"
            "3. Open `c.py`.\n4. Compare token expiry logic.\n"
            "REPORT FORMAT:\nSTATUS: COMPLETED\nFILES_CHANGED: NONE\nVERIFICATION: ok\nNOTES: done\n"
        )

        # 1. Many-to-one collapse: both lite and standard map to "hydrogen", frontier to "carbon"
        env_collapse = {
            "PHILHARMONIC_MODEL_LITE": "hydrogen",
            "PHILHARMONIC_MODEL_STANDARD": "hydrogen",
            "PHILHARMONIC_MODEL_FRONTIER": "carbon",
        }
        with mock.patch.dict(os.environ, env_collapse, clear=False):
            # Legacy flash_lite / flash rewrite to hydrogen while preserving distinct model_tier
            r_scout_leg = self._pre("invoke_subagent", self._sub_args("scout", "Legacy Scout", "flash_lite", simple_read_prompt), call_id="mt_1", base=root_p)
            self.assertEqual(r_scout_leg["decision"], "allow")
            self.assertEqual(r_scout_leg["overwrite"]["Model"], "hydrogen")

            r_impl_leg = self._pre("invoke_subagent", self._sub_args("implementer", "Legacy Impl", "flash"), call_id="mt_2", base=root_p)
            self.assertEqual(r_impl_leg["decision"], "allow")
            self.assertEqual(r_impl_leg["overwrite"]["Model"], "hydrogen")

            # Direct Model="hydrogen" needs no overwrite and assigns role-appropriate model_tier
            r_scout_h = self._pre("invoke_subagent", self._sub_args("scout", "Hydrogen Scout", "hydrogen", simple_read_prompt), call_id="mt_3", base=root_p)
            self.assertEqual(r_scout_h, {"decision": "allow"})

            r_impl_h = self._pre("invoke_subagent", self._sub_args("implementer", "Hydrogen Impl", "hydrogen"), call_id="mt_4", base=root_p)
            self.assertEqual(r_impl_h, {"decision": "allow"})

            # High-reasoning verifier accepts hydrogen (standard) and rewrites pro -> carbon, flash_lite -> hydrogen (standard)
            r_ver_h = self._pre("invoke_subagent", self._sub_args("verifier", "Ver H", "hydrogen", simple_read_prompt), call_id="mt_5", base=root_p)
            self.assertEqual(r_ver_h, {"decision": "allow"})
            r_ver_pro = self._pre("invoke_subagent", self._sub_args("verifier", "Ver Pro", "pro", simple_read_prompt), call_id="mt_6", base=root_p)
            self.assertEqual(r_ver_pro["decision"], "allow")
            self.assertEqual(r_ver_pro["overwrite"]["Model"], "carbon")
            r_ver_lite = self._pre("invoke_subagent", self._sub_args("verifier", "Ver Lite", "flash_lite", simple_read_prompt), call_id="mt_7", base=root_p)
            self.assertEqual(r_ver_lite["decision"], "allow")
            self.assertEqual(r_ver_lite["overwrite"]["Model"], "hydrogen")

            # Best-of-N allows frontier (pro -> carbon) and standard (hydrogen)
            r_bon = self._pre(
                "invoke_subagent",
                self._sub_args("implementer", "BoN Cand", "pro", "MODE: BEST_OF_N\n" + VALID_PROMPT, Workspace="branch"),
                call_id="mt_8",
                base=root_p,
            )
            self.assertEqual(r_bon["decision"], "allow")
            self.assertEqual(r_bon["overwrite"]["Model"], "carbon")

            # Standard worker requesting frontier (either "pro" or "carbon") is denied with updated wire hint
            for bad_m in ("pro", "carbon", "frontier"):
                r_deny = self._pre("invoke_subagent", self._sub_args("implementer", "Bad Frontier", bad_m), call_id=f"mt_deny_{bad_m}", base=root_p)
                self.assertEqual(r_deny["decision"], "deny")
                self.assertIn(f"Model='{bad_m}'", r_deny["reason"])
                self.assertIn("'hydrogen'", r_deny["reason"])

            # Simple readonly research downgrades to model_tier="lite" even when wire model stays "hydrogen"
            r_res_simple = self._pre("invoke_subagent", self._sub_args("research", "Simple Res", "hydrogen", simple_read_prompt), call_id="mt_9", base=root_p)
            self.assertEqual(r_res_simple, {"decision": "allow"})
            r_res_complex = self._pre("invoke_subagent", self._sub_args("research", "Complex Res", "hydrogen", complex_read_prompt), call_id="mt_10", base=root_p)
            self.assertEqual(r_res_complex, {"decision": "allow"})

            spawns_by_role = {p["role"]: p for p in self._load_st(root_p)["pending_spawns"]}
            self.assertEqual(spawns_by_role["Legacy Scout"]["model_tier"], "lite")
            self.assertEqual(spawns_by_role["Legacy Impl"]["model_tier"], "standard")
            self.assertEqual(spawns_by_role["Hydrogen Scout"]["model_tier"], "lite")
            self.assertEqual(spawns_by_role["Hydrogen Impl"]["model_tier"], "standard")
            self.assertEqual(spawns_by_role["Ver Pro"]["model_tier"], "frontier")
            self.assertEqual(spawns_by_role["Ver Lite"]["model_tier"], "standard")
            self.assertEqual(spawns_by_role["Simple Res"]["model_tier"], "lite")
            self.assertEqual(spawns_by_role["Complex Res"]["model_tier"], "standard")

            # send_message enforces lite-tier gate using model_tier even when wire model is identical ("hydrogen")
            self._reg_sub("scout", "h-scout", "hydrogen", payload=root_p)
            self._reg_sub("implementer", "h-impl", "hydrogen", payload=root_p)
            self._reg_sub("research", "h-res-lite", "hydrogen", payload=root_p, model_tier="lite")
            self._reg_sub("research", "h-res-std", "hydrogen", payload=root_p, model_tier="standard")

            r_msg_scout = self._pre("send_message", {"Recipient": "h-scout", "Message": "Edit src/app.py."}, base=root_p)
            self.assertEqual(r_msg_scout["decision"], "deny")
            self.assertIn("lite-tier ('hydrogen'", r_msg_scout["reason"])

            r_msg_res_lite = self._pre("send_message", {"Recipient": "h-res-lite", "Message": "1. A\n2. B\n3. C\n4. D"}, base=root_p)
            self.assertEqual(r_msg_res_lite["decision"], "deny")

            self.assertEqual(self._pre("send_message", {"Recipient": "h-res-std", "Message": "1. A\n2. B\n3. C\n4. D"}, base=root_p)["decision"], "allow")
            self.assertEqual(self._pre("send_message", {"Recipient": "h-impl", "Message": "Edit src/app.py."}, base=root_p)["decision"], "allow")

            # CLI status & doctor surface active modelTiers
            st_out = orchestrator_guard._cli_status(root_p)
            self.assertEqual(st_out["modelTiers"]["lite"], "hydrogen")
            self.assertEqual(st_out["modelTiers"]["standard"], "hydrogen")
            self.assertEqual(st_out["modelTiers"]["frontier"], "carbon")
            doc_out = orchestrator_guard._cli_doctor()
            self.assertTrue(doc_out["healthy"])
            self.assertEqual(doc_out["modelTiers"]["lite"], "hydrogen")

        # 2. Partial one-to-one remap: only lite -> hydrogen (standard stays flash)
        with mock.patch.dict(os.environ, {"PHILHARMONIC_MODEL_LITE": "hydrogen"}, clear=False):
            r_ver_partial = self._pre("invoke_subagent", self._sub_args("verifier", "Ver Partial", "hydrogen", simple_read_prompt), call_id="mt_partial", base=root_p)
            self.assertEqual(r_ver_partial["decision"], "allow")
            self.assertEqual(r_ver_partial["overwrite"]["Model"], "flash")

        # 3. models.json file overrides + env precedence + doctor validation on malformed config
        with tempfile.TemporaryDirectory() as td:
            cfg_file = Path(td) / "models.json"
            cfg_file.write_text(
                json.dumps({"tiers": {"lite": "hydrogen_lite", "standard": "hydrogen"}, "aliases": {"fast_legacy": "lite"}}),
                encoding="utf-8",
            )
            with mock.patch.object(mt, "MODELS_CONFIG_PATH", cfg_file):
                self.assertEqual(mt.get_model_tier_map()["lite"], "hydrogen_lite")
                self.assertEqual(mt.get_model_tier_map()["standard"], "hydrogen")
                self.assertEqual(mt.resolve_model_tiers("fast_legacy"), {"lite"})
                self.assertEqual(mt.validate_model_config(), [])
                with mock.patch.dict(os.environ, {"PHILHARMONIC_MODEL_STANDARD": "helium"}, clear=False):
                    self.assertEqual(mt.get_model_tier_map()["standard"], "helium")

            cfg_file.write_text(json.dumps({"tiers": {"bogus": "x"}, "aliases": {"bad": "nope"}}), encoding="utf-8")
            with mock.patch.object(mt, "MODELS_CONFIG_PATH", cfg_file):
                doc_bad = orchestrator_guard._cli_doctor()
                self.assertFalse(doc_bad["healthy"])
                self.assertTrue(any("Unknown model tier" in iss for iss in doc_bad["issues"]))
                self.assertTrue(any("Invalid model alias" in iss for iss in doc_bad["issues"]))

    def test_blind_judge_hydrogen_and_dynamic_brand_tokens(self) -> None:
        import os
        from unittest import mock

        root_p = self._brain_payload("root-blind-hydrogen", lastUserInput="/philharmonic blind judge tokens")
        orchestrator_guard._cli_activate(root_p)

        def _judge(prompt: str, cid: str) -> Dict[str, Any]:
            return self._pre(
                "invoke_subagent",
                self._sub_args("judge", "Blind Judge", "inherit", f"MODE: COUNCIL\nCOUNCIL_ID: c-hyd\nCOUNCIL_ROLE: judge\n{prompt}"),
                call_id=cid,
                base=root_p,
            )

        # Standalone hydrogen / carbon / neon / argon leaks must all be caught
        for idx, leak in enumerate((
            "Proposal A (Hydrogen) vs Proposal B.",
            "Draft B by Hydrogen Lite",
            "Option 1: Carbon",
            "Hydrogen wrote Proposal A.",
            "Hydrogen's proposal is cleaner.",
            "Model: Hydrogen",
            "Model: flash_lite",
            "Compare [hydrogen] and [carbon].",
            "Review scratch/proposal_hydrogen.md and scratch/proposal_b.md.",
            "Review scratch/hydrogen_lite.md.",
            "Proposal A\nHydrogen\nUse modular handlers.",
        )):
            res = _judge(leak, f"hyd_leak_{idx}")
            self.assertEqual(res["decision"], "deny", f"Should block leak: {leak!r}")
            self.assertIn("Blind judging violation", res["reason"])

        # Swarm council roster names leaking hydrogen or flash must be denied
        for bad_name in ("proposer-hydrogen", "judge-hydrogen-lite", "proposer-flash"):
            res_sw = self._pre(
                "swarm",
                {"action": "add", "name": bad_name, "brief": "MODE: COUNCIL\nCOUNCIL_ID: c-hyd\nCOUNCIL_ROLE: proposer\nDraft."},
                call_id=f"sw_{bad_name}",
                base=root_p,
            )
            self.assertEqual(res_sw["decision"], "deny", f"Should block swarm name {bad_name!r}")
            self.assertIn("Blind council violation", res_sw["reason"])

        # Benign domain prose mentioning carbon / hydrogen remains allowed
        for idx, ok_text in enumerate((
            "# Proposal B\nCarbon accounting with the Gemini API client.",
            "# Proposal B\nHydrogen fuel cell routing design.",
            "Proposal A: use carbon theme for the dashboard.",
        )):
            res_ok = _judge(ok_text, f"hyd_ok_{idx}")
            self.assertEqual(res_ok["decision"], "allow", f"Should allow benign text: {ok_text!r}")

        # Future unknown codename configured via env var is automatically protected
        with mock.patch.dict(os.environ, {"PHILHARMONIC_MODEL_STANDARD": "gemini-5-helium", "PHILHARMONIC_MODEL_BRAND_TOKENS": "quartz"}, clear=False):
            for idx, dyn_leak in enumerate((
                "Proposal A (Helium) vs Proposal B.",
                "Model: Quartz",
                "Compare [helium] vs [quartz].",
                "Review scratch/proposal_helium.md.",
                "Review scratch/quartz.md.",
            )):
                res_dyn = _judge(dyn_leak, f"dyn_leak_{idx}")
                self.assertEqual(res_dyn["decision"], "deny", f"Should block dynamic leak: {dyn_leak!r}")
            res_sw_dyn = self._pre(
                "swarm",
                {"action": "add", "name": "proposer-helium", "brief": "MODE: COUNCIL\nCOUNCIL_ID: c-hyd\nCOUNCIL_ROLE: proposer\nDraft."},
                call_id="sw_helium",
                base=root_p,
            )
            self.assertEqual(res_sw_dyn["decision"], "deny")


if __name__ == "__main__":
    unittest.main()
