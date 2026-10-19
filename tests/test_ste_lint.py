#!/usr/bin/env python3
"""Unit test suite for ste_lint.py as specified in spec_test_ste_lint.md."""

from pathlib import Path
import sys
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

import ste_lint  # noqa: E402


VALID_PROMPT = """TASK: Update authentication parser.

RULES:
1. Open `src/auth/parser.py`.
2. Reject tokens longer than 4096 bytes with ValueError.
3. Run `pytest tests/test_parser.py` to check the result.

REPORT FORMAT:
STATUS: COMPLETED
FILES_CHANGED: src/auth/parser.py
VERIFICATION: pytest tests/test_parser.py passed with exit 0.
NOTES: Token length check enforced.
"""


def _has_violation(violations, rule_name):
    """Helper to detect a violation rule in list of dicts or strings."""
    for v in violations:
        if isinstance(v, dict):
            if v.get("rule") == rule_name or f"[{rule_name}]" in v.get("message", ""):
                return True
        elif isinstance(v, str):
            if f"[{rule_name}]" in v or rule_name in v:
                return True
    return False


def test_valid_ste100_prompt_passes():
    """A valid controlled-language prompt with numbered rules under 20 words

    returns [] from lint_prompt and None from check_subagent_call.
    """
    violations = ste_lint.lint_prompt(VALID_PROMPT)
    assert violations == []

    invoke_call = {
        "Subagents": [
            {
                "TypeName": "implementer",
                "Role": "Token Worker",
                "Prompt": VALID_PROMPT,
            }
        ]
    }
    assert ste_lint.check_subagent_call("invoke_subagent", invoke_call) is None

    send_call = {
        "Recipient": "subagent-1234",
        "Message": VALID_PROMPT,
    }
    assert ste_lint.check_subagent_call("send_message", send_call) is None


def test_long_sentence_violation():
    """A numbered rule with > 20 words returns a [long-sentence] violation,

    while a numbered rule with long inline code collapses the inline code to 1 token and passes.
    """
    prompt_too_long = (
        "TASK: Test long rule.\n\n"
        "RULES:\n"
        "1. This is a very long numbered rule that intentionally contains way more than "
        "twenty English words in order to trigger the rule violation.\n"
    )
    violations_long = ste_lint.lint_prompt(prompt_too_long)
    assert _has_violation(violations_long, "long-sentence")

    prompt_with_code = (
        "TASK: Test inline code collapsing.\n\n"
        "RULES:\n"
        "1. Inspect `very_long_code_identifier_with_many_words` to verify user access "
        "and return the token immediately.\n"
    )
    violations_code = ste_lint.lint_prompt(prompt_with_code)
    assert not _has_violation(violations_code, "long-sentence")


def test_fenced_code_block_ignored():
    """Violations inside triple-backtick fenced code blocks are ignored by lint_prompt."""
    prompt_fenced = (
        "TASK: Check code snippet.\n\n"
        "RULES:\n"
        "1. Inspect the example below.\n\n"
        "```python\n"
        "# Semicolons; figure out; please note; as well as; perform verification\n"
        "x = 1; y = 2;\n"
        "```\n\n"
        "2. Execute the test suite.\n"
    )
    violations = ste_lint.lint_prompt(prompt_fenced)
    assert not _has_violation(violations, "semicolon")
    assert not _has_violation(violations, "phrasal-verb")
    assert not _has_violation(violations, "filler")
    assert not _has_violation(violations, "nominalization")
    assert violations == []


def test_semicolon_violation():
    """A numbered rule containing ; triggers [semicolon],

    whereas a semicolon inside inline backticks does not trigger [semicolon].
    """
    prompt_semi = (
        "TASK: Test semicolon.\n\n"
        "RULES:\n"
        "1. Open `src/file.py`; read contents.\n"
    )
    violations_semi = ste_lint.lint_prompt(prompt_semi)
    assert _has_violation(violations_semi, "semicolon")

    prompt_semi_inline = (
        "TASK: Test semicolon in backticks.\n\n"
        "RULES:\n"
        "1. Split the string on `;` delimiter.\n"
    )
    violations_inline = ste_lint.lint_prompt(prompt_semi_inline)
    assert not _has_violation(violations_inline, "semicolon")


def test_phrasal_verb_violations():
    """Rules containing banned phrasal verbs trigger [phrasal-verb]."""
    banned_phrasals = [
        "figure out",
        "look into",
        "set up",
        "clean up",
        "make sure",
    ]
    for phrase in banned_phrasals:
        prompt = (
            "TASK: Test phrasal verb.\n\n"
            "RULES:\n"
            f"1. Please {phrase} the current bug.\n"
        )
        violations = ste_lint.lint_prompt(prompt)
        assert _has_violation(violations, "phrasal-verb"), f"Expected phrasal-verb for '{phrase}'"


def test_filler_phrase_violations():
    """Rules containing banned filler phrases trigger [filler]."""
    banned_fillers = [
        "in order to",
        "please note",
        "as well as",
        "be able to",
    ]
    for filler in banned_fillers:
        prompt = (
            "TASK: Test filler phrase.\n\n"
            "RULES:\n"
            f"1. Run tests {filler} verify status.\n"
        )
        violations = ste_lint.lint_prompt(prompt)
        assert _has_violation(violations, "filler"), f"Expected filler for '{filler}'"


def test_nominalization_violations():
    """Rules containing banned nominalizations trigger [nominalization]

    and include the suggested imperative verb.
    """
    cases = [
        ("implementation", "implement"),
        ("configuration", "configure"),
        ("verification", "verify"),
        ("investigation", "investigate"),
    ]
    for noun, suggested_verb in cases:
        prompt = (
            "TASK: Test nominalization.\n\n"
            "RULES:\n"
            f"1. Complete the {noun} now.\n"
        )
        violations = ste_lint.lint_prompt(prompt)
        assert _has_violation(violations, "nominalization"), f"Expected nominalization for '{noun}'"

        nom_violations = [
            v for v in violations
            if (isinstance(v, dict) and (v.get("rule") == "nominalization" or "[nominalization]" in v.get("message", "")))
            or (isinstance(v, str) and ("nominalization" in v))
        ]
        matched_suggestion = any(
            suggested_verb in (v.get("message", "") if isinstance(v, dict) else v).lower()
            for v in nom_violations
        )
        assert matched_suggestion, f"Expected suggested verb '{suggested_verb}' for nominalization '{noun}'"


def test_synonym_rotation_violations():
    """A prompt using two words from the same synonym group triggers [synonym-rotation]."""
    synonym_pairs = [
        ("modify", "edit"),
        ("verify", "check"),
        ("remove", "delete"),
    ]
    for w1, w2 in synonym_pairs:
        prompt = (
            "TASK: Test synonym rotation.\n\n"
            "RULES:\n"
            f"1. First {w1} the input config.\n"
            f"2. Then {w2} the target file.\n"
        )
        violations = ste_lint.lint_prompt(prompt)
        assert _has_violation(violations, "synonym-rotation"), f"Expected synonym-rotation for '{w1}' and '{w2}'"


def test_check_subagent_call_invoke_and_send_message():
    """Verify check_subagent_call checks all entries in Subagents for invoke_subagent

    and Message for send_message, returning formatted remediation text on violations
    and None on valid prompts or unrelated tool names.
    """
    assert ste_lint.check_subagent_call("run_command", {"CommandLine": "git status"}) is None
    assert ste_lint.check_subagent_call("write_to_file", {"TargetFile": "a.txt"}) is None

    valid_invoke = {
        "Subagents": [
            {"TypeName": "implementer", "Role": "Worker 1", "Prompt": VALID_PROMPT},
            {"TypeName": "scout", "Role": "Worker 2", "Prompt": VALID_PROMPT},
        ]
    }
    assert ste_lint.check_subagent_call("invoke_subagent", valid_invoke) is None

    valid_send = {
        "Recipient": "sub-100",
        "Message": VALID_PROMPT,
    }
    assert ste_lint.check_subagent_call("send_message", valid_send) is None

    bad_prompt_1 = (
        "TASK: Bad subagent prompt.\n\n"
        "RULES:\n"
        "1. Open `src/file.py`; run verification.\n"
    )
    bad_invoke_first = {
        "Subagents": [
            {"TypeName": "implementer", "Role": "Worker 1", "Prompt": bad_prompt_1},
        ]
    }
    res_bad_1 = ste_lint.check_subagent_call("invoke_subagent", bad_invoke_first)
    assert res_bad_1 is not None
    assert isinstance(res_bad_1, str)
    assert len(res_bad_1) > 0

    bad_invoke_second = {
        "Subagents": [
            {"TypeName": "scout", "Role": "Worker 1", "Prompt": VALID_PROMPT},
            {"TypeName": "implementer", "Role": "Worker 2", "Prompt": bad_prompt_1},
        ]
    }
    res_bad_2 = ste_lint.check_subagent_call("invoke_subagent", bad_invoke_second)
    assert res_bad_2 is not None
    assert isinstance(res_bad_2, str)
    assert len(res_bad_2) > 0

    bad_send = {
        "Recipient": "sub-200",
        "Message": bad_prompt_1,
    }
    res_bad_send = ste_lint.check_subagent_call("send_message", bad_send)
    assert res_bad_send is not None
    assert isinstance(res_bad_send, str)
    assert len(res_bad_send) > 0
