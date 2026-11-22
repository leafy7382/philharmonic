"""Constants, path anchors, regex patterns, and shell/command helpers for Philharmonic guard."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent.parent

ALLOWED_SUBAGENT_MODELS = {"flash", "flash_lite"}
STRICT_READONLY_SUBAGENT_TYPES = {"scout", "verifier", "reviewer", "judge"}
HIGH_REASONING_READONLY_TYPES = {"verifier", "reviewer", "judge"}
SCRATCH_ONLY_SUBAGENT_TYPES = {"research", "researcher", "council"}
READONLY_NON_VERIFICATION_TYPES = {
    "scout",
    "research",
    "researcher",
    "council",
    "verifier",
    "reviewer",
    "judge",
}
ALLOWED_VERIFIER_REVIEWER_MODELS = {"flash", "pro", "inherit"}
ALLOWED_COUNCIL_PROPOSER_MODELS = {"flash", "pro", "inherit"}
ISOLATED_WORKSPACES = {"branch", "share"}
DEFAULT_MODEL_BY_TYPE = {
    "scout": "flash_lite",
    "research": "flash",
    "researcher": "flash",
    "council": "flash",
    "implementer": "flash",
    "verifier": "flash",
    "reviewer": "flash",
    "judge": "flash",
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
_MODEL_BRAND_PATTERN = (
    r"(?:gemini(?:\s+[0-9]+(?:\.[0-9]+)?)?(?:\s+(?:pro|flash(?:_lite)?|carbon|neon|argon|hydrogen))?"
    r"|claude(?:\s+[0-9]+(?:\.[0-9]+)?)?(?:\s+(?:opus|sonnet|haiku|fable)(?:\s+[0-9]+(?:\.[0-9]+)?)?)?"
    r"|opus(?:\s+[0-9]+(?:\.[0-9]+)?)?|sonnet(?:\s+[0-9]+(?:\.[0-9]+)?)?"
    r"|haiku(?:\s+[0-9]+(?:\.[0-9]+)?)?|fable(?:\s+[0-9]+(?:\.[0-9]+)?)?"
    r"|gpt-[0-9][a-z0-9.\-]*|o[134](?:[- ](?:mini|preview|pro))?"
    r"|grok(?:[- ][0-9]+(?:\.[0-9]+)?)?"
    r"|chatgpt|geminipro|claudesonnet|claudeopus"
    r"|openai|anthropic|deepseek|llama|qwen|mistral)"
)
COUNCIL_NAME_LEAK_RE = re.compile(
    r"(?:^|[^a-z])(?:gemini|claude|opus|sonnet|haiku|gpt|chatgpt|grok|o[134]|openai|anthropic|deepseek|llama|qwen|mistral|carbon|neon|argon|hydrogen|fable)(?:$|[^a-z])",
    re.I,
)
BLIND_JUDGE_LEAK_PATTERNS = (
    re.compile(
        rf"\b(?:proposal|draft|option|candidate|design|submission|response|answer)\s+[a-z0-9#_-]{{1,20}}[^\r\n]{{0,80}}(?:\(\s*|:\s*|—\s*|-\s*|\bby\s+|\bfrom\s+){_MODEL_BRAND_PATTERN}\b",
        re.I,
    ),
    re.compile(
        rf"\b(?:proposal|draft|option|candidate|design|submission|response|answer)\s+[a-z0-9#_-]{{1,20}}.{{0,60}}?(?:\(\s*|:\s*|—\s*|-\s*|\bby\s+|\bfrom\s+|\r?\n[ \t]*)\b{_MODEL_BRAND_PATTERN}\b",
        re.I | re.S,
    ),
    re.compile(
        rf"\b{_MODEL_BRAND_PATTERN}\b[^\r\n]{{0,60}}\b(?:wrote|authored|proposed|generated|created|produced|submitted)\b[^\r\n]{{0,60}}\b(?:proposal|draft|option|candidate|design|submission|response|answer)\b",
        re.I,
    ),
    re.compile(
        rf"\b{_MODEL_BRAND_PATTERN}(?:'s|s')\s+(?:proposal|draft|option|candidate|design|submission|response|answer)\b",
        re.I,
    ),
    re.compile(
        rf"(?m)(?:^[ \t]*(?:[-*#]+[ \t]*)?|(?:^|[{{,])[ \t]*)[\"']?(?:model|author|authored[ \t]+by|written[ \t]+by|generated[ \t]+by|proposed[ \t]+by|submitted[ \t]+by|source[ \t]+model|provider|llm)[\"']?[ \t]*:[ \t]*[^\r\n]*\b{_MODEL_BRAND_PATTERN}\b",
        re.I,
    ),
    re.compile(
        rf"\[\s*(?:{_MODEL_BRAND_PATTERN}|flash(?:_lite)?(?:\s+[0-9]+(?:\.[0-9]+)?)?|gpt)\s*\]",
        re.I,
    ),
    re.compile(
        r"\b(?:(?:proposal|draft|option|candidate|design|solution|submission|response|answer|report|output|plan)[a-z0-9._/\-]*?(?:gemini|claude|opus|sonnet|haiku|fable|flash|gpt(?:-[0-9a-z.]*)?|openai|anthropic|deepseek|llama|qwen|mistral)|(?:gemini|claude|opus|sonnet|haiku|fable|flash|gpt(?:-[0-9a-z.]*)?|openai|anthropic|deepseek|llama|qwen|mistral)[a-z0-9._/\-]*?(?:proposal|draft|option|candidate|design|solution|submission|response|answer|report|output|plan))[a-z0-9._/\-]*\.(?:md|txt|json)\b",
        re.I,
    ),
    re.compile(
        r"(?:^|[/\\\s`'\"(,:])(?:gemini|claude|opus|sonnet|haiku|fable|flash(?:_lite)?|gpt(?:-[0-9a-z.]*|[0-9]+[a-z0-9._-]*)?|o[134]|grok|chatgpt|openai|anthropic|deepseek|llama|qwen|mistral)(?:[_-][a-z0-9._-]*)?\.(?:md|txt|json)\b",
        re.I,
    ),
    re.compile(
        r"(?:^|[\s`'\"(,:])(?:[A-Za-z0-9._/\-]*/)?scratch/(?![A-Za-z0-9._\-]+\.(?:md|txt|json)\b)",
        re.I,
    ),
)
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
MUTATION_TOOLS = {
    "write_file",
    "edit_file",
    "write_to_file",
    "replace_file_content",
    "multi_replace_file_content",
}
PATH_ARG_KEYS = (
    "Path",
    "TargetFile",
    "AbsolutePath",
    "FilePath",
    "file_path",
    "SearchPath",
    "SearchDirectory",
    "DirectoryPath",
)

FORBIDDEN_DIRECT_READ_RE = re.compile(
    r"(?:^|/)transcript(?:_full|[0-9a-zA-Z_\-]*)?\.jsonl$|\.log$", re.I
)

SAFE_GIT_ENV_PREFIX_STR = (
    r"(?:(?:"
    r"(?:GIT_EDITOR|GIT_SEQUENCE_EDITOR)=(?:['\"]?(?:true|/bin/true|/usr/bin/true|:)['\"]?)"
    r"|(?:GIT_PAGER|PAGER)=(?:['\"]?(?:cat|/bin/cat|/usr/bin/cat)['\"]?|\"\"|'')"
    r"|GIT_TERMINAL_PROMPT=(?:['\"]?[01]['\"]?)"
    r"|(?:LC_ALL|LANG)=(?:['\"]?(?:C|C\.UTF-8|en_US\.UTF-8)['\"]?)"
    r")\s+)*"
)
_SAFE_GIT_PREFIX_STR = (
    r"(?:/(?:usr/)?bin/)?git\b"
    r"(?:\s+(?:-C\s+\S+|--git-dir(?:=\S+|\s+\S+)|--work-tree(?:=\S+|\s+\S+)|--no-pager|--paginate|-[pP]|--bare|--no-replace-objects|--literal-pathspecs|--glob-pathspecs|--noglob-pathspecs|--icase-pathspecs|--no-optional-locks))*"
)
_GIT_CMD_PREFIX_STR = (
    r"(?:/(?:usr/)?bin/)?git\b"
    r"(?:\s+(?:-C\s+\S+|--git-dir(?:=\S+|\s+\S+)|--work-tree(?:=\S+|\s+\S+)|-c\s+\S+|--namespace(?:=\S+|\s+\S+)|--super-prefix(?:=\S+|\s+\S+)|--attr-source(?:=\S+|\s+\S+)|--no-pager|--paginate|-[pP]|--bare|--no-replace-objects|--literal-pathspecs|--glob-pathspecs|--noglob-pathspecs|--icase-pathspecs|--no-optional-locks))*"
)
GIT_COMMAND_RE = re.compile(rf"^\s*{SAFE_GIT_ENV_PREFIX_STR}{_GIT_CMD_PREFIX_STR}")
SAFE_GIT_INSPECTION_RE = re.compile(
    rf"^\s*{SAFE_GIT_ENV_PREFIX_STR}{_SAFE_GIT_PREFIX_STR}\s+(?:status|diff|log)\b"
    r"(?![\s\S]*(?:--output(?:=|\s)|--ext-diff\b|--exec-path(?:=|\s)|--config-env(?:=|\s)|--no-index\b|--upload-pack(?:=|\s)|--receive-pack(?:=|\s)|--open-files-in-pager(?:=|\s|\b)|\bext::))"
    r"(?!(?:[^'\"\\]|\\.|'[^']*'|\"(?:\\.|[^\"\\])*\")*(?<!<)(?:[0-9]*>(?!&\s*(?:\d+|-)(?=$|[\s;|&)><]))|&>|>>|>\|))"
)
_MUTATING_GIT_SUBCMD_BODY_STR = (
    r"(?:(?:checkout|switch|reset|apply|am|merge|merge-file|rebase|cherry-pick|clean|commit|add|rm|mv|restore|stash|revert|checkout-index|config|clone|pull|fetch|push|init|worktree|submodule|replace|notes|bisect|bundle|format-patch|update-ref|update-index|read-tree|write-tree|commit-tree|mktree|mktag|hash-object|fast-import|fast-export|backfill|gui|citool|gitk|web--browse|filter-branch|gc|prune|prune-packed|repack|rerere|pack-refs|maintenance|sparse-checkout|difftool|mergetool|send-email|credential|credential-cache|credential-store|daemon|svn|p4|interpret-trailers|stage|symbolic-ref|receive-pack|unpack-objects|unpack-file|index-pack|pack-objects|commit-graph|multi-pack-index|bugreport|diagnose|fetch-pack|send-pack|replay|hook|instaweb|http-fetch|http-push|shell|remote-ext|remote-fd|mailinfo|mailsplit|update-server-info|scalar|quiltimport|imap-send|cvsexportcommit|cvsimport|archimport)\b"
    r"|fsck\b[^\n;|&]*--lost-found\b"
    r"|remote\b\s+(?:(?:-v|--verbose)\s+)?(?:add|remove|rm|rename|set-url|set-head|set-branches|prune|update)\b"
    r"|reflog\b\s+(?:expire|delete|drop)\b"
    r"|refs\b\s+migrate\b"
    r"|tag\b(?!\s*(?:$|(?:(?-i:-[lv])|--list|--verify)\b))(?:(?:\s+\S+)*\s+(?:(?-i:-[a-zA-Z]*[dasufmF])(?=\s|$)|--(?:delete|annotate|sign|force|local-user|message|file)(?:=|\s|$))|\s+(?!-)\S+)"
    r"|archive\b(?:\s+\S+)*\s+(?:(?-i:-[a-zA-Z]*o)|--(?:output|exec|remote)(?:=|\s|$))"
    r"|branch\b(?:(?:\s+\S+)*\s+(?:(?-i:-[a-zA-Z]*[dDmMcCftu])(?=\s|$)|--(?:delete|move|copy|force|set-upstream-to|unset-upstream|edit-description|create-reflog|track|no-track)(?:=|\s|$))|\s+(?!-)\S+))"
)
MUTATING_GIT_SUBCMD_RE = re.compile(
    r"^\s*"
    + SAFE_GIT_ENV_PREFIX_STR
    + _GIT_CMD_PREFIX_STR
    + r"\s+"
    + _MUTATING_GIT_SUBCMD_BODY_STR,
    re.I,
)
_SHELL_DQ_STR = r'"(?:\\.|[^"\\])*"'
_SHELL_SQ_STR = r"(?:\$'(?:\\.|[^'\\])*'|'[^']*')"
_SHELL_ARG_SPAN_STR = rf"(?:{_SHELL_SQ_STR}|{_SHELL_DQ_STR}|\\.|[^\n;|&'\"\\])*"


def _in_cmd_arg_pattern(op_pattern: str) -> str:
    """Build a regex fragment matching op_pattern inside shell quotes (including multiline) or unquoted args."""
    return (
        r"(?:"
        + _SHELL_ARG_SPAN_STR
        + r"(?:'[^']*"
        + op_pattern
        + r"[^']*'"
        + r"|\$'(?:\\.|[^'\\])*"
        + op_pattern
        + r"(?:\\.|[^'\\])*'"
        + r'|"(?:\\.|[^"\\])*'
        + op_pattern
        + r'(?:\\.|[^"\\])*"'
        + r"|"
        + op_pattern
        + r"))"
    )


_PYTHON_MUTATING_OPS_STR = (
    r"(?:open\s*\([^)]*['\"][^'\"]*[wax+]"
    r"|open\s*\([^,)]+,\s*(?!['\"][rUtb]+['\"]\s*[,)])"
    r"|\.\s*(?:write|writelines|truncate|write_text|write_bytes|unlink|rename|replace|mkdir|rmdir|chmod|touch)\s*\("
    r"|write_text|write_bytes|unlink|remove"
    r"|removedirs|rmdir|rename|replace|rmtree|symlink|link|truncate"
    r"|chmod|chown|mkdir|makedirs|copyfile|copy2|copytree|move|shutil"
    r"|pathlib|os\.|subprocess|ctypes|exec\s*\(|eval\s*\(|__import__"
    r"|getattr|setattr|builtins|codecs\.open|io\.open)"
)
_PY_INTERP_NAME_STR = r"(?:python(?:\d+(?:\.\d+)*)?|pypy(?:\d+(?:\.\d+)*)?)"
_NON_PY_INTERP_NAME_STR = r"(?:perl|node|ruby|php|lua)(?:\d+(?:\.\d+)*)?"
_INTERP_BIN_STR = rf"(?:(?:/(?:usr/)?bin/)?(?:{_PY_INTERP_NAME_STR}|{_NON_PY_INTERP_NAME_STR}))"
_NON_PY_INTERP_MUT_STR = (
    rf"\b(?:(?:/(?:usr/)?bin/)?{_NON_PY_INTERP_NAME_STR})\b"
    r"(?:\s+(?:--[a-z-]+(?:=\S+)?|-(?!v\b|V\b|h\b)[a-zA-Z]+))*"
    r"\s+(?!(?:--version|--help|-v|-V|-h)(?:\s|$))(?![;&|])[^\s;&|]+"
)
MUTATING_SHELL_CMD_RE = re.compile(
    r"(?:"
    r"\b(?:rm|unlink|shred|mv|cp|tee|touch|truncate|dd|patch|install|ln|chmod|chown|chgrp|mkdir|rmdir|rsync|split|csplit|link|mktemp|mkfifo|mknod|wget|dos2unix|unix2dos|fallocate|sponge)\b"
    r"|\biconv\s+(?:" + _SHELL_ARG_SPAN_STR + r"\s)?(?:(?-i:-[a-zA-Z]*o)|--output(?:=|\s|$))"
    r"|\bcurl\s+(?:" + _SHELL_ARG_SPAN_STR + r"\s)?(?:(?-i:-[a-zA-Z]*[oOKDc])|--(?:output|remote-name-all|remote-name|config|dump-header|cookie-jar|trace|trace-ascii|stderr|alt-svc|hsts|etag-save|libcurl)(?:=|\s|$)|(?:(?-i:-[a-zA-Z]*w)|--write-out)(?:=|\s)[^\n;|&]*%output\{)"
    r"|\b(?:sed|perl(?:\d+(?:\.\d+)*)?)\s+(?:" + _SHELL_ARG_SPAN_STR + r"\s)?(?:-[^\s]*i|--in-place(?:=\S+)?)\b"
    r"|\bsed\s+(?:" + _SHELL_ARG_SPAN_STR + r"\s)?(?:(?-i:-[a-zA-Z]*f\b)|--file(?:=\S+|\s+\S+))"
    r"|\bperl(?:\d+(?:\.\d+)*)?\s+(?:" + _SHELL_ARG_SPAN_STR + r"\s)?(?:-[a-zA-Z]*[eE]\b|--eval\b)"
    r"|\bfind\b" + _SHELL_ARG_SPAN_STR + r"-(?:delete|fprint\w*|fls)\b"
    r"|\b" + _PY_INTERP_NAME_STR + r"\b" + _in_cmd_arg_pattern(_PYTHON_MUTATING_OPS_STR)
    + r"|\b" + _PY_INTERP_NAME_STR + r"\b" + _in_cmd_arg_pattern(r"(?:/tmp(?:/|\b)|\S*scratch(?:/|\b)|/brain(?:/|\b))[^\s;|&'\"]*")
    + r"|" + _NON_PY_INTERP_MUT_STR
    + r"|\b[gmn]?awk\b" + _in_cmd_arg_pattern(r"(?:system\s*\(|(?:print|printf)\b[^';&|]*(?:>|>>|\|&?)|\|&?\s*getline\b|@(?:include|load)\b)")
    + r"|\b[gmn]?awk\s+(?:" + _SHELL_ARG_SPAN_STR + r"\s)?(?:(?-i:-[a-zA-Z]*[ilfEdopD])|(?-i:-W)\s*(?:exec|file|include|load|dump-variables|pretty-print|profile|source)\b|--(?:include|load|file|exec|dump-variables|pretty-print|profile|debug)(?:=\S+|\s+\S+|\b))"
    r"|\bsort\s+(?:" + _SHELL_ARG_SPAN_STR + r"\s)?(?:(?-i:-[a-zA-Z]*o)|--(?:output|compress-program)(?:=\S+|\s+\S+))"
    r"|\b(?:/(?:usr/)?bin/)?time\s+(?:" + _SHELL_ARG_SPAN_STR + r"\s)?(?:(?-i:-[a-zA-Z]*o)|--output(?:=\S+|\s+\S+))"
    r"|\btar\s+(?:(?-i:[xcruAF][a-zA-Z]*)(?=\s)|(?:" + _SHELL_ARG_SPAN_STR + r"\s)?(?:(?-i:-[a-zA-Z]*[xcruIAF])|--(?:create|extract|get|append|update|catenate|concatenate|index-file|info-script|new-volume-script|rmt-command|use-compress-program|checkpoint-action|to-command|rsh-command)(?:=|\s|\b)))"
    r"|\bunzip\b(?!(?:\s+(?:(?-i:-[a-zA-Z]*[lZ]\b)|--help\b|--version\b)))"
    r"|\brg\s+(?:" + _SHELL_ARG_SPAN_STR + r"\s)?--pre(?:=|\s)\S+"
    r"|\b(?:/(?:usr/)?bin/)?git\s+(?:" + _SHELL_ARG_SPAN_STR + r"\s)?(?:(?-i:-c\s+\S+)|--(?:config-env|exec-path|output|ext-diff|upload-pack|receive-pack|open-files-in-pager)(?:=|\s|\b))"
    r"|\b(?:/(?:usr/)?bin/)?git\b[^\n;|&]*\bext::"
    r"|\b(?:/(?:usr/)?bin/)?git\b[^\n;|&]*\bgrep\b[^\n;|&]*(?-i:-[a-zA-Z]*O\b|--open-files-in-pager)"
    r"|^\s*"
    + SAFE_GIT_ENV_PREFIX_STR
    + _GIT_CMD_PREFIX_STR
    + r"\s+"
    + _MUTATING_GIT_SUBCMD_BODY_STR
    + r")",
    re.I,
)
_SHELL_CMD_BOUNDARY_STR = r"(?:^|[;&|\n({)]|\$\(|&&|\|\||\|&|;;|`)"
_SHELL_BIN_STR = r"(?:(?:/(?:usr/)?bin/)?(?:(?:ba|da|z|k|a|c|tc|mk|rba)?sh|fish))"
_SHELL_PRE_C_FLAGS_STR = r"(?:\s+(?:--[a-z-]+(?:=\S+)?|-o\s+\S+|-(?!c\b)[a-zA-Z]+)){0,6}"
_SHELL_ONE_PREFIX_WRAPPER_STR = (
    r"(?:"
    r"(?:/(?:usr/)?bin/)?env\s+(?:(?!(?:(?:/(?:usr/)?bin/)?(?:(?:ba|da|z|k|a|c|tc|mk|rba)?sh|fish)|eval|source|timeout|nice|nohup|stdbuf|command|exec|time|builtin|xargs|find|setsid|chrt|ionice|taskset|flock|script|prlimit|setarch|linux32|linux64|strace|ltrace|numactl|unshare|nsenter|chroot|watch|parallel|uv|poetry|pipenv|npx|bunx|ed|ex|uvx)\b)\S+\s+){0,8}"
    r"|(?:/(?:usr/)?bin/)?timeout\s+(?:(?:--[a-z-]+(?:=\S+)?|-[a-zA-Z]*[sk]\s*\S+|-[a-zA-Z]+)\s+){0,4}(?:\+?(?:(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|0[xX](?:[0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?|\.[0-9a-fA-F]+)(?:[pP][+-]?[0-9]+)?|(?i:inf(?:inity)?))[smhd]?)\s+"
    r"|(?:/(?:usr/)?bin/)?nice\s+(?:(?:-n\s*-?[0-9]+|--adjustment=-?[0-9]+|-[0-9]+)\s+)?"
    r"|(?:/(?:usr/)?bin/)?stdbuf\s+(?:(?:-[ioe](?:\s+\S+|\S+)|--(?:input|output|error)(?:=\S+|\s+\S+))\s+){1,3}"
    r"|(?:/(?:usr/)?bin/)?setsid\s+(?:(?:--[a-z-]+|-[a-zA-Z]+)\s+){0,4}"
    r"|(?:/(?:usr/)?bin/)?chrt\s+(?:(?:--[a-z-]+(?:=\S+)?|(?-i:-[a-zA-Z]*[TPD])\s*\S+|-[a-zA-Z]+(?:\s+[0-9]+)?)\s+){0,4}(?:[0-9]+\s+)?"
    r"|(?:/(?:usr/)?bin/)?ionice\s+(?:(?:--[a-z-]+(?:=\S+)?|(?-i:-[a-zA-Z]*[cnpPu])\s*\S+|-[a-zA-Z]+)\s+){0,4}"
    r"|(?:/(?:usr/)?bin/)?taskset\s+(?:(?:--(?:cpu-list|pid)\s+\S+|--[a-z-]+(?:=\S+)?|-[a-zA-Z]*[cp]\s*\S+|-[a-zA-Z]+)\s+){0,4}(?:(?:0[xX])?[0-9a-fA-F]+\s+)?"
    r"|(?:nohup|command|time|builtin)\s+(?:-(?!v\b|V\b)[a-zA-Z]+\s+){0,2}"
    r"|exec\s+(?:(?:-[a-zA-Z]*a\s*\S+|-(?!v\b|V\b)[a-zA-Z]+)\s+){0,3}"
    r"|(?:(?:uv|poetry|pipenv)\s+(?:--[a-z-]+(?:=\S+)?\s+){0,3}run|(?:npx|bunx)(?:\s+(?:-[yq]+|--(?:yes|quiet|no|no-install)))*)\s+"
    r")"
)
_SHELL_PREFIX_WRAPPERS_STR = (
    r"(?:(?:[A-Za-z_][A-Za-z0-9_]*\+?=\S+\s+)*"
    + _SHELL_ONE_PREFIX_WRAPPER_STR
    + r"){0,4}"
    + r"(?:[A-Za-z_][A-Za-z0-9_]*\+?=\S+\s+)*"
)
SHELL_EXEC_WRAPPER_RE = re.compile(
    r"(?:"
    + _SHELL_CMD_BOUNDARY_STR
    + r"\s*"
    + _SHELL_PREFIX_WRAPPERS_STR
    + r"(?:"
    + _SHELL_BIN_STR
    + r"(?:"
    + _SHELL_PRE_C_FLAGS_STR
    + r"(?:\s+(?:-[a-zA-Z]*c\b|--command\b|-[a-zA-Z]*s\b|-(?=\s|$))|\s*<<<|\s*<(?!=))"
    + r"|"
    + _SHELL_PRE_C_FLAGS_STR
    + r"\s+(?:--\s+)?(?![-;&|><])[^\s;&|><]+"
    + r")"
    + r"|(?:eval|source)(?:\s|$)|\.\s+[^\s;&|><]+"
    + r"|(?:(?:/(?:usr/)?bin/)?xargs)\b(?:\s+(?:-[ILnPdEa](?:\s+\S+|\S*)|-[^\s;|&]+(?:\s+(?:[0-9]+|\{\}|\\n|" + _SHELL_SQ_STR + r"|" + _SHELL_DQ_STR + r"))?|--[a-z-]+(?:=\S+)?)){0,10}\s+"
    + r"(?:(?:/(?:usr/)?bin/)?(?:env\s+(?:\S+\s+){0,4})?)?"
    + r"(?:(?:/(?:usr/)?bin/)?(?:(?:(?:ba|da|z|k|a|c|tc|mk|rba)?sh|fish)|eval|source|rm|unlink|shred|mv|cp|tee|touch|truncate|dd|patch|install|ln|chmod|chown|chgrp|mkdir|rmdir|sed|" + _PY_INTERP_NAME_STR + r"|" + _NON_PY_INTERP_NAME_STR + r"|[gmn]?awk|git)\b|\.?/[^\s;&|]+)"
    + r"|(?:(?:/(?:usr/)?bin/)?find)\b" + _SHELL_ARG_SPAN_STR + r"-(?:exec|execdir|ok|okdir)\b"
    + r"|" + _INTERP_BIN_STR + r"\b(?:\s+(?:--[a-z-]+(?:=\S+)?|-[a-zA-Z]+)){0,4}(?:\s+-(?=\s|$)|\s*<<<|\s*<(?!=))"
    + r"|(?:(?:/(?:usr/)?bin/)?(?:flock|script|prlimit|setarch|linux32|linux64|strace|ltrace|numactl|unshare|nsenter|chroot|watch|parallel|ed|ex|uvx|bun|deno|expect|tclsh|wish|Rscript|julia|pwsh|busybox|sqlite3|make|cmake|pip|pip3))\b"
    + r")"
    + r"|(?<!\|)\|&?(?!\|)\s*"
    + _SHELL_PREFIX_WRAPPERS_STR
    + r"(?:" + _SHELL_BIN_STR + r"|" + _INTERP_BIN_STR + r")\b"
    + r"|(?:\(\s*|\{\s+|\$\(\s*|`)+"
    + r"\s*"
    + _SHELL_PREFIX_WRAPPERS_STR
    + _SHELL_BIN_STR
    + r"\b"
    + r"|(?:(?:"
    + _SHELL_CMD_BOUNDARY_STR
    + r"|(?<!\|)\|&?(?!\|))\s*(?:[A-Za-z_][A-Za-z0-9_]*\+?=\S+\s+)*(?:/(?:usr/)?bin/)?env\s+(?:\S+\s+){0,4}(?:(?-i:-[a-zA-Z]*S\b)|--split-string(?:=|\b)))"
    + r")",
    re.I,
)
_SHELL_MASKING_TAIL_RE = re.compile(
    r"^\s*(?:[A-Za-z_][A-Za-z0-9_]*\+?=\S+\s+)*(?:(?:(?:/(?:usr/)?bin/)?(?:true|echo|printf|cat|pwd|ls|sleep)|exit(?:\s+0)?)\b|:(?=$|\s))",
    re.I,
)
SANDBOX_BOUNDARY_ERROR_RE = re.compile(
    r"(?:There were sandbox errors that may or may not be related to the failure"
    r"|\.git/[A-Za-z0-9._/\-]*['\"]?:\s*Read-only file system"
    r"|Unable to create ['\"]?[^'\"]*\.git/[^'\"]*\.lock['\"]?:\s*Read-only file system"
    r"|cannot open ['\"]?[^'\"]*\.git/[A-Za-z0-9._/\-]*['\"]?:\s*Read-only file system)",
    re.I,
)


def _is_sandbox_boundary_error(
    payload: Dict[str, Any], error_text: str, output_text: str
) -> bool:
    """Return True if the error is an environment/sandbox restriction rather than a genuine code/command failure."""
    if payload.get("sandboxError") is True:
        return True
    if error_text and bool(SANDBOX_BOUNDARY_ERROR_RE.search(error_text)):
        return True
    if output_text and bool(SANDBOX_BOUNDARY_ERROR_RE.search(output_text)):
        return True
    return False
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
TEST_DEF_RE = re.compile(
    r"(?m)^\s*(?:async\s+def|def)\s+test_[A-Za-z0-9_]+\s*\(|^\s*(?:it|test)\s*\("
)
ASSERTION_RE = re.compile(
    r"\b(?:assert\b|self\.assert[A-Za-z0-9_]*\s*\(|expect\s*\(|should\.)"
)
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
    r'/\*[\s\S]*?\*/'
    r'|"""[\s\S]*?"""'
    r"|'''[\s\S]*?'''"
    r'|"(?:\\.|[^"\\\n])*"'
    r"|'(?:\\.|[^'\\\n])*'"
    r"|#(?!\s*\[ignore\b)[^\n]*"
    r"|^\s*//[^\n]*",
    re.M,
)
_STRIP_NON_CODE_C_STYLE_RE = re.compile(
    r'/\*[\s\S]*?\*/'
    r'|"""[\s\S]*?"""'
    r"|'''[\s\S]*?'''"
    r'|"(?:\\.|[^"\\\n])*"'
    r"|'(?:\\.|[^'\\\n])*'"
    r"|#(?!\s*\[ignore\b)[^\n]*"
    r"|//[^\n]*",
    re.M,
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

SLASH_COMMAND_RE = re.compile(r"(?:^|[\s`'\"])/philharmonic(?:\b|$)", re.I)


def _find_project_root() -> Path:
    """Resolve the project workspace root from the script location."""
    current = SCRIPT_DIR
    for parent in [current] + list(current.parents):
        if parent.name == ".agents":
            return parent.parent.resolve()
    return Path.cwd().resolve()


PROJECT_ROOT = _find_project_root()


def _is_shell_comment_start(text: str, idx: int) -> bool:
    """Return True iff text[idx] starts an unquoted shell comment."""
    return (
        idx < len(text)
        and text[idx] == "#"
        and (idx == 0 or text[idx - 1] in " \t\n;|&()")
    )


def _has_unescaped_heredoc_subst(line: str) -> bool:
    """Return True if an unquoted heredoc body line contains unescaped $(...) or backticks."""
    escaped = False
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if ch == "\\":
            escaped = True
            i += 1
            continue
        if ch == "`":
            return True
        if ch == "$" and i + 1 < n and line[i + 1] == "(":
            return True
        i += 1
    return False


def _preprocess_shell_heredocs_and_comments(cmd: str) -> Tuple[str, bool, bool]:
    """Strip unquoted shell comments and heredoc bodies, returning (cleaned_cmd, has_heredoc_cmd_subst, has_unclosed_syntax)."""
    if not cmd:
        return "", False, False
    out: List[str] = []
    has_heredoc_cmd_subst = False
    has_unclosed_syntax = False
    in_single = False
    in_ansi_single = False
    in_double = False
    escaped = False
    prev_unescaped_dollar = False
    pending_heredocs: List[Tuple[str, bool, bool]] = []
    i = 0
    n = len(cmd)
    while i < n:
        ch = cmd[i]
        if escaped:
            out.append(ch)
            escaped = False
            prev_unescaped_dollar = False
            i += 1
            continue
        if in_ansi_single:
            if ch == "\\" and i + 1 < n:
                if cmd[i + 1] == "c" and i + 2 < n and cmd[i + 2] != "'":
                    if (
                        cmd[i + 2] == "\\"
                        and i + 3 < n
                        and cmd[i + 3] in ("\\", "'")
                    ):
                        out.append(cmd[i : i + 4])
                        i += 4
                        continue
                    out.append(cmd[i : i + 3])
                    i += 3
                    continue
                out.append(cmd[i : i + 2])
                i += 2
                continue
            if ch == "'":
                in_ansi_single = False
            out.append(ch)
            i += 1
            continue
        if in_single:
            if ch == "'":
                in_single = False
            out.append(ch)
            i += 1
            continue
        if ch == "\\":
            if i + 1 < n and cmd[i + 1] == "\n":
                i += 2
                continue
            if i + 2 < n and cmd[i + 1] == "\r" and cmd[i + 2] == "\n":
                i += 3
                continue
            escaped = True
            prev_unescaped_dollar = False
            out.append(ch)
            i += 1
            continue
        if not in_double and ch == "$":
            if i + 1 < n and cmd[i + 1] == "'":
                in_ansi_single = True
                prev_unescaped_dollar = False
                out.append("$'")
                i += 2
                continue
            prev_unescaped_dollar = True
            out.append(ch)
            i += 1
            continue
        if not in_double and ch == "'":
            if prev_unescaped_dollar:
                in_ansi_single = True
            else:
                in_single = True
            prev_unescaped_dollar = False
            out.append(ch)
            i += 1
            continue
        if ch == '"':
            in_double = not in_double
            prev_unescaped_dollar = False
            out.append(ch)
            i += 1
            continue
        prev_unescaped_dollar = False
        if not in_double:
            if ch == "#" and (not out or out[-1] in " \t\n;|&()"):
                while i < n and cmd[i] != "\n":
                    i += 1
                continue
            if cmd[i : i + 2] == "<<" and (i + 2 >= n or cmd[i + 2] != "<"):
                out.append("<<")
                i += 2
                strip_tabs = False
                if i < n and cmd[i] == "-":
                    strip_tabs = True
                    out.append("-")
                    i += 1
                while i < n and cmd[i] in " \t":
                    out.append(cmd[i])
                    i += 1
                delim_chars: List[str] = []
                was_quoted = False
                d_single = False
                d_ansi = False
                d_double = False
                d_esc = False
                while i < n:
                    dch = cmd[i]
                    if d_esc:
                        delim_chars.append(dch)
                        out.append(dch)
                        d_esc = False
                        i += 1
                        continue
                    if d_ansi:
                        if dch == "\\" and i + 1 < n:
                            out.append(cmd[i : i + 2])
                            nxt = cmd[i + 1]
                            if nxt == "t":
                                delim_chars.append("\t")
                            elif nxt == "n":
                                delim_chars.append("\n")
                            elif nxt == "r":
                                delim_chars.append("\r")
                            else:
                                delim_chars.append(nxt)
                            i += 2
                            continue
                        out.append(dch)
                        if dch == "'":
                            d_ansi = False
                        else:
                            delim_chars.append(dch)
                        i += 1
                        continue
                    if d_single:
                        out.append(dch)
                        if dch == "'":
                            d_single = False
                        else:
                            delim_chars.append(dch)
                        i += 1
                        continue
                    if d_double:
                        if (
                            dch == "\\"
                            and i + 1 < n
                            and cmd[i + 1] in ('"', "\\", "$", "`", "\n")
                        ):
                            out.append(cmd[i : i + 2])
                            if cmd[i + 1] != "\n":
                                delim_chars.append(cmd[i + 1])
                            i += 2
                            continue
                        out.append(dch)
                        if dch == '"':
                            d_double = False
                        else:
                            delim_chars.append(dch)
                        i += 1
                        continue
                    if dch in " \t\r\n;|&()<>":
                        break
                    if dch == "\\":
                        was_quoted = True
                        out.append(dch)
                        d_esc = True
                        i += 1
                        continue
                    if dch == "$" and i + 1 < n and cmd[i + 1] == "'":
                        was_quoted = True
                        out.append("$'")
                        d_ansi = True
                        i += 2
                        continue
                    if dch == "'":
                        was_quoted = True
                        out.append(dch)
                        d_single = True
                        i += 1
                        continue
                    if dch == '"':
                        was_quoted = True
                        out.append(dch)
                        d_double = True
                        i += 1
                        continue
                    out.append(dch)
                    delim_chars.append(dch)
                    i += 1
                delim = "".join(delim_chars)
                if (
                    d_single
                    or d_ansi
                    or d_double
                    or d_esc
                    or not delim
                    or (not was_quoted and ("$" in delim or "`" in delim))
                ):
                    has_unclosed_syntax = True
                else:
                    pending_heredocs.append((delim, strip_tabs, was_quoted))
                continue
            if ch == "\n":
                out.append("\n")
                i += 1
                if pending_heredocs:
                    for delim, strip_tabs, was_quoted in pending_heredocs:
                        matched = False
                        body_lines: List[str] = []
                        while i < n:
                            line_end = cmd.find("\n", i)
                            if line_end == -1:
                                raw_line = cmd[i:]
                                i = n
                            else:
                                raw_line = cmd[i:line_end]
                                i = line_end + 1
                            candidate = raw_line.lstrip("\t") if strip_tabs else raw_line
                            if candidate == delim:
                                matched = True
                                break
                            body_lines.append(raw_line)
                        if not was_quoted and body_lines:
                            raw_body = "\n".join(body_lines)
                            folded_body = re.sub(
                                r"(?<!\\)((?:\\\\)*)\\\r?\n", r"\1", raw_body
                            )
                            if _has_unescaped_heredoc_subst(folded_body):
                                has_heredoc_cmd_subst = True
                        if not matched:
                            has_unclosed_syntax = True
                    pending_heredocs.clear()
                continue
        out.append(ch)
        i += 1
    if in_single or in_double or in_ansi_single or escaped or pending_heredocs:
        has_unclosed_syntax = True
    return "".join(out), has_heredoc_cmd_subst, has_unclosed_syntax


def _split_shell_commands_with_delims(cmd: str) -> List[Tuple[Optional[str], str]]:
    """Split a shell command string on unquoted &&, ||, ;, |, &, and newline delimiters, tracking preceding delimiter."""
    if not cmd:
        return []
    cmd, _, _ = _preprocess_shell_heredocs_and_comments(cmd)
    if not cmd:
        return []
    parts: List[Tuple[Optional[str], str]] = []
    current: List[str] = []
    preceding_delim: Optional[str] = None
    in_single = False
    in_ansi_single = False
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
        if (
            not in_single
            and not in_double
            and not in_ansi_single
            and ch == "$"
            and i + 1 < n
            and cmd[i + 1] == "'"
        ):
            in_ansi_single = True
            current.append("$'")
            i += 2
            continue
        if ch == "'" and not in_double:
            if in_ansi_single:
                in_ansi_single = False
            else:
                in_single = not in_single
            current.append(ch)
            i += 1
            continue
        if ch == '"' and not in_single and not in_ansi_single:
            in_double = not in_double
            current.append(ch)
            i += 1
            continue
        if not in_single and not in_double and not in_ansi_single:
            if _is_shell_comment_start(cmd, i):
                while i < n and cmd[i] != "\n":
                    i += 1
                continue
            if cmd[i : i + 3] == ";;&":
                part = "".join(current).strip()
                if part or preceding_delim == "&":
                    parts.append((preceding_delim, part))
                preceding_delim = ";;&"
                current = []
                i += 3
                continue
            if cmd[i : i + 2] in ("&&", "||", "|&", ";;", ";&"):
                part = "".join(current).strip()
                if part or preceding_delim == "&":
                    parts.append((preceding_delim, part))
                preceding_delim = cmd[i : i + 2]
                current = []
                i += 2
                continue
            if ch == "&" and (
                (i > 0 and cmd[i - 1] in (">", "<"))
                or (i + 1 < n and cmd[i + 1] == ">")
            ):
                current.append(ch)
                i += 1
                continue
            if ch in (";", "|", "&", "\n"):
                part = "".join(current).strip()
                if part or preceding_delim == "&":
                    parts.append((preceding_delim, part))
                if not (
                    ch == "\n"
                    and part == ""
                    and (
                        preceding_delim is None
                        or preceding_delim in ("|", "|&", "&&", "||")
                    )
                ):
                    preceding_delim = ch
                current = []
                i += 1
                continue
        current.append(ch)
        i += 1
    tail = "".join(current).strip()
    if tail or preceding_delim == "&":
        parts.append((preceding_delim, tail))
    return parts


def _split_shell_commands(cmd: str) -> List[str]:
    """Split a shell command string on unquoted &&, ||, ;, |, |&, &, and newline delimiters."""
    return [part for _, part in _split_shell_commands_with_delims(cmd) if part]


def _has_shell_masking_or_chaining(cmd: str) -> bool:
    """Return True if cmd uses unquoted ||, background &, or |/|&/&/; followed by trivial masking commands."""
    if not cmd:
        return False
    for delim, part in _split_shell_commands_with_delims(cmd):
        if delim == "||" or delim == "&":
            return True
        if delim in ("|", "|&", ";", "\n") and part and _SHELL_MASKING_TAIL_RE.match(part):
            return True
    return False


def _is_git_only_command(cmd: str) -> bool:
    """Return True only if every sub-command is a direct git command with no command substitution or write redirection."""
    if not cmd or "$(" in cmd or "`" in cmd:
        return False
    cleaned, has_h_subst, has_unclosed = _preprocess_shell_heredocs_and_comments(cmd)
    if has_h_subst or has_unclosed:
        return False
    unquoted = re.sub(
        r"\$'(?:\\.|[^'\\])*'|'[^']*'|\"(?:\\.|[^\"\\])*\"|\\.",
        " ",
        cleaned,
    )
    if re.search(
        r"(?:^|[^<])(?:[0-9]*>(?!&\s*(?:\d+|-)(?=$|[\s;|&)><]))|&>|>>|>\|)",
        unquoted,
    ) or re.search(r"[<>]\(", unquoted):
        return False
    sub_cmds = _split_shell_commands(cleaned)
    if not sub_cmds:
        return False
    return all(bool(GIT_COMMAND_RE.match(c)) for c in sub_cmds)


def _dequote_shell_subcmd(subcmd: str) -> str:
    """Dequote shell tokens in a subcommand for flag inspection, falling back to quote stripping."""
    try:
        return " ".join(shlex.split(subcmd))
    except ValueError:
        return re.sub(r"""(['"])(.*?)\1""", r"\2", subcmd)


def _fold_shell_line_continuations(text: str) -> str:
    """Fold unquoted and double-quoted shell line continuations while preserving '...' and $'...' verbatim."""
    if not text or ("\\\n" not in text and "\\\r\n" not in text):
        return text
    out: List[str] = []
    in_single = False
    in_ansi_single = False
    in_double = False
    escaped = False
    prev_unescaped_dollar = False
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if escaped:
            out.append(ch)
            escaped = False
            prev_unescaped_dollar = False
            i += 1
            continue
        if in_ansi_single:
            if ch == "\\" and i + 1 < n:
                if text[i + 1] == "c" and i + 2 < n and text[i + 2] != "'":
                    if (
                        text[i + 2] == "\\"
                        and i + 3 < n
                        and text[i + 3] in ("\\", "'")
                    ):
                        out.append(text[i : i + 4])
                        i += 4
                        continue
                    out.append(text[i : i + 3])
                    i += 3
                    continue
                out.append(text[i : i + 2])
                i += 2
                continue
            if ch == "'":
                in_ansi_single = False
            out.append(ch)
            i += 1
            continue
        if in_single:
            if ch == "'":
                in_single = False
            out.append(ch)
            i += 1
            continue
        if ch == "\\":
            if i + 1 < n and text[i + 1] == "\n":
                i += 2
                continue
            if i + 2 < n and text[i + 1] == "\r" and text[i + 2] == "\n":
                i += 3
                continue
            escaped = True
            prev_unescaped_dollar = False
            out.append(ch)
            i += 1
            continue
        if not in_double and ch == "$":
            if i + 1 < n and text[i + 1] == "'":
                in_ansi_single = True
                prev_unescaped_dollar = False
                out.append("$'")
                i += 2
                continue
            prev_unescaped_dollar = True
            out.append(ch)
            i += 1
            continue
        if not in_double and ch == "'":
            if prev_unescaped_dollar:
                in_ansi_single = True
            else:
                in_single = True
            prev_unescaped_dollar = False
            out.append(ch)
            i += 1
            continue
        if ch == '"':
            in_double = not in_double
            prev_unescaped_dollar = False
            out.append(ch)
            i += 1
            continue
        prev_unescaped_dollar = False
        out.append(ch)
        i += 1
    return "".join(out)


def _has_nul_or_surrogate_input(text: str) -> bool:
    """Return True if text contains raw NUL, lone UTF-16 surrogates, or ANSI-C NUL/surrogate escapes."""
    if not text:
        return False
    if not isinstance(text, str):
        return True
    if "\x00" in text or any(0xD800 <= ord(ch) <= 0xDFFF for ch in text):
        return True
    text = _fold_shell_line_continuations(text)
    in_single = False
    in_double = False
    escaped = False
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if ch == "\\" and not in_single:
            escaped = True
            i += 1
            continue
        if (
            not in_single
            and not in_double
            and ch == "$"
            and i + 1 < n
            and text[i + 1] == "'"
        ):
            i += 2
            while i < n:
                sch = text[i]
                if sch == "'":
                    i += 1
                    break
                if sch == "\\" and i + 1 < n:
                    nxt = text[i + 1]
                    if nxt == "x":
                        j = i + 2
                        while (
                            j < n
                            and j < i + 4
                            and text[j] in "0123456789abcdefABCDEF"
                        ):
                            j += 1
                        if j > i + 2:
                            if (int(text[i + 2 : j], 16) & 0xFF) == 0:
                                return True
                            i = j
                            continue
                        i += 2
                        continue
                    if nxt in ("u", "U"):
                        max_len = 4 if nxt == "u" else 8
                        j = i + 2
                        while (
                            j < n
                            and j < i + 2 + max_len
                            and text[j] in "0123456789abcdefABCDEF"
                        ):
                            j += 1
                        if j > i + 2:
                            cp = int(text[i + 2 : j], 16)
                            if cp == 0 or (0xD800 <= cp <= 0xDFFF):
                                return True
                            i = j
                            continue
                        i += 2
                        continue
                    if nxt in "01234567":
                        j = i + 1
                        while j < n and j < i + 4 and text[j] in "01234567":
                            j += 1
                        if (int(text[i + 1 : j], 8) & 0xFF) == 0:
                            return True
                        i = j
                        continue
                    if nxt == "c" and i + 2 < n and text[i + 2] != "'":
                        if (
                            text[i + 2] == "\\"
                            and i + 3 < n
                            and text[i + 3] in ("\\", "'")
                        ):
                            ctrl_ch = text[i + 3]
                            step = 4
                        else:
                            ctrl_ch = text[i + 2]
                            step = 3
                        if ctrl_ch in ("@", " ", "\x00") or (ord(ctrl_ch) & 0x1F) == 0:
                            return True
                        i += step
                        continue
                    i += 2
                    continue
                i += 1
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        i += 1
    return False


_TEST_CMD_BLOCKED_ENV_VARS = frozenset(
    {
        "PYTEST_ADDOPTS",
        "PYTEST_PLUGINS",
        "GOFLAGS",
        "GOENV",
        "CC",
        "CXX",
        "LD",
        "AR",
        "AS",
        "RUSTC",
        "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER",
        "RUSTFLAGS",
        "RUSTDOCFLAGS",
        "CARGO_HOME",
        "CARGO_BUILD_RUSTC",
        "CARGO_BUILD_RUSTC_WRAPPER",
        "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER",
        "CARGO_TARGET_DIR",
        "PYTHONSTARTUP",
        "PYTHONHOME",
        "PYTHONUSERBASE",
        "PYTHONEXECUTABLE",
        "PYTHONINSPECT",
        "PYTHONWARNINGS",
        "PYTHONBREAKPOINT",
        "PERL5OPT",
        "PERL5LIB",
        "PERLLIB",
        "RUBYOPT",
        "RUBYLIB",
        "NODE_OPTIONS",
        "NODE_PATH",
        "AWKPATH",
        "AWKLIBPATH",
        "GIT_EXTERNAL_DIFF",
        "GIT_EXEC_PATH",
        "GIT_SSH",
        "GIT_SSH_COMMAND",
        "GIT_ASKPASS",
        "GIT_CONFIG",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_SYSTEM",
        "GIT_CONFIG_GLOBAL",
        "PATH",
        "HOME",
        "BASH_ENV",
        "ENV",
        "IFS",
        "CDPATH",
        "SHELL",
        "ZDOTDIR",
        "TMPDIR",
        "XDG_CONFIG_HOME",
        "GCONV_PATH",
        "LOCPATH",
        "NLSPATH",
        "HOSTALIASES",
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "LUA_PATH",
        "LUA_CPATH",
    }
)
_TEST_CMD_BLOCKED_ENV_PREFIXES = (
    "LD_",
    "DYLD_",
    "NPM_CONFIG_",
    "YARN_",
    "PNPM_",
    "POETRY_",
    "PIPENV_",
    "PIP_",
    "CARGO_TARGET_",
    "CARGO_REGISTRIES_",
    "UV_",
    "PYTEST_",
    "GIT_TRACE",
    "LUA_INIT",
    "BASH_FUNC_",
)


def _parse_leading_env_assignments(
    cmd: str,
) -> Tuple[List[Tuple[str, bool, str]], str]:
    """Parse leading shell KEY=VAL / KEY+=VAL assignments in a quote-aware manner."""
    assignments: List[Tuple[str, bool, str]] = []
    n = len(cmd)
    i = 0
    while i < n:
        while i < n and cmd[i] in " \t\r\n":
            i += 1
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)(\+?=)", cmd[i:])
        if not m:
            break
        key = m.group(1)
        is_append = m.group(2) == "+="
        i += m.end()
        val_chars: List[str] = []
        in_sq = False
        in_ansi_sq = False
        in_dq = False
        while i < n:
            ch = cmd[i]
            if in_sq:
                if ch == "'":
                    in_sq = False
                else:
                    val_chars.append(ch)
                i += 1
                continue
            if in_ansi_sq:
                if ch == "'":
                    in_ansi_sq = False
                    i += 1
                    continue
                if ch == "\\" and i + 1 < n:
                    nxt = cmd[i + 1]
                    if nxt == "c" and i + 2 < n and cmd[i + 2] != "'":
                        if (
                            cmd[i + 2] == "\\"
                            and i + 3 < n
                            and cmd[i + 3] in ("\\", "'")
                        ):
                            val_chars.append(cmd[i + 3])
                            i += 4
                        else:
                            val_chars.append(cmd[i + 2])
                            i += 3
                        continue
                    val_chars.append(nxt)
                    i += 2
                    continue
                val_chars.append(ch)
                i += 1
                continue
            if in_dq:
                if ch == '"':
                    in_dq = False
                    i += 1
                    continue
                if ch == "\\" and i + 1 < n:
                    nxt = cmd[i + 1]
                    if nxt == "\n":
                        i += 2
                        continue
                    if nxt == "\r" and i + 2 < n and cmd[i + 2] == "\n":
                        i += 3
                        continue
                    if nxt in ('"', "\\", "$", "`"):
                        val_chars.append(nxt)
                        i += 2
                        continue
                val_chars.append(ch)
                i += 1
                continue
            if ch in " \t\r\n;|&<>()":
                break
            if ch == "$" and i + 1 < n and cmd[i + 1] == "'":
                in_ansi_sq = True
                i += 2
                continue
            if ch == "'":
                in_sq = True
                i += 1
                continue
            if ch == '"':
                in_dq = True
                i += 1
                continue
            if ch == "\\" and i + 1 < n:
                nxt = cmd[i + 1]
                if nxt == "\n":
                    i += 2
                    continue
                if nxt == "\r" and i + 2 < n and cmd[i + 2] == "\n":
                    i += 3
                    continue
                val_chars.append(nxt)
                i += 2
                continue
            val_chars.append(ch)
            i += 1
        assignments.append((key, is_append, "".join(val_chars)))
    return assignments, cmd[i:].strip()


def _is_test_command(cmd: str) -> bool:
    """Return True if any sub-command invokes a recognized test runner without masking chaining."""
    if not cmd or _has_shell_masking_or_chaining(cmd):
        return False
    if SHELL_EXEC_WRAPPER_RE.search(cmd):
        return False
    parts_with_delims = _split_shell_commands_with_delims(cmd)
    if not parts_with_delims:
        return False

    def _env_assignment_blocked(key: str, is_append: bool, val: Optional[str]) -> bool:
        k_upper = key.upper()
        if k_upper == "PYTHONPATH":
            if val is not None and (
                is_append or any(seg not in {"", "."} for seg in val.split(":"))
            ):
                return True
            return False
        return k_upper in _TEST_CMD_BLOCKED_ENV_VARS or k_upper.startswith(
            _TEST_CMD_BLOCKED_ENV_PREFIXES
        )

    found_test = False
    for idx, (_, subcmd) in enumerate(parts_with_delims):
        if not subcmd:
            continue
        folded_subcmd = _fold_shell_line_continuations(subcmd)
        assignments, remainder = _parse_leading_env_assignments(folded_subcmd)
        for key, is_append, val in assignments:
            if _env_assignment_blocked(key, is_append, val):
                return False
        if remainder:
            try:
                rem_tokens = shlex.split(remainder)
            except ValueError:
                rem_tokens = _dequote_shell_subcmd(remainder).split()
            if rem_tokens and os.path.basename(rem_tokens[0]).lower() in {
                "export",
                "declare",
                "typeset",
                "local",
                "readonly",
            }:
                for tok in rem_tokens[1:]:
                    if tok == "--" or tok.startswith("-"):
                        continue
                    if "=" in tok:
                        raw_k, raw_v = tok.split("=", 1)
                        tok_append = raw_k.endswith("+")
                        tok_k = raw_k[:-1] if tok_append else raw_k
                        if _env_assignment_blocked(tok_k, tok_append, raw_v):
                            return False
                    elif _env_assignment_blocked(tok, False, None):
                        return False
        if not remainder or not TEST_CMD_START_RE.match(remainder):
            continue
        dequoted = _dequote_shell_subcmd(folded_subcmd)
        if TRIVIAL_TEST_FLAG_RE.search(folded_subcmd) or TRIVIAL_TEST_FLAG_RE.search(dequoted):
            return False
        if re.search(r"(?:^|\s)vitest\s+(?:list|bench)\b", dequoted, re.I):
            return False
        if re.search(r"(?:^|\s)(?:jest|vitest)\b", dequoted, re.I):
            for _scan_str in (folded_subcmd, dequoted):
                try:
                    _jtoks = shlex.split(_scan_str)
                except ValueError:
                    _jtoks = _scan_str.split()
                for _jt in _jtoks:
                    if _jt == "--":
                        break
                    if _jt == "init" and re.search(r"(?:^|\s)vitest\b", dequoted, re.I):
                        return False
                    _jopt = _jt.split("=", 1)[0]
                    if _jopt in {
                        "-u", "--updateSnapshot", "--update-snapshot", "--init",
                        "--clearCache", "--clear-cache", "--passWithNoTests",
                        "--pass-with-no-tests", "--coverage", "--coverageDirectory",
                        "--reporter", "--reporters", "-R", "--config", "-c",
                        "--setupFiles", "--setupFilesAfterEnv", "--testEnvironment",
                    } or (_jt.startswith("-") and not _jt.startswith("--") and any(c in _jt[1:] for c in "vucR")):
                        return False
        if re.search(r"(?:^|\s)(?:npm|pnpm|yarn)\s+test\b", dequoted, re.I):
            for _scan_str in (folded_subcmd, dequoted):
                try:
                    _ntoks = shlex.split(_scan_str)
                except ValueError:
                    _ntoks = _scan_str.split()
                for _nt in _ntoks:
                    if _nt == "--" or _nt.split("=", 1)[0] in {
                        "-c", "-r", "--config", "--userconfig", "--globalconfig", "--require", "--prefix", "-C",
                    } or (_nt.startswith("-") and not _nt.startswith("--") and any(c in _nt[1:] for c in "crC")):
                        return False
        if re.search(r"(?:^|\s)(?:tox|nox)\b", dequoted, re.I):
            for _scan_str in (folded_subcmd, dequoted):
                try:
                    _ttoks = shlex.split(_scan_str)
                except ValueError:
                    _ttoks = _scan_str.split()
                for _tt in _ttoks:
                    if _tt in {"-", "--"} or _tt.split("=", 1)[0] in {"--installpkg", "--force-dep"}:
                        return False
        if re.search(r"(?:^|\s)(?:-m\s*)?pytest\b", dequoted):
            for _scan_str in (folded_subcmd, dequoted):
                try:
                    _ptoks = shlex.split(_scan_str)
                except ValueError:
                    _ptoks = _scan_str.split()
                _pidx = None
                for _i, _tok in enumerate(_ptoks):
                    if _tok in {"pytest", "py.test", "-mpytest", "-mpy.test"} or _tok.endswith(
                        ("/pytest", "/py.test")
                    ):
                        _pidx = _i
                        break
                if _pidx is None:
                    continue
                _skip_next = False
                for _tok in _ptoks[_pidx + 1 :]:
                    if _skip_next:
                        _skip_next = False
                        continue
                    if _tok == "--":
                        break
                    if _tok in {
                        "--override-ini",
                        "--config-file",
                        "--rootdir",
                        "--confcutdir",
                    } or _tok.startswith(
                        (
                            "--override-ini=",
                            "--config-file=",
                            "--rootdir=",
                            "--confcutdir=",
                        )
                    ):
                        return False
                    if _tok.startswith("-") and not _tok.startswith("--") and len(_tok) > 1:
                        _chars = _tok[1:]
                        for _ci, _ch in enumerate(_chars):
                            if _ch in "ocp":
                                return False
                            if _ch in "kmrWn":
                                if _ci == len(_chars) - 1:
                                    _skip_next = True
                                break
        if re.search(r"(?:^|\s)go\s+test\b", dequoted) and (
            re.search(
                r"(?:^|\s)(?:(?-i:--?c)(?=$|\s|=)|(?-i:--?C)(?==|\s|$)|--?(?:exec|list|toolexec|overlay|modfile|coverprofile|cpuprofile|memprofile|blockprofile|mutexprofile|trace|outputdir|o)(?==|\s|$)|(?-i:--?n)(?=$|\s))",
                folded_subcmd,
            )
            or re.search(
                r"(?:^|\s)(?:(?-i:--?c)(?=$|\s|=)|(?-i:--?C)(?==|\s|$)|--?(?:exec|list|toolexec|overlay|modfile|coverprofile|cpuprofile|memprofile|blockprofile|mutexprofile|trace|outputdir|o)(?==|\s|$)|(?-i:--?n)(?=$|\s))",
                dequoted,
            )
            or re.search(
                r"(?:^|\s)(?:/tmp(?:/|\b)|\S*scratch(?:/|\b)|/brain(?:/|\b))",
                folded_subcmd,
            )
            or re.search(
                r"(?:^|\s)(?:/tmp(?:/|\b)|\S*scratch(?:/|\b)|/brain(?:/|\b))",
                dequoted,
            )
        ):
            return False
        if re.search(r"(?:^|\s)cargo\s+test\b", dequoted) and (
            re.search(
                r"(?:^|\s)(?:--manifest-path(?==|\s|$)|--target-dir(?==|\s|$)|--list(?==|\s|$))",
                folded_subcmd,
            )
            or re.search(
                r"(?:^|\s)(?:--manifest-path(?==|\s|$)|--target-dir(?==|\s|$)|--list(?==|\s|$))",
                dequoted,
            )
        ):
            return False
        next_delim = (
            parts_with_delims[idx + 1][0]
            if idx + 1 < len(parts_with_delims)
            else None
        )
        if next_delim in ("|", "|&", ";", "\n", "||", "&"):
            return False
        found_test = True
    return found_test


def _strip_comments(
    text: str, file_name: str = "", *, preserve_ignore: bool = False
) -> str:
    """Remove comments and string literals before counting test definitions, returns, or assertions."""
    if not text:
        return ""
    def _replace_non_code(match: re.Match[str]) -> str:
        tok = match.group(0)
        if tok.startswith(('"""', "'''", '"')):
            return '""'
        if tok.startswith("'"):
            return "''"
        return ""

    is_non_py = bool(file_name) and Path(file_name).suffix.lower() not in {".py", ".pyi"}
    pattern = _STRIP_NON_CODE_C_STYLE_RE if is_non_py else _STRIP_NON_CODE_RE
    t = pattern.sub(_replace_non_code, text)
    if preserve_ignore:
        return "\n".join(
            line
            for line in t.splitlines()
            if not COMMENT_LINE_RE.match(line) or re.match(r"^\s*#\s*\[ignore\b", line)
        )
    return "\n".join(
        line for line in t.splitlines() if not COMMENT_LINE_RE.match(line)
    )


def _count_early_test_returns(text: str, file_name: str = "") -> int:
    """Count return statements directly inside `def test_*` bodies (ignoring nested defs)."""
    if not text:
        return 0
    cleaned = _strip_comments(text, file_name=file_name)
    count = 0
    in_test_def = False
    test_indent = -1
    body_indent: Optional[int] = None
    nested_def_indent: Optional[int] = None
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip(" \t"))
        if re.match(r"^(?:async\s+)?def\s+test_[A-Za-z0-9_]+\s*\(", stripped):
            m_inline = re.search(r"\)\s*(?:->[^:]+)?:\s*(\S.*)$", stripped)
            if m_inline:
                if re.search(r"(?:^|[;:]\s*)return(?:\b|$)", m_inline.group(1)):
                    count += 1
                in_test_def = False
                test_indent = -1
                body_indent = None
                nested_def_indent = None
                continue
            in_test_def = True
            test_indent = indent
            body_indent = None
            nested_def_indent = None
            continue
        if in_test_def:
            if indent <= test_indent and not stripped.startswith(("@", ")")):
                in_test_def = False
                test_indent = -1
                body_indent = None
                nested_def_indent = None
                continue
            if nested_def_indent is not None:
                if indent <= nested_def_indent:
                    nested_def_indent = None
                else:
                    continue
            if re.match(r"^(?:async\s+)?def\s+", stripped):
                nested_def_indent = indent
                continue
            if body_indent is None:
                body_indent = indent
            if indent >= body_indent and re.search(
                r"(?:^|[;:]\s*)return(?:\b|$)", stripped
            ):
                count += 1
    return count


def _extract_call_id(payload: Dict[str, Any]) -> str:
    """Extract a deterministic tool call identifier from toolCall.id or stepIdx."""
    raw_tc = payload.get("toolCall")
    tool_call = raw_tc if isinstance(raw_tc, dict) else {}
    tc_id = tool_call.get("id")
    if tc_id is not None and str(tc_id) != "":
        return str(tc_id)
    step_idx = payload.get("stepIdx")
    if step_idx is not None and str(step_idx) != "":
        return f"step:{step_idx}"
    return ""
