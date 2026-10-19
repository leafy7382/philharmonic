#!/usr/bin/env python3
"""Deterministic linter for the structural STE-100 rules.

Checks only rules verifiable without ASD's dictionary. Deliberately never
flags hedges or modality (may/might/could): confidence is content, and a
linter that pressures hedges out would rewrite claims.

Also provides `lint_strict_prompt(text)` for inter-agent prompts, which
enforces the 20-word cap on numbered procedural instructions in addition
to the standard 25-word cap on descriptive sentences.
"""
from __future__ import annotations

import json
import re
import sys
from typing import Any

RULES = [
    (
        "semicolon",
        "advisory-free",
        re.compile(r";"),
        "STE bans the semicolon (Rule 8.1). Split into separate sentences.",
    ),
    (
        "phrasal-verb",
        "advisory-free",
        re.compile(
            r"\b(spin(?:ning|s)? up|spun up|reach(?:ing|es|ed)? out|"
            r"div(?:e|es|ing|ed) into|dove into|kick(?:ing|s|ed)? off|"
            r"circl(?:e|es|ing|ed) back|touch(?:ing|es|ed)? base)\b",
            re.I,
        ),
        "Soft phrasal verb. Use the single plain verb (start, contact, read, begin).",
    ),
    (
        "marketing-adjective",
        "advisory-free",
        re.compile(
            r"\b(seamless(?:ly)?|robust(?:ly)?|cutting-edge|effortless(?:ly)?|"
            r"blazing[- ]fast|world-class|state-of-the-art|game-chang(?:ing|er))\b",
            re.I,
        ),
        "Marketing adjective. Delete, or replace with the measurement that earns the claim.",
    ),
    (
        "nominalization",
        "advisory-free",
        re.compile(
            r"\b(perform|performs|performed|conduct|conducts|conducted|"
            r"carry out|carries out|carried out)\s+(?:a|an|the)\s+"
            r"\w+(?:tion|sion|ment|ance|ence|ysis)\b",
            re.I,
        ),
        "Action frozen into a noun. Use the verb (analyze, not perform an analysis of).",
    ),
    (
        "passive-voice",
        "advisory",
        re.compile(
            r"\b(is|are|was|were|been|being)\s+"
            r"(\w+ed|given|taken|made|done|found|seen|known|shown|written|"
            r"built|sent|set|run|read|kept|held|left|put)\b"
            r"(?!\s+(?:to|for|by)\s+\w+ing)",
            re.I,
        ),
        "Possible passive voice. Name the actor and use an active verb, unless the actor is unknown or irrelevant.",
    ),
    (
        "present-perfect",
        "advisory",
        re.compile(
            r"(?<!\bmay )(?<!\bmight )(?<!\bcould )(?<!\bshould )(?<!\bwould )(?<!\bmust )"
            r"\b(has|have|had)\s+(?:been\s+)?\w+(?:ed|en)\b",
            re.I,
        ),
        "Compound tense. Use simple past/present unless current relevance is the point (then keep and flag).",
    ),
]

SYNONYM_GROUPS = [
    ("check", "verify", "confirm", "validate"),
    ("delete", "remove", "erase"),
    ("start", "launch", "begin", "initiate"),
    ("stop", "halt", "terminate"),
    ("show", "display"),
    ("use", "utilize", "employ"),
    ("fix", "repair", "correct"),
    ("send", "transmit"),
    ("get", "retrieve", "fetch", "obtain"),
    ("change", "modify", "alter"),
    ("modify", "edit"),
]

MAX_RULE_WORDS: int = 20
MAX_WORDS = 25
MAX_INSTRUCTION_WORDS = 20

PHRASAL_VERBS: list[str] = [
    "figure out",
    "look into",
    "set up",
    "clean up",
    "make sure",
    "spin up",
    "spun up",
    "reach out",
    "dive into",
    "dove into",
    "kick off",
    "circle back",
    "touch base",
]

FILLER_PHRASES: list[str] = [
    "in order to",
    "please note",
    "as well as",
    "be able to",
]

NOMINALIZATIONS: dict[str, str] = {
    "implementation": "implement",
    "configuration": "configure",
    "verification": "verify",
    "investigation": "investigate",
}

RULE_LINE_RE: re.Pattern[str] = re.compile(r"^\s*([0-9]+[.)]|[-*])\s+(.+)$")
FENCED_CODE_RE: re.Pattern[str] = re.compile(r"```.*?```", flags=re.DOTALL)
INLINE_CODE_RE: re.Pattern[str] = re.compile(r"`[^`]+`")
WORD_TOKEN_RE: re.Pattern[str] = re.compile(r"[A-Za-z0-9_-]+")

PHRASAL_VERB_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (verb, re.compile(rf"\b{re.escape(verb)}\b", re.IGNORECASE))
    for verb in PHRASAL_VERBS
]

FILLER_PHRASE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (phrase, re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE))
    for phrase in FILLER_PHRASES
]

NOMINALIZATION_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (noun, re.compile(rf"\b{re.escape(noun)}\b", re.IGNORECASE), replacement)
    for noun, replacement in NOMINALIZATIONS.items()
]

SYNONYM_GROUP_PATTERNS: list[list[tuple[str, re.Pattern[str]]]] = [
    [(w, re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE)) for w in group]
    for group in SYNONYM_GROUPS
]

CODE_FENCE = re.compile(r"^(```|~~~)")
INLINE_CODE = re.compile(r"`[^`]*`")
LIST_ITEM_START = re.compile(
    r"^(?P<indent> {0,3})(?P<marker>[-*+]|[0-9]+[.)])(?P<gap> +)(?P<body>.*)$"
)
NUMBERED_INSTRUCTION_START = re.compile(r"^ {0,3}[0-9]+[.)]\s+(?P<body>.*)$")
CONJUNCTION_END = re.compile(r"\b(?:and|or)\s*$", re.I)
TABLE_SEPARATOR_CELL = re.compile(r"^:?-{3,}:?$")

_WORD_RES: dict[str, re.Pattern[str]] = {
    base: re.compile(r"\b" + base + r"(?:s|es|ed|d|ing)?\b", re.I)
    for group in SYNONYM_GROUPS
    for base in group
}


def _word_re(base: str) -> re.Pattern[str]:
    return _WORD_RES.get(base) or re.compile(r"\b" + base + r"(?:s|es|ed|d|ing)?\b", re.I)


def _strip_code(text: str) -> str:
    cleaned = FENCED_CODE_RE.sub(" ", text)
    return INLINE_CODE_RE.sub("CODE_TOKEN", cleaned)


def _word_count(sentence: str) -> int:
    cleaned = INLINE_CODE_RE.sub("CODE", sentence)
    return len(WORD_TOKEN_RE.findall(cleaned))


def lint_prompt(prompt: str) -> list[str]:
    """Lint a prompt string against STE-100 rules for subagent instructions.

    Returns a list of violation strings.
    """
    violations: list[str] = []

    # Strip triple-backtick fenced code blocks so violations inside them are ignored.
    text_without_fences = FENCED_CODE_RE.sub(" ", prompt)

    # Extract numbered rule lines
    rule_lines: list[tuple[int, str]] = []
    for lineno, line in enumerate(text_without_fences.splitlines(), 1):
        m = RULE_LINE_RE.match(line)
        if m:
            rule_lines.append((lineno, m.group(2).strip()))

    # If no numbered rule lines found, treat non-empty stripped lines as rules
    # (excluding uppercase headers or standard section labels)
    if not rule_lines:
        for lineno, line in enumerate(text_without_fences.splitlines(), 1):
            s = line.strip()
            if (
                s
                and not s.isupper()
                and not s.endswith(":")
                and not any(
                    s.startswith(p)
                    for p in (
                        "TASK:",
                        "RULES:",
                        "REPORT FORMAT:",
                        "STATUS:",
                        "FILES_CHANGED:",
                        "VERIFICATION:",
                        "NOTES:",
                    )
                )
            ):
                rule_lines.append((lineno, s))

    for lineno, rule_text in rule_lines:
        # Check sentence length (collapsing inline code to 1 token)
        words = _word_count(rule_text)
        if words > MAX_RULE_WORDS:
            violations.append(
                f"line {lineno} [long-sentence]: Rule exceeds {MAX_RULE_WORDS} words ({words} words): {rule_text}"
            )

        # Mask inline code for remaining rule line checks so semicolons or words in code are not flagged
        cleaned_rule = INLINE_CODE_RE.sub("CODE_TOKEN", rule_text)

        # Semicolon check
        if ";" in cleaned_rule:
            violations.append(
                f"line {lineno} [semicolon]: Semicolons are forbidden in rules: {rule_text}"
            )

        # Phrasal verbs check
        for verb, pat in PHRASAL_VERB_PATTERNS:
            if pat.search(cleaned_rule):
                violations.append(
                    f"line {lineno} [phrasal-verb]: Banned phrasal verb '{verb}' found: {rule_text}"
                )

        # Filler phrases check
        for phrase, pat in FILLER_PHRASE_PATTERNS:
            if pat.search(cleaned_rule):
                violations.append(
                    f"line {lineno} [filler]: Banned filler phrase '{phrase}' found: {rule_text}"
                )

        # Nominalizations check
        for noun, pat, replacement in NOMINALIZATION_PATTERNS:
            if pat.search(cleaned_rule):
                violations.append(
                    f"line {lineno} [nominalization]: Banned nominalization '{noun}' found; use '{replacement}': {rule_text}"
                )

    # Synonym rotation check across the prompt (outside fenced code and inline code)
    prompt_for_synonyms = INLINE_CODE_RE.sub(" ", text_without_fences)
    for group_patterns in SYNONYM_GROUP_PATTERNS:
        words_found = [
            w for w, pat in group_patterns
            if pat.search(prompt_for_synonyms)
        ]
        if len(words_found) > 1:
            violations.append(
                f"[synonym-rotation]: Multiple synonyms from same group used: {', '.join(words_found)}. Pick one and use it consistently."
            )

    return violations


def check_subagent_call(tool_name: str, args: dict[str, Any] | None = None) -> str | None:
    """Check tool call arguments for STE-100 violations.

    Returns formatted remediation text on violations, or None if valid.
    """
    if not isinstance(args, dict):
        return None
    if tool_name == "invoke_subagent":
        subagents = args.get("Subagents") or []
        all_violations: list[str] = []
        for i, subagent in enumerate(subagents, 1):
            if not isinstance(subagent, dict):
                continue
            prompt = subagent.get("Prompt", "")
            violations = lint_prompt(prompt)
            if violations:
                role = subagent.get("Role") or subagent.get("TypeName") or f"Subagent #{i}"
                all_violations.append(f"{role}: " + "; ".join(violations))
        if all_violations:
            return (
                "[STE-100 LINT GATE] Subagent prompt violation(s):\n"
                + "\n".join(f"- {v}" for v in all_violations)
            )
    elif tool_name == "send_message":
        msg = args.get("Message", "")
        violations = lint_prompt(msg)
        if violations:
            return (
                "[STE-100 LINT GATE] Message violation(s):\n"
                + "\n".join(f"- {v}" for v in violations)
            )
    return None



def _leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _is_list_continuation(line: str, content_indent: int) -> bool:
    if not line.strip():
        return True
    if LIST_ITEM_START.match(line):
        return False
    return _leading_spaces(line) >= content_indent


def _split_table_row(line: str):
    left = len(line) - len(line.lstrip())
    right = len(line.rstrip())
    content = line[left:right]
    if "|" not in content:
        return None
    if content.startswith("|"):
        content = content[1:]
        left += 1
    if content.endswith("|"):
        content = content[:-1]
    raw_cells = re.split(r"(?<!\\)\|", content)
    if len(raw_cells) < 2:
        return None

    cells = []
    column = left
    for raw_cell in raw_cells:
        leading = len(raw_cell) - len(raw_cell.lstrip())
        cells.append((raw_cell.strip(), column + leading))
        column += len(raw_cell) + 1
    return cells


def _markdown_table_cells(lines):
    table_cells = {}
    index = 1
    while index < len(lines):
        separator = _split_table_row(lines[index])
        header = _split_table_row(lines[index - 1])
        if (
            not separator
            or not header
            or len(separator) != len(header)
            or not all(TABLE_SEPARATOR_CELL.fullmatch(cell) for cell, _ in separator)
        ):
            index += 1
            continue

        table_cells[index - 1] = header
        table_cells[index] = []
        index += 1
        while index < len(lines):
            row = _split_table_row(lines[index])
            if not row or len(row) != len(separator):
                break
            table_cells[index] = row
            index += 1
    return table_cells


def _dangling_conjunction_findings(text: str, filename: str):
    lines = text.splitlines()
    findings = []
    in_fence = False
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if CODE_FENCE.match(stripped):
            in_fence = not in_fence
            index += 1
            continue
        if in_fence:
            index += 1
            continue
        start = LIST_ITEM_START.match(line)
        if not start:
            index += 1
            continue

        content_indent = (
            len(start.group("indent"))
            + len(start.group("marker"))
            + len(start.group("gap"))
        )
        item_lines = [(index, start.group("body"))]
        next_index = index + 1
        item_fence = False
        while next_index < len(lines):
            candidate = lines[next_index]
            candidate_stripped = candidate.strip()
            if CODE_FENCE.match(candidate_stripped):
                item_fence = not item_fence
                next_index += 1
                continue
            if item_fence:
                next_index += 1
                continue
            if not _is_list_continuation(candidate, content_indent):
                break
            item_lines.append((next_index, candidate))
            next_index += 1

        meaningful = []
        for line_index, item_line in item_lines:
            cleaned = INLINE_CODE.sub(" CODE ", item_line).strip()
            if cleaned:
                meaningful.append((line_index, cleaned))
        if meaningful:
            end_line_index, end_line = meaningful[-1]
            conjunction = CONJUNCTION_END.search(end_line)
        else:
            end_line_index, end_line, conjunction = None, None, None
        if conjunction:
            if end_line_index == index:
                finding_line = index + 1
                finding_col = start.start("marker") + 1
            else:
                raw_end_line = next(
                    raw
                    for line_index, raw in item_lines
                    if line_index == end_line_index
                )
                masked_end_line = INLINE_CODE.sub(
                    lambda match: " " * len(match.group(0)), raw_end_line
                )
                raw_conjunction = CONJUNCTION_END.search(masked_end_line)
                finding_line = end_line_index + 1
                finding_col = raw_conjunction.start() + 1 if raw_conjunction else 1
            findings.append(
                {
                    "file": filename,
                    "line": finding_line,
                    "col": finding_col,
                    "rule": "dangling-conjunction",
                    "level": "advisory-free",
                    "match": end_line,
                    "message": "List item ends with a coordinating conjunction. Complete the item or join it with the next item.",
                }
            )
        index = next_index
    return findings


def lint(text: str, filename: str = "<stdin>", strict_instructions: bool = False):
    findings = []
    words_total = 0
    in_fence = False
    lines = text.splitlines()
    table_cells = _markdown_table_cells(lines)
    seen_synonyms = {}
    for lineno, raw_line in enumerate(lines, 1):
        if CODE_FENCE.match(raw_line.strip()):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        is_numbered = bool(NUMBERED_INSTRUCTION_START.match(raw_line))
        segments = table_cells.get(lineno - 1, [(raw_line, 0)])
        for segment, source_column in segments:
            line = INLINE_CODE.sub("", segment)
            words_total += len(line.split())
            for rule_id, level, pattern, msg in RULES:
                for m in pattern.finditer(line):
                    findings.append(
                        {
                            "file": filename,
                            "line": lineno,
                            "col": source_column + m.start() + 1,
                            "rule": rule_id,
                            "level": level,
                            "match": m.group(0),
                            "message": msg,
                        }
                    )
            for gi, group in enumerate(SYNONYM_GROUPS):
                for base in group:
                    if (gi, base) in seen_synonyms:
                        continue
                    m = _word_re(base).search(line)
                    if m:
                        seen_synonyms[(gi, base)] = (
                            lineno,
                            source_column + m.start() + 1,
                            m.group(0),
                        )
            for sent in re.split(r"(?<=[.!?])\s+", line):
                # Strip leading list marker when counting instruction words
                cleaned_sent = re.sub(r"^ {0,3}(?:[-*+]|[0-9]+[.)])\s+", "", sent)
                n = len(cleaned_sent.split())
                cap = (
                    MAX_INSTRUCTION_WORDS
                    if (strict_instructions and is_numbered)
                    else MAX_WORDS
                )
                if n > cap:
                    findings.append(
                        {
                            "file": filename,
                            "line": lineno,
                            "col": source_column + 1,
                            "rule": "long-sentence",
                            "level": "advisory-free",
                            "match": f"{n} words",
                            "message": f"Sentence has {n} words (cap {cap}). Split it.",
                        }
                    )
    for gi, group in enumerate(SYNONYM_GROUPS):
        present = [
            (seen_synonyms[(gi, b)], b) for b in group if (gi, b) in seen_synonyms
        ]
        if len(present) > 1:
            present.sort()
            first_base = present[0][1]
            for (lineno, col, match), base in present[1:]:
                findings.append(
                    {
                        "file": filename,
                        "line": lineno,
                        "col": col,
                        "rule": "synonym-rotation",
                        "level": "advisory-free",
                        "match": match,
                        "message": f"'{base}' and '{first_base}' name the same action. Pick one and use it every time.",
                    }
                )
    findings.extend(_dangling_conjunction_findings(text, filename))
    findings.sort(key=lambda f: (f["line"], f["col"]))
    return findings, words_total


def report(findings, words_total, as_json, hard_count, baseline):
    rate = round(len(findings) * 100 / words_total, 1) if words_total else 0.0
    if as_json:
        print(
            json.dumps(
                {
                    "violations": findings,
                    "count": len(findings),
                    "hard_count": hard_count,
                    "baseline": baseline,
                    "words": words_total,
                    "per_100_words": rate,
                },
                indent=2,
            )
        )
        return
    for f in findings:
        print(
            f"{f['file']}:{f['line']}:{f['col']} {f['rule']}: {f['message']} [{f['match']}]"
        )
    print(
        f"\n{len(findings)} violations ({hard_count} hard, baseline {baseline}), "
        f"{words_total} words, {rate} per 100 words"
    )


def main(argv):
    as_json = "--json" in argv
    strict_instructions = "--strict-instructions" in argv
    baseline = 0
    disabled = set()
    paths = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--baseline":
            i += 1
            baseline = int(argv[i])
        elif a == "--disable":
            i += 1
            disabled = set(argv[i].split(","))
        elif not a.startswith("--"):
            paths.append(a)
        i += 1

    findings, words_total = [], 0
    if paths:
        for p in paths:
            with open(p, encoding="utf-8") as fh:
                f, w = lint(
                    fh.read(), filename=p, strict_instructions=strict_instructions
                )
            findings.extend(f)
            words_total += w
    else:
        findings, words_total = lint(
            sys.stdin.read(), strict_instructions=strict_instructions
        )

    findings = [f for f in findings if f["rule"] not in disabled]
    hard_count = sum(1 for f in findings if f["level"] == "advisory-free")
    report(findings, words_total, as_json, hard_count, baseline)
    return 1 if hard_count > baseline else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
