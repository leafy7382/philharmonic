"""Capability-tier model registry, models.json/env overrides, and dynamic blind-judge leak patterns."""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

MODELS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "models.json"

MODEL_TIERS: Tuple[str, ...] = ("lite", "standard", "frontier", "inherit")
MODEL_TIERS_SET = frozenset(MODEL_TIERS)
TIER_ENV_VARS: Dict[str, str] = {
    "lite": "PHILHARMONIC_MODEL_LITE",
    "standard": "PHILHARMONIC_MODEL_STANDARD",
    "frontier": "PHILHARMONIC_MODEL_FRONTIER",
    "inherit": "PHILHARMONIC_MODEL_INHERIT",
}

DEFAULT_TIER_BY_TYPE: Dict[str, str] = {
    "scout": "lite",
    "research": "standard",
    "researcher": "standard",
    "council": "standard",
    "implementer": "standard",
    "verifier": "standard",
    "reviewer": "standard",
    "judge": "standard",
}
ALLOWED_STANDARD_SUBAGENT_TIERS = {"lite", "standard"}
ALLOWED_HIGH_REASONING_TIERS = {"standard", "frontier", "inherit"}
ALLOWED_BEST_OF_N_TIERS = {"standard", "frontier"}
ALLOWED_COUNCIL_PROPOSER_TIERS = {"standard", "frontier", "inherit"}

DEFAULT_TIER_TO_MODEL: Dict[str, str] = {
    "lite": "flash_lite",
    "standard": "flash",
    "frontier": "pro",
    "inherit": "inherit",
}
DEFAULT_MODEL_ALIASES: Dict[str, str] = {
    "lite": "lite", "standard": "standard", "frontier": "frontier", "inherit": "inherit",
    "flash_lite": "lite", "flash-lite": "lite", "flash": "standard", "pro": "frontier",
    "hydrogen": "standard", "hydrogen_lite": "lite", "hydrogen-lite": "lite",
    "carbon": "frontier", "neon": "frontier", "argon": "frontier",
}

GOOGLE_MODEL_CODENAMES: Tuple[str, ...] = ("carbon", "neon", "argon", "hydrogen")
CLAUDE_MODEL_CODENAMES: Tuple[str, ...] = ("opus", "sonnet", "haiku", "fable")
PROVIDER_BRAND_TOKENS: Tuple[str, ...] = (
    "gemini", "claude", "chatgpt", "grok", "openai", "anthropic", "deepseek", "llama", "qwen", "mistral",
)
_GENERIC_NON_BRAND_TOKENS = frozenset({
    "inherit", "lite", "standard", "frontier", "pro", "default", "auto", "fast",
    "small", "medium", "large", "mini", "max", "base", "high", "low", "model",
    "models", "preview", "latest", "stable", "exp", "experimental", "alpha",
    "beta", "custom", "local", "remote", "test",
})
_BUILTIN_BRAND_TOKENS = frozenset({
    *PROVIDER_BRAND_TOKENS, *CLAUDE_MODEL_CODENAMES, *GOOGLE_MODEL_CODENAMES,
    "flash", "flash_lite", "flash-lite", "hydrogen_lite", "hydrogen-lite",
    "gpt", "o1", "o3", "o4", "geminipro", "claudesonnet", "claudeopus",
})
_BRAND_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{2,63}$")


def _is_valid_wire_model(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    s = value.strip()
    return bool(s) and not any(c in s for c in ("\x00", "\r", "\n")) and not any(0xD800 <= ord(c) <= 0xDFFF for c in s)


def _extract_brand_tokens_from_wire_model(wire_model: str) -> List[str]:
    cleaned = wire_model.strip().lower()
    if not cleaned or cleaned in _GENERIC_NON_BRAND_TOKENS or cleaned in _BUILTIN_BRAND_TOKENS:
        return []
    tokens = [cleaned] if _BRAND_TOKEN_RE.fullmatch(cleaned) else []
    for part in re.split(r"[-_./\s]+", cleaned):
        if (
            len(part) >= 3 and part not in _GENERIC_NON_BRAND_TOKENS and part not in _BUILTIN_BRAND_TOKENS
            and _BRAND_TOKEN_RE.fullmatch(part) and not re.fullmatch(r"v?[0-9]+(?:\.[0-9]+)*", part) and part not in tokens
        ):
            tokens.append(part)
    return tokens


def _apply_overrides(
    raw_obj: Any, kind: str, target: Any, source: str, issues: List[str]
) -> None:
    if kind == "brand_tokens":
        if not isinstance(raw_obj, list):
            issues.append(f"{source} must be a list of model brand token strings.")
            return
        for tok in raw_obj:
            if not isinstance(tok, str) or not _BRAND_TOKEN_RE.fullmatch(tok.strip()):
                issues.append(f"Invalid brand token {tok!r} in {source}.")
            elif (norm := tok.strip().lower()) not in _GENERIC_NON_BRAND_TOKENS and norm not in _BUILTIN_BRAND_TOKENS:
                target.append(norm)
        return
    if not isinstance(raw_obj, dict):
        issues.append(f"{source} must be a JSON object.")
        return
    for raw_k, raw_v in raw_obj.items():
        k = str(raw_k or "").strip().lower()
        if kind == "tiers":
            if k not in MODEL_TIERS_SET:
                issues.append(f"Unknown model tier '{raw_k}' in {source} (expected {', '.join(MODEL_TIERS)}).")
            elif not _is_valid_wire_model(raw_v):
                issues.append(f"Invalid wire model value for tier '{k}' in {source}.")
            else:
                target[k] = raw_v.strip()
        else:
            v = str(raw_v or "").strip().lower()
            if not k or v not in MODEL_TIERS_SET:
                issues.append(f"Invalid model alias '{raw_k}' -> '{raw_v}' in {source}.")
            else:
                target[k] = v


def _load_model_settings(
    config_path: Optional[Path] = None,
) -> Tuple[Dict[str, str], Dict[str, str], Tuple[str, ...], List[str]]:
    """Load merged tier map, alias map, dynamic brand tokens, and validation issues."""
    tiers, aliases, extra_brands, issues = dict(DEFAULT_TIER_TO_MODEL), dict(DEFAULT_MODEL_ALIASES), [], []
    cfg_path = config_path if config_path is not None else MODELS_CONFIG_PATH
    if cfg_path.exists():
        try:
            raw_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"Unreadable {cfg_path.name}: {exc}")
            raw_cfg = None
        if raw_cfg is not None:
            if not isinstance(raw_cfg, dict):
                issues.append(f"{cfg_path.name} root must be a JSON object.")
            else:
                for key, target in (("tiers", tiers), ("aliases", aliases), ("brand_tokens", extra_brands)):
                    if key in raw_cfg:
                        _apply_overrides(raw_cfg[key], key, target, f"{cfg_path.name}.{key}", issues)

    for env_name, kind, target in (
        ("PHILHARMONIC_MODEL_TIERS", "tiers", tiers),
        ("PHILHARMONIC_MODEL_ALIASES", "aliases", aliases),
    ):
        if raw_env := os.environ.get(env_name):
            try:
                _apply_overrides(json.loads(raw_env), kind, target, env_name, issues)
            except json.JSONDecodeError:
                issues.append(f"{env_name} is not valid JSON.")

    if raw_brands := (os.environ.get("PHILHARMONIC_MODEL_BRAND_TOKENS") or "").strip():
        if raw_brands.startswith("["):
            try:
                _apply_overrides(json.loads(raw_brands), "brand_tokens", extra_brands, "PHILHARMONIC_MODEL_BRAND_TOKENS", issues)
            except json.JSONDecodeError:
                issues.append("PHILHARMONIC_MODEL_BRAND_TOKENS is not valid JSON.")
        else:
            _apply_overrides([t.strip() for t in raw_brands.split(",") if t.strip()], "brand_tokens", extra_brands, "PHILHARMONIC_MODEL_BRAND_TOKENS", issues)

    for tier_key, env_var in TIER_ENV_VARS.items():
        if raw_val := os.environ.get(env_var):
            if _is_valid_wire_model(raw_val):
                tiers[tier_key] = raw_val.strip()
            else:
                issues.append(f"Invalid wire model value in {env_var}.")

    for wire_model in tiers.values():
        extra_brands.extend(_extract_brand_tokens_from_wire_model(wire_model))
    return tiers, aliases, tuple(sorted(set(extra_brands))), issues


def get_model_tier_map(config_path: Optional[Path] = None) -> Dict[str, str]:
    return _load_model_settings(config_path=config_path)[0]


def get_model_alias_map(config_path: Optional[Path] = None) -> Dict[str, str]:
    return _load_model_settings(config_path=config_path)[1]


def get_model_for_tier(tier: str, tier_map: Optional[Dict[str, str]] = None) -> str:
    active = tier_map if tier_map is not None else get_model_tier_map()
    return active.get(str(tier or "").strip().lower()) or active.get("standard") or DEFAULT_TIER_TO_MODEL["standard"]


def validate_model_config(config_path: Optional[Path] = None) -> List[str]:
    return _load_model_settings(config_path=config_path)[3]


def resolve_model_tiers(
    raw_model: Optional[str], tier_map: Optional[Dict[str, str]] = None, alias_map: Optional[Dict[str, str]] = None
) -> Set[str]:
    """Resolve a requested wire model or alias to the set of matching capability tiers."""
    norm = str(raw_model or "").strip().lower()
    if not norm:
        return set()
    active_tiers = tier_map if tier_map is not None else get_model_tier_map()
    if configured := {t for t, w in active_tiers.items() if t in MODEL_TIERS_SET and w.strip().lower() == norm}:
        return configured
    variants = (norm, norm.replace("-", "_"), norm.replace("_", "-"))
    if configured_sep := {t for t, w in active_tiers.items() if t in MODEL_TIERS_SET and w.strip().lower() in variants}:
        return configured_sep
    active_aliases = alias_map if alias_map is not None else get_model_alias_map()
    for cand in variants:
        if (mapped := active_aliases.get(cand)) in MODEL_TIERS_SET:
            return {mapped}
    return set()


def select_target_tier(matched_tiers: Set[str], allowed_tiers: Set[str], default_tier: str) -> str:
    """Select a deterministic capability tier from matched_tiers constrained by allowed_tiers."""
    valid = matched_tiers & allowed_tiers
    if default_tier in valid:
        return default_tier
    return next((t for t in ("standard", "lite", "frontier", "inherit") if t in valid), default_tier)


def infer_subagent_tier(
    model: Optional[str], type_name: Optional[str] = None,
    tier_map: Optional[Dict[str, str]] = None, alias_map: Optional[Dict[str, str]] = None,
) -> str:
    """Infer a subagent's canonical capability tier from its stored model and subagent type."""
    default_tier = DEFAULT_TIER_BY_TYPE.get(str(type_name or "").strip().lower(), "")
    matched = resolve_model_tiers(model, tier_map=tier_map, alias_map=alias_map)
    if len(matched) == 1:
        return next(iter(matched))
    if matched:
        return default_tier if default_tier in matched else next((t for t in ("standard", "lite", "frontier", "inherit") if t in matched), default_tier)
    return default_tier


@lru_cache(maxsize=32)
def _compile_council_name_leak_re(extra_tokens: Tuple[str, ...] = ()) -> re.Pattern[str]:
    terms = [
        *PROVIDER_BRAND_TOKENS, *CLAUDE_MODEL_CODENAMES, "carbon", "neon", "argon",
        r"hydrogen(?:[_-]lite)?", r"flash(?:[_-]lite)?", "gpt", r"o[134]",
        *(re.escape(tok) for tok in sorted(extra_tokens, key=len, reverse=True)),
    ]
    return re.compile(rf"(?:^|[^a-z])(?:{'|'.join(terms)})(?:$|[^a-z])", re.I)


@lru_cache(maxsize=32)
def _compile_blind_judge_leak_patterns(extra_tokens: Tuple[str, ...] = ()) -> Tuple[re.Pattern[str], ...]:
    extra_esc = [re.escape(tok) for tok in sorted(extra_tokens, key=len, reverse=True)]
    claude_alt = "|".join(CLAUDE_MODEL_CODENAMES)
    provider_alt = "|".join(t for t in PROVIDER_BRAND_TOKENS if t not in {"gemini", "claude", "grok"})
    google_alt = "|".join([r"hydrogen(?:[_ -]lite)?", "carbon", "neon", "argon", *extra_esc])
    unambiguous = (
        rf"(?:gemini(?:\s+[0-9]+(?:\.[0-9]+)?)?(?:[\s_-]+(?:pro|flash(?:[_ -]lite)?|{google_alt}))?"
        rf"|claude(?:\s+[0-9]+(?:\.[0-9]+)?)?(?:[\s_-]+(?:{claude_alt})(?:\s+[0-9]+(?:\.[0-9]+)?)?)?"
        rf"|(?:{claude_alt})(?:\s+[0-9]+(?:\.[0-9]+)?)?"
        r"|gpt-[0-9][a-z0-9.\-]*|o[134](?:[- ](?:mini|preview|pro))?"
        rf"|grok(?:[- ][0-9]+(?:\.[0-9]+)?)?|geminipro|claudesonnet|claudeopus|{provider_alt})"
    )
    codename = rf"(?:{google_alt})(?:\s+[0-9]+(?:\.[0-9]+)?)?"
    full_brand = rf"(?:{unambiguous}|{codename})"
    meta_brand = rf"(?:{full_brand}|flash(?:[_ -]lite)?(?:\s+[0-9]+(?:\.[0-9]+)?)?)"
    pair_alt = "|".join([*PROVIDER_BRAND_TOKENS, *CLAUDE_MODEL_CODENAMES, *GOOGLE_MODEL_CODENAMES, "flash", r"gpt(?:-[0-9a-z.]*)?", *extra_esc])
    solo_alt = "|".join([*PROVIDER_BRAND_TOKENS, *CLAUDE_MODEL_CODENAMES, "carbon", "neon", "argon", r"hydrogen(?:[_-]lite)?", r"flash(?:[_-]lite)?", r"gpt(?:-[0-9a-z.]*|[0-9]+[a-z0-9._-]*)?", r"o[134]", *extra_esc])
    subj = r"(?:proposal|draft|option|candidate|design|submission|response|answer)"
    fsubj = r"(?:proposal|draft|option|candidate|design|solution|submission|response|answer|report|output|plan)"
    return (
        re.compile(rf"\b{subj}\s+[a-z0-9#_-]{{1,20}}[^\r\n]{{0,80}}(?:\(\s*|:\s*|—\s*|-\s*|\bby\s+|\bfrom\s+){full_brand}\b", re.I),
        re.compile(rf"\b{subj}\s+[a-z0-9#_-]{{1,20}}.{{0,60}}?(?:(?:\(\s*|:\s*|—\s*|-\s*|\bby\s+|\bfrom\s+)\b{full_brand}\b|\r?\n[ \t]*(?:{unambiguous}\b|{codename}(?=[ \t]*(?:\r?\n|$))))", re.I | re.S),
        re.compile(rf"\b{full_brand}\b[^\r\n]{{0,60}}\b(?:wrote|authored|proposed|generated|created|produced|submitted)\b[^\r\n]{{0,60}}\b{subj}\b", re.I),
        re.compile(rf"\b{full_brand}(?:'s|s')\s+{subj}\b", re.I),
        re.compile(rf"(?m)(?:^[ \t]*(?:[-*#]+[ \t]*)?|(?:^|[{{,])[ \t]*)[\"']?(?:model|author|authored[ \t]+by|written[ \t]+by|generated[ \t]+by|proposed[ \t]+by|submitted[ \t]+by|source[ \t]+model|provider|llm)[\"']?[ \t]*:[ \t]*[^\r\n]*\b{meta_brand}\b", re.I),
        re.compile(rf"\[\s*(?:{meta_brand}|gpt)\s*\]", re.I),
        re.compile(rf"\b(?:{fsubj}[a-z0-9._/\-]*?(?:{pair_alt})|(?:{pair_alt})[a-z0-9._/\-]*?{fsubj})[a-z0-9._/\-]*\.(?:md|txt|json)\b", re.I),
        re.compile(rf"(?:^|[/\\\s`'\"(,:])(?:{solo_alt})(?:[_-][a-z0-9._-]*)?\.(?:md|txt|json)\b", re.I),
        re.compile(r"(?:^|[\s`'\"(,:])(?:[A-Za-z0-9._/\-]*/)?scratch/(?![A-Za-z0-9._\-]+\.(?:md|txt|json)\b)", re.I),
    )


def get_council_name_leak_re(config_path: Optional[Path] = None) -> re.Pattern[str]:
    return _compile_council_name_leak_re(_load_model_settings(config_path=config_path)[2])


def get_blind_judge_leak_patterns(config_path: Optional[Path] = None) -> Tuple[re.Pattern[str], ...]:
    return _compile_blind_judge_leak_patterns(_load_model_settings(config_path=config_path)[2])


COUNCIL_NAME_LEAK_RE = _compile_council_name_leak_re()
BLIND_JUDGE_LEAK_PATTERNS = _compile_blind_judge_leak_patterns()
