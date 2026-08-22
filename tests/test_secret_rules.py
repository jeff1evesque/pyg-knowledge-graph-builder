"""The scanner rules must compile for gitleaks, not merely for Python.

WHY THIS FILE EXISTS. `.gitleaks.toml` is read by two engines: Python's `re`
(scripts/check_secrets.py, at commit time) and RE2 (the gitleaks binary, in CI).
Python's `re` is a SUPERSET. A rule using a lookahead compiles locally, passes
the hook on every commit, and then panics gitleaks:

    panic: regexp: Compile(...): error parsing regexp: bad perl operator: `(?=`
    Error: Process completed with exit code 2

Exit 2 means no scan ran at all, so the branch is not protected -- a broken rule
is worse than a missing one. That happened, which is why this is a test rather
than a comment.
"""
import re
import tomllib
from pathlib import Path

import pytest

CONFIG = Path(__file__).resolve().parents[1] / ".gitleaks.toml"

# RE2 supports none of these. Ordered as they appear in the error message.
RE2_UNSUPPORTED = {
    r"(?=": "lookahead",
    r"(?!": "negative lookahead",
    r"(?<=": "lookbehind",
    r"(?<!": "negative lookbehind",
}


def _patterns():
    """Every regex in the config, as (label, pattern)."""
    cfg = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    for rule in cfg.get("rules", []):
        rid = rule.get("id", "<unnamed>")
        for field in ("regex", "path"):
            if field in rule:
                yield f"{rid}.{field}", rule[field]
        allow = rule.get("allowlist", {})
        for i, rx in enumerate(allow.get("regexes", [])):
            yield f"{rid}.allowlist[{i}]", rx
    for i, allow in enumerate(cfg.get("allowlists", [])):
        for j, rx in enumerate(allow.get("regexes", [])):
            yield f"allowlists[{i}].regexes[{j}]", rx
        for j, rx in enumerate(allow.get("paths", [])):
            yield f"allowlists[{i}].paths[{j}]", rx


def test_the_config_parses():
    assert list(_patterns()), "no patterns found; the config did not parse"


@pytest.mark.parametrize("label,pattern", list(_patterns()))
def test_every_pattern_compiles_under_python(label, pattern):
    """The local hook has to be able to run it."""
    re.compile(pattern)


@pytest.mark.parametrize("label,pattern", list(_patterns()))
def test_no_pattern_uses_a_construct_re2_lacks(label, pattern):
    """The CI scanner has to be able to run it.

    Express the constraint inside the capture group, or split the rule in two.
    A lookahead here takes CI down rather than making it strict.
    """
    for token, name in RE2_UNSUPPORTED.items():
        assert token not in pattern, (
            f"{label} uses a {name} ({token!r}), which RE2 rejects. It will "
            f"compile here and panic gitleaks in CI with exit 2, leaving the "
            f"branch unscanned. See this module's docstring."
        )


@pytest.mark.parametrize("label,pattern", list(_patterns()))
def test_no_pattern_uses_a_backreference(label, pattern):
    """RE2 has no backreferences either.

    Matched as a digit escape outside a character class. `\\d` and `\\1` differ
    only by the character, so this checks the digit specifically.
    """
    stripped = re.sub(r"\[[^\]]*\]", "", pattern)
    assert not re.search(r"(?<!\\)\\[1-9]", stripped), (
        f"{label} appears to use a backreference, which RE2 rejects"
    )
