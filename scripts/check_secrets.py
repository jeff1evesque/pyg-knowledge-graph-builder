#!/usr/bin/env python3
"""Reject staged changes containing a committed secret.

Reads the custom rules out of ``.gitleaks.toml`` and applies them with the
standard library alone, so the hook works in a fresh clone with nothing
installed. That is the same bargain ``check_unicode.py`` makes: CI is the
enforcement, and this only fails faster.

It is deliberately a SUBSET of what CI runs. The ``gitleaks`` binary carries
~150 provider rules and scans full history; reimplementing that here would be a
second source of truth that drifts from the first. What this catches is the
custom rules -- the ones written for this stack, where a miss costs the most --
plus private keys, read from the same file CI reads so the two cannot disagree
about them.

Notebook OUTPUTS are scanned like any other content and are the reason this
matters: a stored CDK bootstrap line published an AWS account id, and nobody
reads an output cell in a diff.

Python's ``re`` is a superset of the RE2 engine gitleaks uses, so every pattern
that compiles there compiles here.

Usage:
    python scripts/check_secrets.py              # scan everything tracked
    python scripts/check_secrets.py FILE...      # scan specific files
    python scripts/check_secrets.py --staged     # scan staged files (hooks)
"""

import json
import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - python < 3.11
    tomllib = None

CONFIG = ".gitleaks.toml"

#: Always checked, config or not. A private key is unambiguous and is the one
#: finding that must never depend on a config file being present or parseable.
PRIVATE_KEY = re.compile(
    r"-----BEGIN (?:RSA|OPENSSH|DSA|EC|PGP) PRIVATE KEY( BLOCK)?-----"
)


def _repo_root():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        )
        return Path(out.stdout.strip())
    except Exception:  # noqa: BLE001 - not a repo is not an error here
        return Path.cwd()


def _staged_files():
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        capture_output=True, text=True,
    )
    return [f for f in out.stdout.split("\n") if f]


def _tracked_files():
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
    return [f for f in out.stdout.split("\n") if f]


def load_rules(root):
    """``(rules, path_allowlists, global_allowlists)`` from the shared config.

    A config this cannot parse yields no custom rules rather than an error: the
    private-key check below still runs, and CI still applies the full ruleset.
    Failing the commit because a TOML dialect moved would train people to pass
    --no-verify, which costs more than it saves.
    """
    if tomllib is None:
        return [], [], []

    path = root / CONFIG
    if not path.is_file():
        return [], [], []

    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"Notice: {CONFIG} not parsed ({exc}); running the private-key check only.")
        return [], [], []

    rules = []
    for rule in data.get("rules", []):
        try:
            pattern = re.compile(rule["regex"])
        except (KeyError, re.error):
            continue
        allow = []
        for entry in _allowlist_entries(rule.get("allowlist")):
            allow.extend(_compiled(entry.get("regexes")))
        rules.append((rule.get("id", "rule"), pattern, allow,
                      rule.get("secretGroup"), _compiled(
                          [rule["path"]] if rule.get("path") else [])))

    paths, globals_ = [], []
    for entry in _allowlist_entries(data.get("allowlists") or data.get("allowlist")):
        paths.extend(_compiled(entry.get("paths")))
        globals_.extend(_compiled(entry.get("regexes")))

    return rules, paths, globals_


def _allowlist_entries(value):
    if not value:
        return []
    return value if isinstance(value, list) else [value]


def _compiled(patterns):
    out = []
    for p in patterns or []:
        try:
            out.append(re.compile(p))
        except re.error:
            continue
    return out


def scan(text, rules, global_allow, relative):
    """Findings in one file, as ``(rule_id, line_number, line)``."""
    findings = []
    lines = text.splitlines()

    for number, line in enumerate(lines, 1):
        if PRIVATE_KEY.search(line):
            findings.append(("private-key", number, line))

    for rule_id, pattern, allow, group, path_res in rules:
        if path_res and not any(p.search(relative) for p in path_res):
            continue
        for match in pattern.finditer(text):
            secret = match.group(group) if group and match.groups() else match.group(0)
            number = text.count("\n", 0, match.start()) + 1
            line = lines[number - 1] if number - 1 < len(lines) else ""
            if any(a.search(secret) for a in allow):
                continue
            if any(a.search(line) for a in allow):
                continue
            if any(a.search(line) for a in global_allow):
                continue
            findings.append((rule_id, number, line))

    return findings


def readable(path):
    """File text, with notebook outputs included, or None if binary."""
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None

    if path.suffix == ".ipynb":
        # Read as json so an output cell's text is scanned as text rather than
        # as escaped json, which no regex written for source would match.
        try:
            book = json.loads(text)
        except ValueError:
            return text
        parts = []
        for cell in book.get("cells", []):
            parts.extend(cell.get("source", []))
            for output in cell.get("outputs", []):
                parts.extend(output.get("text", []))
                for value in (output.get("data") or {}).values():
                    if isinstance(value, list):
                        parts.extend(str(v) for v in value)
                    elif isinstance(value, str):
                        parts.append(value)
        return "\n".join(parts)

    return text


def main(argv):
    root = _repo_root()
    staged = "--staged" in argv
    names = [a for a in argv if not a.startswith("-")]

    if names:
        files = names
    elif staged:
        files = _staged_files()
    else:
        files = _tracked_files()

    rules, path_allow, global_allow = load_rules(root)

    ignored = set()
    ignore_file = root / ".gitleaksignore"
    if ignore_file.is_file():
        ignored = {
            line.strip() for line in ignore_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }

    failures = 0
    scanned = 0
    for name in files:
        if any(p.search(name) for p in path_allow):
            continue
        path = root / name
        if not path.is_file():
            continue
        text = readable(path)
        if text is None:
            continue
        scanned += 1
        for rule_id, number, line in scan(text, rules, global_allow, name):
            #
            # a fingerprint recorded in .gitleaksignore has been reviewed. Line
            # numbers move, so the path and rule are enough to match on here --
            # CI does the exact-fingerprint check against history.
            #
            if any(name in entry and rule_id in entry for entry in ignored):
                continue
            failures += 1
            shown = line.strip()
            if len(shown) > 100:
                shown = shown[:100] + "..."
            print(f"{name}:{number}: {rule_id}")
            print(f"    {shown}")

    if failures:
        print(f"\nFAIL: {failures} possible secret(s) in {scanned} file(s).")
        print("A secret is compromised once pushed -- rotate it rather than")
        print("only removing it. If this is a false positive, add the value's")
        print(f"shape to an allowlist in {CONFIG}.")
        return 1

    print(f"OK: no secrets detected in {scanned} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
